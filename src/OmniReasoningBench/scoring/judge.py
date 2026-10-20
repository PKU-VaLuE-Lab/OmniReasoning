"""Score OmniReasoningBench generations with the judge protocol.

  * mcq            -> rule-based option-letter match (deterministic); an output with
                      no parseable letter is scored wrong
  * open-ended     -> deepseek-v4-flash; open_numeric uses a numeric fallback only
                      if the judge fails to return a verdict (--no_rule_fallback disables it)

The judge is any OpenAI-compatible chat endpoint: --judge_api_base / JUDGE_API_BASE
(falls back to OPENAI_API_BASE) and --judge_api_key / JUDGE_API_KEY (falls back to
OPENAI_API_KEY).

Output (under --out_dir):
  predictions.jsonl   one row per item (raw output, extracted answer, correct, method)
  summary.json        overall + per-task_type / per-format accuracy

Rows whose generation is missing or empty (including provider/service errors) are
recorded with method "error" and scored wrong; every reference item stays in the
denominator, as do unresolved judge failures (method "judge_failed"). Numeric fallbacks
are recorded separately as "rule_numeric_fallback", with the original judge error.

usage: omnireasoning score-judge --predictions OUT.jsonl --references BENCH.jsonl --out_dir DIR
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .numeric import _numeric_correct


JUDGE_MODEL = "deepseek-v4-flash"


class ModelError(Exception):
    pass


_FINAL_RE = re.compile(r"final\s*answer\s*[:：]\s*(.+)", re.IGNORECASE)

# Thinking checkpoints put the chain of thought in <think>…</think> and may then commit
# to a BARE answer after it ("</think>\n\nC"), ignoring the prompt's "Final Answer:"
# instruction. Scoring the whole string then reads letters out of the reasoning instead
# of the answer: extract_pred_letters falls through to "last 'X)' anywhere in the text",
# which picks up option references the model was *rejecting*. Same trap for the judge
# path, where a "Final Answer: X" the model quoted while planning inside the CoT wins.
# Only the committed segment is scored, for every checkpoint by default.
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\s*>", re.IGNORECASE)


def committed_answer(text: str) -> str:
    """The part of a thinking model's output that is actually its answer.

    Returns "" when the CoT opened but never closed (max_tokens hit mid-reasoning):
    the model committed to nothing, and falling back to the full text would put the
    CoT-letter bug straight back.
    """
    text = text or ""
    if _THINK_CLOSE_RE.search(text):
        text = _THINK_CLOSE_RE.split(text)[-1]
    if _THINK_OPEN_RE.search(text):
        return ""
    return text.strip()


def final_answer_line(text: str) -> str:
    matches = _FINAL_RE.findall(text or "")
    if matches:
        return matches[-1].strip().splitlines()[0].strip()
    # fall back to the last non-empty line
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return lines[-1] if lines else ""


_CONNECTORS = {"AND", "OR"}


def _bare_letter_list(seg: str, valid: set[str]) -> list[str]:
    """['B','C'] iff seg is nothing but option letters + separators ("B", "B and C").

    Requires *every* alphabetic token to be a single valid letter (or and/or), so
    restated option prose can never come through here.
    """
    toks = re.findall(r"[A-Za-z]+", seg)
    single = [t.upper() for t in toks if len(t) == 1 and t.upper() in valid]
    if single and all(len(t) == 1 or t.upper() in _CONNECTORS for t in toks):
        return sorted(set(single))
    return []


_LABELLED_LETTER_RE = re.compile(
    r"^(?:the\s+)?(?:correct\s+|right\s+)?(?:option|choice|answer|选项|答案)\s*(?:is\s+)?[:：]?\s*"
    r"\*{0,2}([A-Za-z])\*{0,2}\.?$", re.IGNORECASE)


def _labelled_letter(seg: str, valid: set[str]) -> list[str]:
    """['A'] for "Option A" / "Answer: A" / "答案 A" — a labelled but still committed letter.

    An option body can itself contain a label ("A) Arrow B", "E) Arrow A").
    "Final Answer: Option A" commits to option A regardless of letters in its text.
    """
    mo = _LABELLED_LETTER_RE.match(seg.strip())
    if mo and mo.group(1).upper() in valid:
        return [mo.group(1).upper()]
    return []


def extract_pred_letters(text: str, valid: set[str]) -> list[str]:
    """Extract the option letter(s) a model committed to (strict).

    Prefers the explicit 'Final Answer:' line; only treats it as a letter list
    when it is actually letters + separators (not restated option prose).
    Falls back to the last 'X)' commitment anywhere in the text.
    """
    text = committed_answer(text)
    matches = _FINAL_RE.findall(text)
    if matches:
        seg = matches[-1].splitlines()[0].strip()
        # 1) "X)" forms inside the final-answer segment
        phits = [m.group(1).upper() for m in re.finditer(r"([A-Za-z])\)", seg)
                 if m.group(1).upper() in valid]
        if phits:
            return sorted(set(phits))
        # 2) the segment is a bare letter list ("B", "B, C", "B and C")
        bare = _bare_letter_list(seg, valid)
        if bare:
            return bare
        # 2a) "Option A" / "choice A" / "答案 A" — a labelled letter is still a committed
        #     letter (see _labelled_letter).
        lab = _labelled_letter(seg, valid)
        if lab:
            return lab
    else:
        # 2b) No "Final Answer:" line, but the committed output is only letters.
        # Thinking checkpoints often put a bare "C" after </think>.
        bare = _bare_letter_list((text or "").strip(), valid)
        if bare:
            return bare
        lab = _labelled_letter((text or "").strip(), valid)
        if lab:
            return lab
    # 3) last "X)" commitment anywhere in the output
    hits = [m.group(1).upper() for m in re.finditer(r"\b([A-Za-z])\)", text or "")
            if m.group(1).upper() in valid]
    return [hits[-1]] if hits else []



def llm_judge(question: str, gold: str, model_answer: str, *, judge_model: str,
              timeout: int, max_retries: int,
              judge_api_base: str = "", judge_api_key: str = "") -> tuple[bool, str]:
    """Ask the benchmark's open-ended judge; raise if no verdict can be obtained."""
    if judge_model != JUDGE_MODEL:
        raise ValueError(f"The benchmark judge must be {JUDGE_MODEL}")
    prompt = (
        "You are grading an answer to a question about a video. "
        "Decide whether the model's answer matches the reference answer in meaning and value. "
        "Accept equivalent wording and mathematically equivalent quantities, accounting for units. "
        "Reject answers with an incorrect value, meaning, or contradictory conclusion.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference (correct) answer:\n{gold}\n\n"
        f"Model's answer:\n{model_answer}\n\n"
        "Reply with exactly one word: CORRECT or INCORRECT."
    )
    url = f"{judge_api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {judge_api_key}"}
    payload = {"model": judge_model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 4096}
    last = ""
    for attempt in range(max(1, max_retries)):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code != 200:
                last = f"judge HTTP {r.status_code}: {r.text[:200]}"
            else:
                msg = r.json()["choices"][0]["message"]
                txt = (msg.get("content") or "").strip()
                if not txt:
                    # deepseek-v4-* are thinking models: when the budget goes to reasoning
                    # the verdict lands in reasoning_content and `content` is "". Falling
                    # through with "" would make `"CORRECT" in ""` False and silently score
                    # the item INCORRECT — which would sink every open_numeric row, since
                    # those all go through the judge. Read the reasoning, else retry.
                    txt = (msg.get("reasoning_content") or "").strip()
                if not txt:
                    last = "judge empty content (and empty reasoning_content)"
                else:
                    tail = txt.splitlines()[-1].strip().strip("*` ")
                    verdict = re.fullmatch(r"(?:(?:VERDICT|ANSWER)\s*[:：]\s*)?"
                                           r"(INCORRECT|CORRECT)[.!。]?", tail, re.IGNORECASE)
                    if verdict:
                        return verdict.group(1).upper() == "CORRECT", txt
                    last = f"judge returned no parseable verdict: {txt[:200]}"
        except Exception as e:  # noqa: BLE001
            last = f"judge error: {e}"
        if attempt + 1 < max(1, max_retries):
            time.sleep(1.5 * (attempt + 1))
    # The caller records a numeric fallback or an unresolved judge_failed result.
    raise ModelError(f"judge failed: {last}")


# ─────────────────────────────── scoring ────────────────────────────────────
def score_item(item: dict, raw_text: str, *, judge_model: str, timeout: int,
               max_retries: int, judge_api_base: str = "", judge_api_key: str = "",
               rule_fallback: bool = True) -> dict:
    fmt = item["answer_format"]
    raw_text = committed_answer(raw_text)
    if not raw_text:
        return {"method": "no_answer_truncated", "pred": "", "correct": False,
                "gold": ",".join(sorted(set(item.get("answer_letters") or [])))
                        or item.get("answer", "")}
    pred_final = final_answer_line(raw_text)
    if fmt == "mcq":
        valid = {re.match(r"\s*([A-Za-z])", o).group(1).upper()
                 for o in item["options"] if re.match(r"\s*([A-Za-z])", o)}
        pred_letters = extract_pred_letters(raw_text, valid)
        gold = sorted(set(item.get("answer_letters") or []))
        return {"method": "rule_mcq", "pred": ",".join(pred_letters),
                "gold": ",".join(gold), "correct": bool(pred_letters) and pred_letters == gold}
    # Always ask the LLM first, even when the numeric strings are identical.
    try:
        correct, verdict = llm_judge(item["question"], str(item["answer"]), pred_final or raw_text,
                                    judge_model=judge_model, timeout=timeout, max_retries=max_retries,
                                    judge_api_base=judge_api_base, judge_api_key=judge_api_key)
    except ModelError as exc:
        if not rule_fallback or fmt != "open_numeric":
            raise
        return {"method": "rule_numeric_fallback", "pred": pred_final, "gold": item["answer"],
                "correct": _numeric_correct(str(item["answer"]), pred_final or raw_text),
                "judge_model": judge_model, "judge_error": str(exc)[:300]}
    return {"method": "llm_judge", "pred": pred_final, "gold": item["answer"],
            "correct": correct, "judge_verdict": verdict, "judge_model": judge_model}


def summarize(rows: list[dict]) -> dict:
    def acc(subset):
        n = len(subset)
        c = sum(1 for r in subset if r.get("correct"))
        return {"n": n, "correct": c, "accuracy": round(c / n, 4) if n else 0.0}

    by_type: dict[str, list] = {}
    by_fmt: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["task_type"], []).append(r)
        by_fmt.setdefault(r["answer_format"], []).append(r)
    errors = sum(1 for r in rows if r.get("method") == "error")
    no_answer = sum(1 for r in rows if r.get("method") == "no_answer_truncated")
    judge_failed = sum(1 for r in rows if r.get("method") == "judge_failed")
    fallback = [r for r in rows if r.get("method") == "rule_numeric_fallback"]
    out = {
        "no_answer_truncated": no_answer,
        "judge_failed": judge_failed,
        "numeric_fallback": acc(fallback),
        "overall": acc(rows),
        "errors": errors,
        "by_answer_format": {k: acc(v) for k, v in sorted(by_fmt.items())},
        "by_task_type": {k: acc(v) for k, v in sorted(by_type.items(), key=lambda x: -len(x[1]))},
    }
    # Figure rows vs text-only rows are different difficulty regimes — split them.
    if any("sent_image" in r for r in rows):
        out["by_figure"] = {
            "with_figure": acc([r for r in rows if r.get("sent_image")]),
            "text_only": acc([r for r in rows if not r.get("sent_image")]),
        }
    return out


def read_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in fh if line.strip()]
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def raw_text_of(pred: dict) -> str:
    if "raw_output" in pred:
        return pred.get("raw_output") or ""
    gen = pred.get("gen")
    if isinstance(gen, list):
        gen = gen[0] if gen else ""
    return gen or ""


def process_one(item: dict, pred: dict | None, args) -> dict:
    row = {"id": item["id"], "answer_format": item["answer_format"],
           "task_type": item["task_type"], "source": item.get("source", ""),
           "sent_image": bool((pred or {}).get("image_path") or (pred or {}).get("sent_image"))}
    raw = raw_text_of(pred) if pred else ""
    row["raw_output"] = raw
    if not raw.strip() or (pred or {}).get("error"):
        row["error"] = (pred or {}).get("error") or ("no generation" if pred else "missing prediction")
        row["correct"] = False
        row["method"] = "error"
        return row
    scored_text = committed_answer(raw)
    row["answer_segment"] = scored_text
    if not scored_text:
        row.update({"method": "no_answer_truncated", "pred": "",
                    "gold": ",".join(sorted(set(item.get("answer_letters") or [])))
                            or item.get("answer", ""),
                    "correct": False})
        return row
    try:
        sc = score_item(item, scored_text, judge_model=args.judge_model, timeout=args.timeout,
                        max_retries=args.max_retries, judge_api_base=args.judge_api_base,
                        judge_api_key=args.judge_api_key, rule_fallback=args.rule_fallback)
    except Exception as e:  # noqa: BLE001
        # An unresolved judge failure counts wrong and is recorded separately.
        row.update({"method": "judge_failed", "pred": "", "correct": False,
                    "gold": item.get("answer", ""), "error": str(e)[:300]})
        return row
    row.update(sc)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="generations (JSONL / JSON list) keyed by id, with `gen` or `raw_output`")
    ap.add_argument("--references", required=True, help="benchmark records (JSONL / JSON list)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--judge_model", choices=[JUDGE_MODEL],
                    default=os.environ.get("JUDGE_MODEL") or JUDGE_MODEL)
    ap.add_argument("--no_rule_fallback", action="store_false", dest="rule_fallback",
                    help="count judge failures wrong instead of using the numeric fallback")
    ap.add_argument("--judge_api_base",
                    default=os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE", ""))
    ap.add_argument("--judge_api_key",
                    default=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--answer_after_think", action="store_true",
                    help="compatibility flag: scoring always excludes <think> content; an unclosed CoT is no_answer_truncated")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only score first N items (0 = all)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max_retries", type=int, default=3)
    args = ap.parse_args()
    if args.judge_model != JUDGE_MODEL:
        ap.error(f"JUDGE_MODEL must be {JUDGE_MODEL}")
    if not args.judge_api_base:
        ap.error("set --judge_api_base or JUDGE_API_BASE")

    items = read_records(args.references)
    if args.limit:
        items = items[: args.limit]
    preds: dict[str, dict] = {}
    for r in read_records(args.predictions):
        preds.setdefault(str(r["id"]), r)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    print(f"judge={args.judge_model}  total={len(items)}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex, pred_path.open("w", encoding="utf-8") as fh:
        futs = {ex.submit(process_one, it, preds.get(str(it["id"])), args): it for it in items}
        for n, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            results.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            flag = "OK " if row.get("correct") else ("ERR" if row.get("method") == "error" else "x  ")
            print(f"[{n}/{len(items)}] {flag} {row['id']} ({row.get('method')})")

    summary = summarize(results)
    summary["judge_model"] = args.judge_model
    summary["rule_fallback_enabled"] = args.rule_fallback
    summary["n_items"] = len(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary["overall"]))
    print(f"numeric fallback: {summary['numeric_fallback']['n']} items")
    print(f"errors {summary['errors']} and judge_failed {summary['judge_failed']} are counted wrong"
          " (all reference items in the denominator)")
    print(f"\npredictions -> {pred_path}\nsummary     -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
