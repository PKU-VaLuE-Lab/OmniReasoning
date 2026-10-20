"""Stage 4 · Think-process synthesis (text-only, DeepSeek).

From each accepted question (accepted.jsonl produced by Stage 2, or the verified /
refined pool from Stage 3), synthesize a first-person "watching the video" chain of
thought and emit an SFT training row:

    assistant = "<think>\n{thinking}\n</think>\n\n{gold_letter}"

INPUT to the LLM is TEXT ONLY — no video is sent:
    • the question + options
    • the "key moments" built from candidate.reasoning_steps + audio/visual clues
    • the full caption bundle (summary + per-segment see/hear with timestamps)
    • the verified gold letter (candidate.correct_option) — ANCHORED: the model
      reconstructs the honest reasoning that LANDS ON that gold, it does not re-derive
      a (possibly different) answer.

Inputs follow the Omni-QA layout:
  • caption bundles at <caption-root>/<dataset>/raw/<video_id>/caption_bundle.json
  • gold = candidate.correct_option (Stage 2/3 already verify & repair it, so the
    gold is trustworthy and safe to anchor on).

The synthesized think obeys strict grounding rules (only timestamps present in the
content, no fabricated details, forbidden pipeline vocabulary). Rows that fail the
hard QC or emit the [UNGROUNDED] sentinel are quarantined out of the SFT pool.

Concurrency + resume: httpx in a thread pool, RateLimiter, append-jsonl resume keyed
by (source, video_id, row idx).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob as _glob
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

WRITE_LOCK = threading.Lock()
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNGROUNDED = "[UNGROUNDED]"

ROLE_DEF = (
    "# Role Definition\n\nYou are a professional video reasoning assistant. "
    "Analyze the provided video, including visual and audio information when "
    "available, and answer the user's question accurately."
)

# Default: ./data/caption relative to the working directory (Stage 1 output layout).
DEFAULT_CAPTION_ROOT = Path("data") / "caption"

# ══════════════════════════ Anchored prompt ══════════════════════════

PROMPT_TEMPLATE = """You are a professional video reasoning model. You have just watched a video (its visuals and its audio) and you are writing your own private thinking before answering a multiple-choice question. Write in the first person as someone who genuinely watched and listened to this video.

The correct answer to this question is option {gold}. Your task is NOT to decide the answer — it is already established as option {gold}. Your task is to reconstruct the honest first-person chain of watching/listening that arrives at option {gold}, using ONLY what the video content below actually states. Your conclusion MUST be option {gold}.

Follow this reasoning PATTERN (it is how a strong solver thinks; do not label the steps, just let the thinking flow through them in order):
- First, break down what the question is really asking: what target entity or moment it pins down, what has to be found, and what could be confused among the options.
- Then locate that target in the video: find the referent by what is seen and/or heard, and lock onto the exact time window.
- Then carry out the cross-modal, multi-hop chain at concrete timestamps: connect what you see with what you hear to arrive at option {gold}.
- Then go through the options and, crucially, EXPLAIN EACH WRONG OPTION by tying it to what it actually corresponds to elsewhere in the video (a different entity, a different moment, a different quantity, or a mishearing/misreading) — but ONLY when the video content below actually contains that other thing. If the content does not say what a wrong option corresponds to, just state plainly that nothing in the video supports it; do NOT invent a source for it.
- Finally, conclude naturally with option {gold}.

The thinking MUST use exactly these two standalone transition lines, in this order:
  Let me carefully watch the video:
  Now let me carefully analyze the options:
The query breakdown comes before the first line; the video reasoning between the two lines; the option analysis and conclusion after the second line.

Hard requirements:
- Output ONLY the thinking text. No JSON, no headings, no bullet points, no numbering. The only structured separators are the two exact transition lines above.
- Start exactly with: Got it!
- Do NOT wrap the text in <think> tags.
- You are watching the video. NEVER reveal or imply that any notes, timestamps, captions, summaries, reasoning steps, labels, metadata, a dataset, a prompt, or a correct answer were supplied to you. Everything you state must read as your own direct observation. In particular, never say that the answer "is given", "is established", "is option {gold}" as if told — arrive at it through observation.
- ABSOLUTELY FORBIDDEN phrasing (these break the illusion of watching): "visual-only", "audio-only", "visual only", "audio only", "priors", "a priors guess", "single modality", "single-modality", "modality", "the visual job", "the audio job", "visual track", "audio track", "shortcut", "trap", "distractor", "clue", "evidence", "reasoning step", "caption", "annotation", "metadata", "the correct answer is", "the gold answer".
- When ruling out an option, describe it as content that belongs to another moment/entity, a different quantity, or a mishearing/misreading of what actually happened — never as "what a X-only solver would pick".
- Every timestamp must be introduced with exact "At ..." phrasing, e.g. "At 01:23-01:26, I see...". Never use vague words like around, about, near, roughly, approximately, or circa.
- ALWAYS write timestamps in plain MM:SS form (minutes:seconds). NEVER include milliseconds, fractional seconds, or an extra ":xxx"/".xxx" part. If the video content below shows a fractional value, drop the fraction and keep only MM:SS.
- Every "At" timestamp must EXACTLY match one of the timestamps listed in the video content below. Do not invent, round, shift, merge, or approximate timestamps. If you cannot find a timestamp for something, do not state that something at all.

ANTI-FABRICATION (the single most important rule):
- Ground EVERY concrete detail — every colour, count, number, name, on-screen text, object, action, spoken line, and sound — in an explicit statement in the video content below. If a detail is not written in the video content below, you MUST NOT mention it. Do not "fill in" plausible colours, counts, or fine-grained specifics from imagination or general knowledge.
- Do NOT contradict the video content below. Do NOT upgrade a vague description into a precise one.
- Quote or closely paraphrase the exact spoken line / on-screen detail from the content that decides the answer.

GROUNDING CHECK (do this silently before writing):
- Verify that the video content below genuinely supports option {gold} for the asked target.
- If — and only if — after honest reading the video content does NOT support option {gold} (it is silent on the deciding detail, or it actually supports a different option), DO NOT fabricate a justification. Instead output a single line consisting of exactly:
{ungrounded}
  followed by one sentence naming which option the content actually supports (or that it is silent). Do not output the normal thinking in that case.

- Otherwise (the content supports option {gold}), write the full first-person thinking as specified above and conclude: "This points to option {gold}."
- Keep it high quality and thorough: usually 14-20 sentences, showing the full referent-binding, time-window, cross-modal, and per-option grounding chain — but never at the cost of inventing a single ungrounded detail.

Question and options:
{question_text}

The key moments most relevant to the target (express as your own first-person seeing/hearing):
{moments_text}

Full video content you observed, for grounding every option (summary, then per-segment what is seen and heard with timestamps; the timestamps here and above are the only timestamps you may write):
{caption_text}
"""

# ══════════════════════════ Timestamp normalisation ══════════════════════════


def normalize_ts(v: str) -> str:
    return re.sub(r"\s*-\s*", "-", str(v or "").strip())


def _one_mmss(ts: str) -> str:
    ts = re.sub(r"\.\d+$", "", ts.strip())
    parts = [p for p in ts.split(":") if p != ""]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return ts
    if len(nums) == 3:
        h, m, s = nums
        return f"{h * 60 + m:02d}:{s:02d}"
    if len(nums) == 2:
        m, s = nums
        return f"{m:02d}:{s:02d}"
    return ts


def to_mmss(v: str) -> str:
    return "-".join(_one_mmss(p) for p in normalize_ts(v).split("-"))


_TS_TOKEN = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?")


def mmss_text(text: str) -> str:
    return _TS_TOKEN.sub(lambda m: _one_mmss(m.group(0)), text)


def _ts_variants(ts: str) -> set[str]:
    norm = normalize_ts(ts)
    return {norm, re.sub(r"(\d{1,2}:\d{2})\.\d+", r"\1", norm), to_mmss(ts)}


# ══════════════════════════ Caption bundle (Omni-QA layout) ══════════════════


def load_caption_bundle(video_id: str, caption_root: Path, source: str = "") -> dict[str, Any] | None:
    """Find <caption_root>/<source>/raw/<video_id>/caption_bundle.json, falling back
    to a glob over every dataset dir (Stage 1 layout)."""
    if not video_id:
        return None
    src = str(source or "").split("::")[0]  # strip Stage 2 path-hash disambiguator
    candidates: list[Path] = []
    if src:
        candidates.append(Path(caption_root) / src / "raw" / video_id / "caption_bundle.json")
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    for hit in _glob.glob(str(Path(caption_root) / "*" / "raw" / video_id / "caption_bundle.json")):
        try:
            return json.loads(Path(hit).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def build_caption_text(bundle: dict[str, Any] | None, max_chars: int = 14000) -> str:
    if not isinstance(bundle, dict):
        return "Not available."
    lines: list[str] = []
    if bundle.get("whole_video_summary"):
        lines.append(f"Overall: {bundle['whole_video_summary']}")
    for seg in bundle.get("event_segments") or []:
        rng = to_mmss(f"{seg.get('start', '')}-{seg.get('end', '')}")
        lines.append(f"[{rng}] {seg.get('summary', '')}")
    for ev in bundle.get("event_captions") or []:
        for a in ev.get("audio_captions") or []:
            if a.get("timestamp"):
                lines.append(f"[{to_mmss(a['timestamp'])}] HEAR: {a.get('caption', '')}")
        for v in ev.get("visual_captions") or []:
            if v.get("timestamp"):
                lines.append(f"[{to_mmss(v['timestamp'])}] SEE: {v.get('caption', '')}")
    text = "\n".join(ln for ln in lines if ln.strip())
    return text[:max_chars] if text else "Not available."


def build_moments_text(candidate: dict[str, Any]) -> str:
    lines: list[str] = []
    ac = candidate.get("audio_clue") or {}
    vc = candidate.get("visual_clue") or {}
    if vc.get("timestamp"):
        lines.append(f"[{to_mmss(vc['timestamp'])}] SEE: {vc.get('quote') or vc.get('fact')}")
    if ac.get("timestamp"):
        lines.append(f"[{to_mmss(ac['timestamp'])}] HEAR: {ac.get('quote') or ac.get('fact')}")
    for st in candidate.get("reasoning_steps") or []:
        ts = to_mmss(st.get("timestamp", ""))
        mod = (st.get("modality") or "").upper()
        verb = {"VISUAL": "SEE", "AUDIO": "HEAR", "AUDIOVISUAL": "SEE+HEAR"}.get(mod, mod or "OBSERVE")
        lines.append(f"[{ts}] {verb}: {st.get('content', '')}")
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return "\n".join(out)


def collect_allowed_timestamps(candidate: dict[str, Any], bundle: dict[str, Any] | None) -> set[str]:
    ts: set[str] = set()
    for src in (candidate.get("audio_clue"), candidate.get("visual_clue")):
        if isinstance(src, dict) and src.get("timestamp"):
            ts |= _ts_variants(src["timestamp"])
    for st in candidate.get("reasoning_steps") or []:
        if st.get("timestamp"):
            ts |= _ts_variants(st["timestamp"])
    if isinstance(bundle, dict):
        for seg in bundle.get("event_segments") or []:
            if seg.get("start") and seg.get("end"):
                ts |= _ts_variants(f"{seg['start']}-{seg['end']}")
        for ev in bundle.get("event_captions") or []:
            for a in ev.get("audio_captions") or []:
                if a.get("timestamp"):
                    ts |= _ts_variants(a["timestamp"])
            for v in ev.get("visual_captions") or []:
                if v.get("timestamp"):
                    ts |= _ts_variants(v["timestamp"])
    return ts


def build_prompt(question_text: str, candidate: dict[str, Any], gold: str,
                 bundle: dict[str, Any] | None) -> str:
    return PROMPT_TEMPLATE.format(
        gold=gold, ungrounded=UNGROUNDED,
        question_text=question_text.strip(),
        moments_text=build_moments_text(candidate),
        caption_text=build_caption_text(bundle))


# ══════════════════════════ QC / post-processing ══════════════════════════

WATCH = "Let me carefully watch the video:"
ANALYZE = "Now let me carefully analyze the options:"
AT_TS_RE = re.compile(r"\bAt\s+(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)")
CONC_RE = re.compile(
    r"(?:points to|supports|the answer is|this is|best fits)\s+option\s+([A-Z])\b[.!?\s]*$", re.I)

FORBIDDEN = [
    "visual-only", "audio-only", "visual only", "audio only", "priors",
    "single modality", "single-modality", "modality", "visual track", "audio track",
    "the visual job", "the audio job", "shortcut", "trap", "distractor", "clue",
    "evidence", "reasoning step", "caption", "annotation", "metadata",
    "the correct answer is", "gold answer",
]


def find_forbidden(text: str) -> list[str]:
    low = text.lower()
    return [p for p in FORBIDDEN if p in low]


def clean(text: str) -> str:
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S).strip()
    text = re.sub(r"</?think>", "", text, flags=re.I).strip()
    text = mmss_text(text)
    if text.startswith(UNGROUNDED):
        return text
    if not text.startswith("Got it!"):
        text = "Got it! " + text
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def concluded_option(text: str) -> str | None:
    """Read the closing conclusion, never an earlier option discussion."""
    match = CONC_RE.search(text)
    return match.group(1).upper() if match else None


def qc(text: str, allowed: set[str], gold: str) -> tuple[str | None, list[str]]:
    """Return (concluded_letter, issues). Empty issues = clean."""
    issues: list[str] = []
    if WATCH not in text:
        issues.append("missing-watch")
    if ANALYZE not in text:
        issues.append("missing-analyze")
    leaks = find_forbidden(text)
    if leaks:
        issues.append(f"forbidden:{leaks}")
    allowed_mmss = {to_mmss(a) for a in allowed}
    starts = {a.split("-")[0] for a in allowed_mmss}
    bad = [t for t in AT_TS_RE.findall(text) if t not in allowed_mmss and t.split("-")[0] not in starts]
    if bad:
        issues.append(f"bad_ts:{sorted(set(bad))[:5]}")
    concluded = concluded_option(text)
    if concluded is None:
        issues.append("missing-conclusion")
    elif concluded != gold:
        issues.append("conclusion-mismatch")
    return concluded, issues


# ══════════════════════════ LLM call ══════════════════════════

class RateLimiter:
    def __init__(self, qps: float):
        self.interval = 1.0 / qps if qps > 0 else 0.0
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait_sec = max(0.0, self.next_start - now)
            self.next_start = max(now, self.next_start) + self.interval
        if wait_sec > 0:
            time.sleep(wait_sec)


def call_llm(prompt: str, args: argparse.Namespace, limiter: RateLimiter) -> str:
    """One text-LLM completion. deepseek-v4-* are reasoning models: pass max_tokens and
    treat an empty `content` (all budget spent on hidden reasoning) as a retryable miss."""
    url = f"{args.api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_output_tokens, "temperature": args.temperature}
    last = ""
    for attempt in range(max(1, args.max_retries)):
        limiter.wait()
        try:
            with httpx.Client(timeout=args.request_timeout) as cl:
                r = cl.post(url, headers=headers, json=body)
            if r.status_code < 400:
                d = r.json()
                content = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                if content.strip():
                    return content.strip()
                last = "empty content"
            else:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code not in (408, 409, 425, 429, 500, 502, 503, 504):
                    raise RuntimeError(last)
        except httpx.HTTPError as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(args.retry_backoff * (attempt + 1))
    raise RuntimeError(last)


# ══════════════════════════ Per-row synthesis ══════════════════════════


def question_text_from_candidate(candidate: dict[str, Any]) -> str:
    opts = candidate.get("options") or []
    lettered = [f"{LETTERS[i]}.{o}" for i, o in enumerate(opts)]
    return "\n".join([str(candidate.get("question", "")).strip(), *lettered, "The best answer is:"])


def resolve_video_ref(row: dict) -> str:
    """Video reference to embed in the SFT user turn (OSS preferred, then local)."""
    for k in ("time_video_oss_path", "video_path", "raw_video_local_path", "time_video_local_path"):
        v = str(row.get(k) or "").strip()
        if v:
            return v
    return ""


def synth_one(row: dict, idx: int, args: argparse.Namespace, limiter: RateLimiter) -> dict:
    candidate = row.get("candidate") or {}
    video_id = candidate.get("video_id") or row.get("video_id") or ""
    source = row.get("source", "")
    gold = str(candidate.get("correct_option") or "").strip().upper()
    qtext = question_text_from_candidate(candidate)

    base = {"idx": idx, "source": source, "video_id": video_id,
            "task_type_id": row.get("task_type_id"), "status": "ok"}

    if not (candidate.get("question") and candidate.get("options") and len(gold) == 1
            and gold in LETTERS[:len(candidate.get("options") or [])]):
        return {**base, "status": "skipped", "reason": "missing question/options/gold"}

    bundle = load_caption_bundle(video_id, args.caption_root, source)
    prompt = build_prompt(qtext, candidate, gold, bundle)
    allowed = collect_allowed_timestamps(candidate, bundle)

    best: tuple[str, list[str], str | None] | None = None
    ungrounded = False
    try:
        for _ in range(max(1, args.gen_tries)):
            think = clean(call_llm(prompt, args, limiter))
            if think.startswith(UNGROUNDED):
                ungrounded = True
                best = (think, ["ungrounded"], None)
                break
            concluded, issues = qc(think, allowed, gold)
            if best is None or len(issues) < len(best[1]):
                best = (think, issues, concluded)
            if not issues:
                break
    except Exception as exc:
        return {**base, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    think, issues, concluded = best  # type: ignore
    video_ref = resolve_video_ref(row)
    sft_messages = [
        {"role": "system", "content": ROLE_DEF},
        {"role": "user", "content": [
            {"type": "video", "video": video_ref, "use_audio": True},
            {"type": "text", "text": qtext},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": f"<think>\n{think}\n</think>\n\n{gold}"}
        ]},
    ]
    return {
        **base,
        "gold": gold,
        "concluded": concluded,
        "concluded_matches_gold": (concluded == gold) if concluded else None,
        "caption_available": bundle is not None,
        "video_ref": video_ref,
        "ungrounded": ungrounded,
        "qc_issues": issues,
        "think": think,
        "sft_messages": sft_messages,
    }


# ══════════════════════════ IO / runner ══════════════════════════


def append_jsonl(path: Path, row: dict) -> None:
    with WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "skipped" or is_clean_sft(r):
                done.add(f"{r.get('source', '')}::{r.get('video_id', '')}::{r.get('idx')}")
    return done


def is_clean_sft(rec: dict) -> bool:
    return (rec.get("status") == "ok" and not rec.get("ungrounded")
            and not rec.get("qc_issues") and bool(rec.get("sft_messages"))
            and isinstance(rec.get("gold"), str) and len(rec["gold"]) == 1
            and rec["gold"] in LETTERS
            and concluded_option(rec.get("think") or "") == rec["gold"])


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 4 · think-process synthesis (text-only, DeepSeek)")
    p.add_argument("--input", type=Path, required=True, help="accepted.jsonl from Stage 2/3")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--caption-root", type=Path, default=DEFAULT_CAPTION_ROOT,
                   help="root holding <dataset>/raw/<video_id>/caption_bundle.json")
    p.add_argument("--model", default=os.environ.get("SYNTH_MODEL", "deepseek-v4-pro"))
    p.add_argument("--api-base", default=os.environ.get(
        "OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    p.add_argument("--api-key", default=os.environ.get(
        "OPENAI_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "")))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--qps", type=float, default=6.0)
    p.add_argument("--gen-tries", type=int, default=2, help="regenerate up to N times on QC failure")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-backoff", type=float, default=3.0)
    p.add_argument("--request-timeout", type=int, default=300)
    p.add_argument("--max-output-tokens", type=int, default=16000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-rows", type=int, default=0, help="0 = all (use small for smoke)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: set OPENAI_API_KEY (or DASHSCOPE_API_KEY) or pass --api-key.", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.max_rows > 0:
        rows = rows[:args.max_rows]
    all_path = args.output_dir / "synth_all.jsonl"
    done = load_done_keys(all_path) if args.resume else set()
    pending = [(i, r) for i, r in enumerate(rows)
               if f"{r.get('source', '')}::{r.get('video_id', '')}::{i}" not in done]
    print(f"Loaded {len(rows)} rows; resumable done {len(done)}, pending {len(pending)}; "
          f"model={args.model}, workers={args.workers}, qps={args.qps}", flush=True)

    limiter = RateLimiter(args.qps)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(synth_one, r, i, args, limiter): i for i, r in pending}
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures), desc="synth-think", unit="q"):
            append_jsonl(all_path, fut.result())

    # ---- rebuild SFT pool + summary from the full log (dedup by key, last wins) ----
    latest: dict[str, dict] = {}
    for line in open(all_path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[f"{r.get('source', '')}::{r.get('video_id', '')}::{r.get('idx')}"] = r
    recs = list(latest.values())

    sft_path = args.output_dir / "sft.jsonl"
    n_sft = n_ungrounded = n_qc_fail = n_skip = n_err = 0
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    match = clean_cnt = cap_cnt = 0
    with open(sft_path, "w", encoding="utf-8") as fs:
        for rec in recs:
            tid = rec.get("task_type_id") or "?"
            st = rec.get("status")
            if st == "skipped":
                n_skip += 1; by_type[tid]["skipped"] += 1; continue
            if st == "error":
                n_err += 1; by_type[tid]["error"] += 1; continue
            if rec.get("ungrounded"):
                n_ungrounded += 1; by_type[tid]["ungrounded"] += 1; continue
            if not is_clean_sft(rec):
                n_qc_fail += 1; by_type[tid]["qc_fail"] += 1; continue
            fs.write(json.dumps({"messages": rec["sft_messages"]}, ensure_ascii=False) + "\n")
            n_sft += 1
            by_type[tid]["sft"] += 1
            clean_cnt += 1
            cap_cnt += bool(rec.get("caption_available"))
            match += bool(rec.get("concluded_matches_gold"))

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "total": len(recs),
        "sft_written": n_sft,
        "ungrounded": n_ungrounded,
        "qc_fail": n_qc_fail,
        "skipped": n_skip,
        "error": n_err,
        "caption_available_of_sft": cap_cnt,
        "concluded_matches_gold_of_sft": match,
        "per_task_type": {k: dict(v) for k, v in by_type.items()},
    }
    tmp = (args.output_dir / "summary.json").with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(args.output_dir / "summary.json")

    print(f"\nSFT {n_sft} | ungrounded {n_ungrounded} | qc_fail {n_qc_fail} | "
          f"skipped {n_skip} | error {n_err}", flush=True)
    print(f"of SFT: caption_available {cap_cnt}/{n_sft}, concluded==gold {match}/{n_sft}", flush=True)
    print(f"-> {sft_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
