"""Stage 3 · VLM refinement for generated AV questions (FIX, not just filter).

Root cause of most bad questions: caption hallucination -> wrong answer / ambiguous
options / hallucinated reasoning steps. Discarding them at scale is too costly, so
this stage asks a VLM (default: qwen3.6-flash, vision-only; use qwen3.5-omni-plus for
audio+video) to REPAIR each accepted question (accepted.jsonl from Stage 2,
OmniQA.stage2_generate):

  input  : the real video + question + options(+gold) + reasoning_steps
  checks : 1) is the gold answer actually correct?
           2) is there exactly ONE correct option?
           3) are the VISUAL reasoning steps supported by the video?
             (audio steps are assumed correct — NOT checked; a vision-only model
              cannot hear them, and even the omni model is told to leave them intact)
  action : pass | fixed (return the full repaired question) | reject (unfixable)

Call convention: dashscope MultiModalConversation (file:// upload), RateLimiter +
ThreadPoolExecutor + append_jsonl resume.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

WRITE_LOCK = threading.Lock()
LETTERS = "ABCDEFGH"

REFINE_PROMPT = r"""You are auditing and repairing an automatically generated multiple-choice
question about a video. Watch the video carefully and verify each item
below AGAINST THE ACTUAL VIDEO — never trust the question's own claims.

## THE QUESTION (gold answer is marked)
{question_block}

## REASONING STEPS
Audio steps are ASSUMED CORRECT. HARD RULES:
- NEVER claim an audio step is wrong. You cannot hear the audio — the audio
  captions are authoritative and out of your reach.
- NEVER modify, rewrite, or delete an audio step, not even its wording.
- If a conclusion seems to conflict with an audio step, the problem is in the
  VISUAL side or the question/options — repair THOSE, keep the audio steps intact.
{steps_block}

## TASKS
1. ANSWER CHECK: is the marked gold option actually correct given the video?
2. OPTION UNIQUENESS: is there EXACTLY ONE correct option? (a second option that is
   also defensible = defect)
3. VISUAL STEP CHECK: is each VISUAL reasoning step's content supported by the
   video? Flag every step whose claim is not visible / contradicted. Audio steps
   are out of scope — never flag them, never "fix" them; repair the chain by
   correcting the visual steps, the question, or the options instead.
4. DECIDE + REPAIR (prefer repair over rejection):
   - "pass":   all checks pass — return the question unchanged.
   - "fixed":  ANY defect found — repair it. You may freely edit the QUESTION text
     itself (e.g. fix a wrong moment/anchor reference), re-mark or edit options and
     the gold, correct clue timestamps/facts, and rewrite reasoning steps — whatever
     makes the question fully consistent with the ACTUAL video content. Keep the
     same task type and rough difficulty where possible. As long as the video
     contains enough audio+visual evidence to form ONE self-consistent question of
     this type, you MUST choose "fixed", not "reject".
   - "reject": use ONLY when the video fundamentally lacks the evidence for any
     question of this type at all (e.g. the answer value simply does not exist
     anywhere in the video and no repair can be built). Do NOT reject a question
     you could have repaired by editing text/options/steps.

Return ONLY valid JSON (no other text):
{{"answer_correct": true, "options_unique": true,
  "visual_steps": [{{"step_id": "s1", "supported": true, "issue": "", "correction": ""}}],
  "verdict": "pass|fixed|reject",
  "fixed_question": null,
  "rationale": "short, specific"}}
When verdict == "fixed", fixed_question must be
{{"question": "...", "options": ["...", ...], "correct_option": "A",
  "reasoning_steps": [{{"step_id": "...", "modality": "...", "timestamp": "...",
                        "content": "...", "depends_on": [], "bridge": "..."}}]}}
REQUIREMENTS for the repaired reasoning_steps:
- Each step's content must be a COMPLETE, self-contained evidence description
  (what is seen/heard, at that timestamp, with the relevant details) — NOT a
  modification note like "s1 was wrong, actually..." or "the fridge appears later".
- VISUAL steps in particular must cover the full observation needed by the chain
  (objects, attributes, positions, on-screen text — everything the inference uses).
- Keep the DAG shape: depends_on edges and a final step that yields the answer.
"""


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


def format_options(options: list[str], gold: str = "") -> str:
    lines = []
    for i, opt in enumerate(options):
        letter = LETTERS[i] if i < len(LETTERS) else str(i)
        body = re.sub(r"^[A-Ha-h]\s*[).:]\s*", "", str(opt).strip()).strip()
        mark = "  <-- GOLD" if letter == gold else ""
        lines.append(f"{letter}. {body}{mark}")
    return "\n".join(lines)


def format_steps(steps: list[dict]) -> str:
    lines = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        mod = str(s.get("modality", "?"))
        tag = "AUDIO (presumed correct unless chain breaks)" if mod == "audio" else "VISUAL (verify this)"
        deps = s.get("depends_on") or []
        lines.append(f"- {s.get('step_id','?')} [{tag}] {s.get('timestamp','')}: "
                     f"{s.get('content','')} (depends_on: {deps})")
    return "\n".join(lines) or "(none)"


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    try:
        output = response.output
        if hasattr(output, "choices") and output.choices:
            content = output.choices[0].message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return "".join(item.get("text", "") if isinstance(item, dict) else str(item)
                               for item in content).strip()
        if hasattr(output, "text") and output.text:
            return output.text.strip()
    except (AttributeError, IndexError, TypeError):
        pass
    return str(response)


def load_video_map(path: str) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    if not path or not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "mapped" and r.get("video_path"):
                out[(r.get("source", ""), r.get("video_id", ""))] = r["video_path"]
    return out


def resolve_video(row: dict, vmap: dict[tuple[str, str], str]) -> str:
    src = str(row.get("source", "")).split("::")[0]
    hit = vmap.get((src, str(row.get("video_id", ""))))
    if hit and Path(hit).exists():
        return hit
    for k in ("video_path", "raw_video_local_path", "time_video_local_path"):
        p = str(row.get(k) or "")
        if p and Path(p).exists():
            return p
    return ""


def validate_fixed(fq: dict, num_options: int) -> str:
    """Sanity-check a repaired question; '' = ok, else the defect."""
    if not isinstance(fq, dict):
        return "fixed_question not an object"
    opts = fq.get("options") or []
    gold = str(fq.get("correct_option", "")).strip().upper()
    if not fq.get("question"):
        return "fixed question has no question text"
    if len(opts) != num_options:
        return f"fixed options count {len(opts)} != {num_options}"
    if gold not in LETTERS[:len(opts)]:
        return f"fixed correct_option {gold!r} out of range"
    return ""


def refine_one(row: dict, idx: int, video_path: str, args: argparse.Namespace,
               limiter: RateLimiter) -> dict:
    from dashscope import MultiModalConversation

    c = row.get("candidate") or {}
    gold = str(c.get("correct_option", "")).strip().upper()
    question_block = (f"Question: {c.get('question', '')}\n\n"
                      + format_options(c.get("options") or [], gold))
    steps_block = format_steps(c.get("reasoning_steps") or [])
    prompt = REFINE_PROMPT.format(question_block=question_block, steps_block=steps_block)
    # adaptive frame sampling: videos longer than max_frames seconds are downsampled
    # so the 768 frames still span the WHOLE video (density traded for coverage)
    duration = row.get("video_duration_sec")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None
    fps = args.fps
    if duration and duration * args.fps > args.max_frames:
        fps = args.max_frames / duration
    messages = [{"role": "user", "content": [
        {"video": f"file://{video_path}", "fps": fps, "max_frames": args.max_frames},
        {"text": prompt}]}]

    started = time.time()
    verdict_raw, error = None, ""
    for attempt in range(max(1, args.max_retries)):
        limiter.wait()
        try:
            response = MultiModalConversation.call(
                model=args.model, messages=messages, api_key=args.api_key)
            if response.status_code != 200:
                error = f"API error {response.status_code}: {response.code} - {response.message}"
                if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < args.max_retries:
                    time.sleep(args.retry_backoff * (attempt + 1))
                    continue
                break
            raw = extract_response_text(response)
            verdict_raw = extract_json(raw)
            if verdict_raw is None:
                error = f"could not parse verdict: {raw[:200]}"
            else:
                error = ""
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < args.max_retries:
                time.sleep(args.retry_backoff * (attempt + 1))
                continue

    result: dict[str, Any] = {
        "idx": idx, "video_id": row.get("video_id", ""), "source": row.get("source", ""),
        "task_type_id": row.get("task_type_id", ""),
        "model": args.model, "latency_sec": round(time.time() - started, 3),
    }
    if verdict_raw is None:
        result.update(status="error", error=error)
        return result

    verdict = str(verdict_raw.get("verdict", "")).strip().lower()
    fixed = verdict_raw.get("fixed_question")
    fix_err = ""
    if verdict == "fixed":
        fix_err = validate_fixed(fixed, len(c.get("options") or []))
        if fix_err:
            verdict = "reject"
    result.update(
        status="ok",
        answer_correct=verdict_raw.get("answer_correct"),
        options_unique=verdict_raw.get("options_unique"),
        visual_steps=verdict_raw.get("visual_steps") or [],
        verdict=verdict,
        fixed_question=(fixed if verdict == "fixed" and not fix_err else None),
        fix_error=fix_err,
        rationale=str(verdict_raw.get("rationale", ""))[:400],
    )
    return result


def append_jsonl(path: Path, row: dict) -> None:
    with WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_fix(row: dict, refine: dict) -> dict:
    """Return a copy of row with the fixed question applied into candidate (the
    ORIGINAL candidate is preserved under candidate_original for audit)."""
    fq = refine.get("fixed_question") or {}
    out = dict(row)
    out["candidate_original"] = row.get("candidate")
    c = dict(out.get("candidate") or {})
    c["question"] = fq.get("question", c.get("question"))
    c["options"] = fq.get("options", c.get("options"))
    new_gold = str(fq.get("correct_option", "")).strip().upper()
    if new_gold and new_gold != c.get("correct_option"):
        c["correct_option_original"] = c.get("correct_option")
        c["correct_option"] = new_gold
        opts = c.get("options") or []
        i = LETTERS.find(new_gold)
        if 0 <= i < len(opts):
            c["answer"] = re.sub(r"^[A-Ha-h]\s*[).:]\s*", "", str(opts[i]).strip()).strip()
        c["answer_reassigned_by_vlm"] = True
    if fq.get("reasoning_steps"):
        # HARD RULE: audio steps are authoritative (the vision-only VLM cannot judge
        # them). Restore ALL original audio/audiovisual steps regardless of how the
        # VLM relabeled them (it sometimes rewrites 'audiovisual' as 'VISUAL'), and
        # re-insert any it dropped. Visual steps take the VLM's repaired version.
        orig_steps = (row.get("candidate") or {}).get("reasoning_steps") or []
        fixed_by_id = {str(s.get("step_id")): s for s in fq["reasoning_steps"]
                       if isinstance(s, dict)}
        merged = []
        for os_ in orig_steps:
            if not isinstance(os_, dict):
                continue
            oid = str(os_.get("step_id"))
            if "audio" in str(os_.get("modality", "")).lower():
                merged.append(os_)                       # audio: always the original
                fixed_by_id.pop(oid, None)               # discard the VLM's version
            else:
                merged.append(fixed_by_id.pop(oid, os_))  # visual: repaired or original
        merged.extend(fixed_by_id.values())               # extra steps added by VLM
        c["reasoning_steps"] = merged
    c["refined_by_vlm"] = True
    out["candidate"] = c
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 3 · VLM refinement (repair) for generated questions")
    p.add_argument("--input", type=Path, required=True, help="accepted.jsonl from Stage 2 (OmniQA.stage2_generate)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="qwen3.6-flash",
                   help="qwen3.6-flash (vision-only, default) or qwen3.5-omni-plus (audio+video)")
    p.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY",
                                                       os.environ.get("QWEN_API_KEY", "")))
    p.add_argument("--api-base", default=os.environ.get("DASHSCOPE_BASE_URL", ""),
                   help="optional DashScope HTTP API base URL (SDK default when empty)")
    p.add_argument("--sample", type=int, default=0, help="0 = all rows")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--qps", type=float, default=2.0)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=768)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-backoff", type=float, default=5.0)
    p.add_argument("--video-map",
                   default=str(Path.cwd() / "data" / "video_map.jsonl"),
                   help="optional caption<->video map; falls back to per-row paths if absent")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: set DASHSCOPE_API_KEY or pass --api-key", file=sys.stderr)
        return 1
    if args.api_base:
        import dashscope
        dashscope.base_http_api_url = args.api_base
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    vmap = load_video_map(args.video_map)
    usable = []
    for r in rows:
        vp = resolve_video(r, vmap)
        if vp:
            r["_video_path"] = vp
            usable.append(r)
    if args.sample > 0:
        rng = random.Random(args.seed)
        usable = rng.sample(usable, min(args.sample, len(usable)))
    print(f"loaded {len(rows)} rows, refining {len(usable)} (video-resolvable)", flush=True)

    all_path = args.output_dir / "refine_all.jsonl"
    done: set[str] = set()
    if args.resume and all_path.exists():
        for line in open(all_path):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("vlm_refine") or {}
            if v.get("status") == "ok":
                done.add(f"{r.get('source')}::{r.get('video_id')}::{v.get('idx')}")
    pending = [(i, r) for i, r in enumerate(usable)
               if f"{r.get('source','')}::{r.get('video_id','')}::{i}" not in done]
    print(f"resumable done: {len(done)}, pending: {len(pending)}", flush=True)

    limiter = RateLimiter(args.qps)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(refine_one, r, i, r["_video_path"], args, limiter): i
                   for i, r in pending}
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures), desc="Refining", unit="q"):
            res = fut.result()
            out_row = dict(pending[futures[fut]][1])
            out_row.pop("_video_path", None)
            out_row["vlm_refine"] = res
            append_jsonl(all_path, out_row)

    # ---- split pools + summary ----
    refined_rows = [json.loads(l) for l in open(all_path) if l.strip()]
    counts: dict[str, int] = defaultdict(int)
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    acc_f = open(args.output_dir / "refined_accepted.jsonl", "w", encoding="utf-8")
    rej_f = open(args.output_dir / "refined_rejected.jsonl", "w", encoding="utf-8")
    for row in refined_rows:
        v = row.get("vlm_refine") or {}
        verdict = v.get("verdict", "error")
        counts[verdict] += 1
        by_type[row.get("task_type_id", "?")][verdict] += 1
        if verdict == "pass":
            acc_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif verdict == "fixed":
            acc_f.write(json.dumps(apply_fix(row, v), ensure_ascii=False) + "\n")
        else:
            rej_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    acc_f.close(); rej_f.close()
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "total": len(refined_rows), "verdicts": dict(counts),
        "by_task_type": {k: dict(v) for k, v in sorted(by_type.items())},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nverdicts: {dict(counts)}", flush=True)
    print(f"-> {args.output_dir}/refined_accepted.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
