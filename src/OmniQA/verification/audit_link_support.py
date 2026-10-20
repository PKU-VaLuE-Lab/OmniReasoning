"""Stage 3 · Link-support & anchor-uniqueness audit (text-LLM, no video).

A cheap diagnostic gate over accepted questions (accepted.jsonl from Stage 2,
OmniQA.stage2_generate). For each question it asks one text-LLM call:
  1. LINK_SUPPORT : does the presupposition quote ACTUALLY establish that the audio
     referent and the visual referent are the SAME entity? (fabricated-binding probe)
  2. ANCHOR_UNIQUE: from the audio clue alone, is the target referent UNIQUELY
     identifiable, or could several candidates fit?

Unlike stage3_verify / refine_questions this stage does NOT open the video —
it reads only the question's own clues + presupposition_check. It writes a per-row
verdict and a summary; it is a REPORT (it does not by itself drop rows), but the
verdict fields make it easy to filter downstream.

Uses the OpenAI-compatible endpoint (same as Stage 2's gates). Key from --api-key or
the OPENAI_API_KEY / DASHSCOPE_API_KEY env var. RateLimiter-free (bounded by
--workers); append_jsonl resume keyed by (source, video_id, row idx).
"""
from __future__ import annotations

import argparse
import concurrent.futures
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

WRITE_LOCK = threading.Lock()

JUDGE = r"""You are auditing an AI-generated audio-visual question for a SPECIFIC defect:
fabricated referent binding.

Question: {question}
Claimed audio clue [{a_ts}]: {a_fact}
Claimed visual clue [{v_ts}]: {v_fact}
Presupposed link: {link}
Quote offered as support for the link: "{support}"

Judge TWO things strictly:
1. LINK_SUPPORT: does the offered quote DIRECTLY establish that the referent in the
   audio clue and the referent in the visual clue are THE SAME entity? A quote that
   merely repeats one side, or that requires plausibility/guessing, is NOT support.
2. ANCHOR_UNIQUE: from the audio clue alone, is the target referent UNIQUELY
   identifiable (exactly one card/person/object fits), or could several fit?

Return ONLY valid JSON:
{{"link_support": true/false, "anchor_unique": true/false,
  "reason": "one or two sentences, be specific"}}
"""


def extract_json(t: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", t or "")
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            return None
    return None


def judge(row: dict, idx: int, args: argparse.Namespace, sem: threading.Semaphore) -> dict:
    c = row.get("candidate") or {}
    presups = c.get("presupposition_check") or []
    p = presups[0] if presups and isinstance(presups[0], dict) else {}
    sb = p.get("supported_by")
    if isinstance(sb, list):
        sb = sb[0] if sb else {}
    if isinstance(sb, str):
        sb = {"quote": sb}
    sb = sb or {}
    prompt = JUDGE.format(
        question=c.get("question", ""),
        a_ts=(c.get("audio_clue") or {}).get("timestamp", ""),
        a_fact=(c.get("audio_clue") or {}).get("fact", ""),
        v_ts=(c.get("visual_clue") or {}).get("timestamp", ""),
        v_fact=(c.get("visual_clue") or {}).get("fact", ""),
        link=p.get("link", ""), support=sb.get("quote", ""))
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048}
    base = {"idx": idx, "source": row.get("source", ""), "video_id": row.get("video_id", ""),
            "task_type_id": row.get("task_type_id"), "question": c.get("question", "")}
    with sem:
        for att in range(max(1, args.max_retries)):
            try:
                with httpx.Client(timeout=300) as cl:
                    r = cl.post(f"{args.api_base}/chat/completions", json=body,
                                headers={"Authorization": f"Bearer {args.api_key}"})
                    d = r.json()
                txt = d["choices"][0]["message"].get("content") or ""
                j = extract_json(txt)
                if j is not None:
                    return {**base, "status": "ok", **j}
            except Exception:
                time.sleep(args.retry_backoff * (att + 1))
    return {**base, "status": "error", "link_support": None, "anchor_unique": None,
            "reason": "api_error"}


def append_jsonl(path: Path, row: dict) -> None:
    with WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in open(path, "r", encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok":
            done.add(f"{r.get('source', '')}::{r.get('video_id', '')}::{r.get('idx')}")
    return done


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 3 · link-support & anchor-uniqueness audit")
    p.add_argument("--input", type=Path, required=True, help="accepted.jsonl from Stage 2 (OmniQA.stage2_generate)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default=os.environ.get("AUDIT_MODEL", "deepseek-v4-flash"))
    p.add_argument("--api-base", default=os.environ.get(
        "OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    p.add_argument("--api-key", default=os.environ.get(
        "OPENAI_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "")))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--retry-backoff", type=float, default=2.0)
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
    all_path = args.output_dir / "audit_all.jsonl"
    done = load_done_keys(all_path) if args.resume else set()
    pending = [(i, r) for i, r in enumerate(rows)
               if f"{r.get('source', '')}::{r.get('video_id', '')}::{i}" not in done]
    print(f"Loaded {len(rows)} rows; resumable done {len(done)}, pending {len(pending)}; "
          f"model={args.model}, workers={args.workers}", flush=True)

    sem = threading.Semaphore(max(1, args.workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(judge, r, i, args, sem) for i, r in pending]
        done_n = 0
        for fut in concurrent.futures.as_completed(futures):
            append_jsonl(all_path, fut.result())
            done_n += 1
            if done_n % 10 == 0:
                print(f"{done_n}/{len(pending)}", flush=True)

    # ---- summary over the full audit_all.jsonl (dedup by key, last wins) ----
    latest: dict[str, dict] = {}
    for line in open(all_path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[f"{r.get('source', '')}::{r.get('video_id', '')}::{r.get('idx')}"] = r
    results = list(latest.values())
    n = max(1, len(results))
    bad_link = [r for r in results if r.get("link_support") is False]
    bad_anchor = [r for r in results if r.get("anchor_unique") is False]
    errors = [r for r in results if r.get("status") == "error"]
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        tid = r.get("task_type_id") or "?"
        by_type[tid]["total"] += 1
        if r.get("link_support") is False:
            by_type[tid]["fabricated_link"] += 1
        if r.get("anchor_unique") is False:
            by_type[tid]["non_unique_anchor"] += 1
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "total": len(results),
        "fabricated_link": len(bad_link),
        "non_unique_anchor": len(bad_anchor),
        "api_error": len(errors),
        "per_task_type": {k: dict(v) for k, v in by_type.items()},
    }
    tmp = (args.output_dir / "summary.json").with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(args.output_dir / "summary.json")

    print(f"\ntotal {len(results)} | fabricated_link {len(bad_link)} "
          f"({100 * len(bad_link) / n:.0f}%) | non_unique_anchor {len(bad_anchor)} "
          f"({100 * len(bad_anchor) / n:.0f}%) | api_error {len(errors)}", flush=True)
    print("\n-- fabricated link examples --")
    for r in bad_link[:8]:
        print(" •", str(r.get("question", ""))[:90])
        print("   ", str(r.get("reason"))[:160])
    print(f"\n-> {all_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
