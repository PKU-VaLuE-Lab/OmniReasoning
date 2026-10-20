"""Convert adaptation outputs into accepted question records."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

LETTERS = "ABCDEFGHIJ"


ACCEPTED_STATUSES = {"ok", "verified", "verified_textonly"}


def extract_option_letter(text: Any) -> str:
    if not text:
        return ""
    t = str(text).strip()
    if t.upper() in set(LETTERS):
        return t.upper()
    m = re.match(r"^([A-Ja-j])\s*[).:]\s*", t)
    if m:
        return m.group(1).upper()
    for c in t:
        if c.upper() in LETTERS:
            return c.upper()
    return ""


def _video_knowledge(cand: dict) -> dict:
    """The taught fact, under either the new or the legacy key (mirrors the engine)."""
    kf = cand.get("video_derived_knowledge") or cand.get("video_knowledge_fact") or {}
    return kf if isinstance(kf, dict) else {"what": str(kf)}


def _knowledge_text(cand: dict) -> str:
    kf = _video_knowledge(cand)
    for k in ("what", "fact", "statement", "point"):
        if kf.get(k):
            return str(kf[k])
    return ""


def _derivation_steps(cand: dict) -> list[dict]:
    """RBY `derivation` (list of {step, source_timestamp, source_quote}) or a legacy
    string -> ROV-style reasoning_steps [{content, timestamp, modality}]."""
    d = cand.get("derivation")
    steps: list[dict] = []
    if isinstance(d, list):
        for s in d:
            if not isinstance(s, dict):
                if str(s).strip():
                    steps.append({"content": str(s).strip(), "timestamp": "", "modality": ""})
                continue
            content = str(s.get("step") or s.get("content") or "").strip()
            if not content:
                continue
            ts = str(s.get("source_timestamp") or s.get("timestamp") or "").strip()
            steps.append({
                "content": content,
                "timestamp": ts,


                "modality": "VISUAL" if ts else "",
                "source_quote": str(s.get("source_quote") or "").strip(),
            })
    elif isinstance(d, str) and d.strip():
        steps.append({"content": d.strip(), "timestamp": "", "modality": ""})
    return steps


def _visual_clue(cand: dict, steps: list[dict]) -> dict:
    """Synthesize a checkable visual clue from the first grounded derivation step."""
    for s in steps:
        if s.get("source_quote") and s.get("timestamp"):
            return {
                "quote": s["source_quote"],
                "fact": _knowledge_text(cand) or s["content"],
                "timestamp": s["timestamp"],
            }
    return {}


def _source_from_bundle_path(bundle_path: str, video_id: str) -> str:
    """Best-effort dataset label (matches the caption store `<dataset>/raw/<vid>` layout),
    used only as the video_map lookup key downstream. Falls back to 'adaptation'."""
    if not bundle_path:
        return "adaptation"
    parts = Path(bundle_path).parts

    try:
        vi = len(parts) - 1 - list(reversed(parts)).index(video_id)
        if vi >= 2 and parts[vi - 1] == "raw":
            return parts[vi - 2]
        if vi >= 1:
            return parts[vi - 1]
    except ValueError:
        pass
    return "adaptation"


def to_accepted_row(rec: dict) -> dict | None:
    cand = rec.get("candidate") or {}
    if not cand.get("question"):
        return None
    video_id = cand.get("video_id") or rec.get("video_id") or ""
    options = list(cand.get("options_final") or cand.get("options") or [])
    correct = (cand.get("answer_letter")
               or extract_option_letter(cand.get("correct_option")))
    steps = _derivation_steps(cand)
    vclue = _visual_clue(cand, steps)
    form_key = cand.get("form_key") or rec.get("form_key") or "adaptation"
    qa_type = cand.get("qa_type") or rec.get("qa_type") or "Adaptation"
    row = (rec.get("row") or {})


    out_candidate = {
        "video_id": video_id,
        "question": cand["question"],
        "options": options,
        "correct_option": correct,
        "question_type": cand.get("question_type") or ("open" if not options else "multiple-choice"),
        "task_type": qa_type,
        "task_type_id": f"adaptation_{form_key}",
        "reasoning_steps": steps,

        "video_derived_knowledge": _video_knowledge(cand),
        "image_scenario": cand.get("image_scenario", ""),
        "figure_spec": cand.get("figure_spec", ""),
        "derivation": cand.get("derivation"),
        "why_both_needed": cand.get("why_both_needed", ""),
        "adaptation_domain": cand.get("adaptation_domain", ""),
        "image_path": row.get("image_path") or rec.get("image_path", ""),
        "link_selected": row.get("link_selected", ""),
    }
    if vclue:
        out_candidate["visual_clue"] = vclue

    return {
        "video_id": video_id,
        "source": _source_from_bundle_path(rec.get("bundle_path", ""), video_id),
        "final_status": "accepted",
        "generation_status": rec.get("status", "ok"),
        "attempts": rec.get("attempts", 0),
        "revised": False,
        "breaks_found": [],
        "task_type_id": f"adaptation_{form_key}",
        "task_type": qa_type,
        "type_selected_by": "adaptation",
        "gen_mode": "RBY",


        "video_path": "",
        "link_selected": out_candidate["link_selected"],
        "image_path": out_candidate["image_path"],
        "candidate": out_candidate,
        "filters": [{"stage": "adaptation_verify", "passed": True,
                     "details": cand.get("verify") or {}}],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert reasoning-beyond-video adaptation output into the reasoning-over-video accepted.jsonl schema.")
    ap.add_argument("--outdir", default=None,
                    help="RBY output dir; defaults --adaptation-jsonl / --output under it.")
    ap.add_argument("--adaptation-jsonl", default=None,
                    help="input adaptation_all.jsonl (default: <outdir>/adaptation_all.jsonl)")
    ap.add_argument("--output", default=None,
                    help="output accepted.jsonl (default: <outdir>/accepted.jsonl)")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else None
    in_path = Path(args.adaptation_jsonl) if args.adaptation_jsonl else \
        (outdir / "adaptation_all.jsonl" if outdir else None)
    out_path = Path(args.output) if args.output else \
        (outdir / "accepted.jsonl" if outdir else None)
    if not in_path or not out_path:
        print("ERROR: provide --outdir, or both --adaptation-jsonl and --output.",
              file=sys.stderr)
        return 2
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1


    latest: dict[str, dict] = {}
    order: list[str] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("id") or rec.get("video_id") or ""
            if key not in latest:
                order.append(key)
            latest[key] = rec

    n_in = len(latest)
    n_out = 0
    n_skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for key in order:
            rec = latest[key]
            if rec.get("status") not in ACCEPTED_STATUSES:
                n_skipped += 1
                continue
            row = to_accepted_row(rec)
            if row is None:
                n_skipped += 1
                continue
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[adaptation_to_accepted] {in_path.name}: {n_in} unique records -> "
          f"{n_out} accepted rows ({n_skipped} skipped) -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
