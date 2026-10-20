"""Convert stage-4 *.thinking.jsonl into CPT ChatML.

Input rows carry the source ``messages`` plus a ``<model>-thinking`` and a
``<model>-response`` field and the gold ``correct_option``.

Filter: keep only rows where the model's own final letter == the gold correct_option
(quality gate — the reasoning trace actually arrives at the verified answer).

Output ChatML row:

    {
      "messages": [
        {"role": "user", "content": [
            {"type": "video", "video": "<oss raw video>"},
            {"type": "image", "image": "<oss png>"}?,          # only for image adaptation
            {"type": "text",  "text": "<CoT preamble>\n<original MCQ>"}
        ]},
        {"role": "assistant", "content": "<think>\n{thinking}\n</think>\n{gold}"}
      ],
      "task": "Audio-Visual Reasoning"
    }

Video URIs must already be oss:// paths in the source ``messages`` (rows with other
video references are dropped). Figure images are uploaded separately to the prefix
given by the OMNIQA_IMAGE_PREFIX env var; here the local path is mapped to
OMNIQA_IMAGE_PREFIX + basename.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


PREAMBLE = (
    "Please think about this question as if you were a human pondering deeply, "
    "carefully considering the video information before answering, engaging in an "
    "internal dialogue using expressions such as let me think, wait, hmm, oh I see, "
    "or let's break it down, including self-reflection or verification in the "
    "reasoning process, providing the detailed reasoning between the <think> </think> "
    "tags, and finally giving only the single option letter as the final answer.\n"
)
TASK = "Audio-Visual Reasoning"
IMG_OSS_PREFIX = os.environ.get("OMNIQA_IMAGE_PREFIX", "")


def pred_letter(resp: str) -> str:
    m = re.search(r"\b([A-Z])\b", resp or "")
    return m.group(1) if m else ""


def find_model_keys(row: dict[str, Any]) -> tuple[str | None, str | None]:
    tk = rk = None
    for k in row:
        if k.endswith("-thinking"):
            tk = k
        elif k.endswith("-response"):
            rk = k
    return tk, rk


def extract_user_parts(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return (video_oss, image_local, question_text) from the user message."""
    video = image = text = ""
    for msg in row.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                t = p.get("type")
                if t == "video":
                    video = str(p.get("video") or "")
                elif t == "image":
                    image = str(p.get("image") or "")
                elif t == "text":
                    text = str(p.get("text") or "")
    return video, image, text


def to_chatml(row: dict[str, Any], *, want_image: bool) -> dict[str, Any] | None:
    tk, rk = find_model_keys(row)
    gold = str(row.get("correct_option") or "").strip().upper()
    if not tk or not rk or not gold:
        return None

    # quality filter: the model's own answer must match gold
    if pred_letter(row[rk]) != gold:
        return None

    video, image_local, qtext = extract_user_parts(row)
    if not video or not video.startswith("oss://"):
        return None
    if not qtext:
        return None

    user_content: list[dict[str, Any]] = [{"type": "video", "video": video}]
    if want_image:
        if not image_local:
            return None
        image_oss = IMG_OSS_PREFIX + os.path.basename(image_local)
        user_content.append({"type": "image", "image": image_oss})


    qtext = re.sub(r"<image\s*\d+>[ \t]*", "", qtext)


    qtext = re.sub(
        r"Respond with only the letter \([^)]*\) of the correct option",
        "Respond with only the letter of the correct option",
        qtext,
    )
    user_content.append({"type": "text", "text": PREAMBLE + qtext})

    thinking = str(row.get(tk) or "").strip()
    assistant = f"<think>\n{thinking}\n</think>\n\n{gold}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant},
        ],
        "task": TASK,
    }


def convert(inputs: list[Path], output: Path, *, want_image: bool) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"in": 0, "kept": 0, "dropped_filter_or_bad": 0}
    ids_seen: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as fout:
        for inp in inputs:
            for line in inp.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                stats["in"] += 1
                row = json.loads(line)
                cm = to_chatml(row, want_image=want_image)
                if cm is None:
                    stats["dropped_filter_or_bad"] += 1
                    continue
                rid = str(row.get("id") or row.get("video_id") or "")
                ids_seen[rid] = ids_seen.get(rid, 0) + 1
                fout.write(json.dumps(cm, ensure_ascii=False) + "\n")
                stats["kept"] += 1
    stats["dup_ids"] = sum(1 for c in ids_seen.values() if c > 1)
    stats["unique_ids"] = len(ids_seen)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--with-image", action="store_true", help="Emit an OSS image part (adaptation_image).")
    args = ap.parse_args()
    stats = convert(args.inputs, args.output, want_image=args.with_image)
    print(f"[chatml] {args.output.name}  {stats}")


if __name__ == "__main__":
    main()
