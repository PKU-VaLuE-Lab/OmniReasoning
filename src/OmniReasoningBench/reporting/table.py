"""Pool one system's two settings into the OmniReasoningBench table row.

Columns: over-video MCQ / OE, beyond-video MCQ / OE, Overall.

  * both settings: every reference item is in the denominator; an absent row, a row
    without a generation, or a provider/service error counts wrong. Correctness is
    `correct` (score-judge output), including explicitly marked numeric fallback.
    Legacy MCQ rows may use `acc >= 1`.
  * Overall pools by counts, not by averaging percentages.

usage: omnireasoning report-table --over_video_pred P1 --over_video_ref R1 \\
           --beyond_video_pred P2 --beyond_video_ref R2 [--label NAME]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scored import outcome


def jsonl(p: Path):
    text = p.read_text(encoding="utf-8")
    if p.suffix != ".jsonl":
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def over_video(pred: Path, fmt_of: dict[str, str]) -> dict | None:
    """Counts split by answer_format; absent and error rows stay in the denominator."""
    if not pred.exists() or pred.stat().st_size == 0:
        return None
    return beyond_video(pred, {i: {"answer_format": f} for i, f in fmt_of.items()})


def beyond_video(pred: Path, meta: dict[str, dict]) -> dict:
    """Counts split by answer_format; absent rows stay in the denominator."""
    by = {str(r["id"]): r for r in jsonl(pred)}
    d = {"mcq": [0, 0], "open_numeric": [0, 0]}
    for i in meta:
        fmt = meta[i]["answer_format"]
        f = "mcq" if fmt == "mcq" else "open_numeric"
        d[f][1] += 1
        r = by.get(str(i))
        if r is None:
            continue
        correct, _ = outcome({**r, "answer_format": fmt})
        d[f][0] += int(correct)
    return d


def pct(c: int, n: int) -> str:
    return f"{100 * c / n:.1f}" if n else "--"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--over_video_pred", required=True, help="scored over-video predictions")
    ap.add_argument("--over_video_ref", required=True, help="over-video benchmark records")
    ap.add_argument("--beyond_video_pred", required=True, help="scored beyond-video predictions")
    ap.add_argument("--beyond_video_ref", required=True, help="beyond-video benchmark records")
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    over = {r["id"]: r for r in jsonl(Path(args.over_video_ref))}
    fmt_of = {i: r["answer_format"] for i, r in over.items()}
    sub = jsonl(Path(args.beyond_video_ref))
    meta = {r["id"]: r for r in sub}
    print(f"over-video pool   = {len(over)}")
    print(f"beyond-video pool = {len(meta)} ("
          f"{sum(1 for r in sub if r['answer_format'] == 'mcq')} mcq + "
          f"{sum(1 for r in sub if r['answer_format'] == 'open_numeric')} open_numeric)\n")

    ov = over_video(Path(args.over_video_pred), fmt_of)
    b = beyond_video(Path(args.beyond_video_pred), meta)
    hdr = (f"| {'row':<30} | {'ov MCQ':>12} | {'ov OE':>12} | {'bv MCQ':>12} | "
           f"{'bv OE':>12} | {'Overall':>16} |")
    print(hdr)
    print("|" + "|".join("-" * len(c) for c in hdr.split("|")[1:-1]) + "|")
    cells = []
    for d, key in ((ov, "mcq"), (ov, "open_numeric"), (b, "mcq"), (b, "open_numeric")):
        if d is None:
            cells.append("tbd")
        else:
            c, n = d[key]
            cells.append(f"{pct(c, n)} ({c}/{n})")
    if ov and b:
        co = ov["mcq"][0] + ov["open_numeric"][0]
        no = ov["mcq"][1] + ov["open_numeric"][1]
        cb = b["mcq"][0] + b["open_numeric"][0]
        nb = b["mcq"][1] + b["open_numeric"][1]
        overall = f"{pct(co + cb, no + nb)} ({co + cb}/{no + nb})"
    else:
        overall = "tbd"
    print(f"| {args.label:<30} | " + " | ".join(f"{c:>12}" for c in cells) + f" | {overall:>16} |")


if __name__ == "__main__":
    main()
