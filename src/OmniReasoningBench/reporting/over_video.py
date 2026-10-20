"""Break a reasoning-over-video run down by answer_format / task_type / source.

Reads `score-judge` predictions.jsonl, including explicitly marked numeric fallbacks.
Accuracy is `correct / ALL rows`: missing outputs, unresolved judge failures, and
rows truncated before an answer count WRONG and stay in the denominator.
"""
import argparse
import csv
import json
import os
import sys

from .scored import outcome


def unwrap(x):
    return (x[0] if x else None) if isinstance(x, list) else x


def load(path):
    if not os.path.isfile(path):
        sys.exit(f"not found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return path, rows


def group(rows, key):
    agg = {}
    for r in rows:
        k = r.get(key) or "?"
        a = agg.setdefault(k, [0.0, 0, 0, 0])  # sum_acc, n, n_failed, n_truncated
        correct, failed = outcome(r)
        a[0] += correct
        a[1] += 1
        a[2] += failed
        er = unwrap(r.get("end_reason"))
        a[3] += int(r.get("method") == "no_answer_truncated" or (er is not None and er != "stop"))
    return agg


def emit(title, agg, writer=None):
    print(f"\n{title}")
    print(f"  {'group':<34}{'n':>6}{'acc%':>8}{'failed':>8}{'trunc':>7}")
    print("  " + "-" * 63)
    for k, (s, n, nf, nt) in sorted(agg.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        print(f"  {str(k):<34}{n:>6}{100 * s / n:>8.2f}{nf:>8}{nt:>7}")
        if writer:
            writer.writerow([title, k, n, round(100 * s / n, 2), nf, nt])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="scored output JSONL")
    ap.add_argument("--csv", default=None, help="also write the breakdown to this csv")
    ap.add_argument("--expected-count", type=int, default=None)
    args = ap.parse_args()

    path, rows = load(args.run)
    if not rows:
        sys.exit(f"no rows in {path}")

    fh = writer = None
    if args.csv:
        fh = open(args.csv, "w", encoding="utf-8", newline="")
        writer = csv.writer(fh)
        writer.writerow(["axis", "group", "n", "acc%", "failed", "truncated"])

    n = len(rows)
    acc = sum(outcome(r)[0] for r in rows)
    nf = sum(outcome(r)[1] for r in rows)
    print(f"===== Reasoning over video: {path} =====")
    print(f"rows={n}  acc={100 * acc / n:.2f}%  failed={nf} ({100 * nf / n:.1f}%)")
    if args.expected_count is not None and n != args.expected_count:
        print(f"NOTE: expected {args.expected_count} rows; this file has {n}.")

    for axis, key in (("by answer_format", "answer_format"),
                      ("by task_type", "task_type"),
                      ("by source", "source")):
        emit(axis, group(rows, key), writer)

    if fh:
        fh.close()
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
