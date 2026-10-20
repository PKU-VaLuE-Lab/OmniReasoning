"""Break a reasoning-beyond-video run down by answer_format / figure / task_type.

Reads `score-judge` predictions.jsonl, including explicitly marked numeric fallbacks.
Besides the mcq-vs-open_numeric split, this setting has one more axis: whether the
row pairs the video with a new figure (`requires_image`).

Accuracy is `correct / ALL rows`: missing outputs, unresolved judge failures,
and rows truncated before an answer count WRONG and stay in the
denominator. Rows the endpoint could not serve at all (e.g. a long video whose
prompt exceeds the serve's context window) come back with no generation; they are
split out so the "servable rows only" number can be quoted beside the full one.
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
    warn_if_tail_only(path, rows)
    return path, rows


def warn_if_tail_only(path, rows):
    """A final output file may hold only the rows of the LAST (resumed) pass.

    Reading it alone would report that tail's accuracy as if it were the benchmark,
    so warn when the incremental `_turn.jsonl` beside it holds more ids.
    """
    turn = path[: -len(".jsonl")] + "_turn.jsonl" if path.endswith(".jsonl") else ""
    if not turn or not os.path.isfile(turn):
        return
    ids = set()
    with open(turn, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line).get("id"))
            except json.JSONDecodeError:
                pass
    if len(ids) > len(rows):
        print(f"WARNING: {os.path.basename(path)} holds {len(rows)} rows but "
              f"{os.path.basename(turn)} holds {len(ids)} distinct ids — this is a "
              f"tail-only 'final' file, so every number below is over those "
              f"{len(rows)} rows only.")


def gen_text(r):
    if "raw_output" in r:
        return r.get("raw_output") or ""
    g = r.get("gen")
    if isinstance(g, list):
        g = g[0] if g else ""
    return g or ""


def group(rows, keyfn):
    agg = {}
    for r in rows:
        k = keyfn(r)
        a = agg.setdefault(k, [0.0, 0, 0, 0, 0])  # sum_acc, n, failed, truncated, empty
        correct, failed = outcome(r)
        a[0] += correct
        a[1] += 1
        a[2] += failed
        er = unwrap(r.get("end_reason"))
        a[3] += int(r.get("method") == "no_answer_truncated" or (er is not None and er != "stop"))
        a[4] += int(not gen_text(r).strip())
    return agg


def emit(title, agg, writer=None):
    print(f"\n{title}")
    print(f"  {'group':<34}{'n':>6}{'acc%':>8}{'failed':>8}{'trunc':>7}{'empty':>7}")
    print("  " + "-" * 70)
    for k, (s, n, nf, nt, ne) in sorted(agg.items(),
                                        key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        print(f"  {str(k):<34}{n:>6}{100 * s / n:>8.2f}{nf:>8}{nt:>7}{ne:>7}")
        if writer:
            writer.writerow([title, k, n, round(100 * s / n, 2), nf, nt, ne])


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
        writer.writerow(["axis", "group", "n", "acc%", "failed", "truncated", "empty_gen"])

    n = len(rows)
    acc = sum(outcome(r)[0] for r in rows)
    nf = sum(outcome(r)[1] for r in rows)
    print(f"===== Reasoning beyond video: {path} =====")
    print(f"rows={n}  acc={100 * acc / n:.2f}%  failed={nf} ({100 * nf / n:.1f}%)")
    if args.expected_count is not None and n != args.expected_count:
        print(f"NOTE: expected {args.expected_count} rows; this file has {n}.")

    # Rows the endpoint never generated for (context overflow / hard error). They are
    # `failed` by the scorer, but for an input-side reason, so quote both numbers.
    unservable = [r for r in rows if not gen_text(r).strip()]
    if unservable:
        served = [r for r in rows if gen_text(r).strip()]
        s_acc = sum(outcome(r)[0] for r in served)
        print(f"no generation at all: {len(unservable)} row(s) "
              f"(context overflow / endpoint error)")
        print(f"servable rows only  : {len(served)} rows  acc={100 * s_acc / max(len(served), 1):.2f}%")
        longest = sorted(unservable, key=lambda r: -(r.get("est_prompt_tokens") or 0))[:8]
        for r in longest:
            print(f"    {str(r.get('id')):<44} D={r.get('video_duration')}s "
                  f"est_prompt={r.get('est_prompt_tokens')}")

    for axis, keyfn in (
        ("by answer_format", lambda r: r.get("answer_format") or "?"),
        ("by figure (requires_image)",
         lambda r: "video+figure" if r.get("sent_image", r.get("requires_image")) else "video only"),
        ("by task_type", lambda r: r.get("task_type") or "?"),
    ):
        emit(axis, group(rows, keyfn), writer)

    if fh:
        fh.close()
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
