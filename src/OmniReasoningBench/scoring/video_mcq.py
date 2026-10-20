"""Score video multiple-choice predictions and grouped results."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

from .judge import committed_answer

_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*\(?([A-Za-z])\)?\s*</answer>", re.IGNORECASE
)
_BOXED_RE = re.compile(r"\\?boxed\{([A-Za-z])\}", re.IGNORECASE)

_ANSWER_PREFIXES = [
    r"(?:the\s+)?(?:best|correct|final|right)\s+(?:answer|option|choice)\s+is\s*:?\s*",
    r"answer\s*:\s*",
    r"option\s*:\s*",
    r"i\s+(?:would\s+)?(?:choose|pick|select|go\s+with)\s+",
]
_PREFIX_RE = re.compile(
    r"(?:" + "|".join(_ANSWER_PREFIXES) + r")\(?([A-Z])\)?",
    re.IGNORECASE,
)


def extract_mcq_answer(text, valid_letters="ABCDEFGH"):
    if not text or not isinstance(text, str):
        return None
    text = committed_answer(text)
    if not text:
        return None

    valid_set = set(valid_letters.upper())

    def _valid(ch):
        ch = ch.upper()
        return ch if ch in valid_set else None

    m = _ANSWER_TAG_RE.search(text)
    if m:
        v = _valid(m.group(1))
        if v:
            return v

    m = _BOXED_RE.search(text)
    if m:
        v = _valid(m.group(1))
        if v:
            return v

    cleaned = text

    if len(cleaned) == 1:
        return _valid(cleaned)

    m = re.match(r"^\s*\(?([A-Za-z])\)?[\s.、:：;)\]，。]*$", cleaned)
    if m:
        return _valid(m.group(1))

    m = _PREFIX_RE.search(cleaned)
    if m:
        return _valid(m.group(1))

    letters = re.findall(r"\b([A-Z])\b", cleaned.upper())
    valid_found = [ch for ch in letters if ch in valid_set]

    if len(valid_found) == 1:
        return valid_found[0]

    if len(cleaned.split()) <= 10 and valid_found:
        return valid_found[-1]

    return None


def parse_duration_seconds(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return None


def get_duration_bucket(seconds):
    if seconds is None:
        return None
    if seconds < 60:
        return "<1min"
    elif seconds < 300:
        return "1min-5min"
    elif seconds < 600:
        return "5min-10min"
    else:
        return ">10min"


def score_file(turn_file):
    turn_file = str(turn_file)
    with open(turn_file, "r", encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f if line.strip()]

    total = len(jobs)
    correct = 0
    failed = 0

    by_video_type = defaultdict(lambda: {"total": 0, "correct": 0, "failed": 0})
    by_question_type = defaultdict(lambda: {"total": 0, "correct": 0, "failed": 0})
    by_audio_type = defaultdict(lambda: {"total": 0, "correct": 0, "failed": 0})
    by_duration = defaultdict(lambda: {"total": 0, "correct": 0, "failed": 0})

    for job in jobs:
        gen = job.get("model_answer", job.get("raw_output", job.get("gen", "")))
        if isinstance(gen, list):
            gen = gen[0] if gen else ""
        answer = str(job.get("correct_answer", job.get("answer", "")) or "")
        options = job.get("options", [])
        num_choices = len(options)
        valid_letters = "".join(chr(ord("A") + i) for i in range(num_choices)) if num_choices else "ABCDEFGH"

        provider_error = bool(job.get("error")) or str(gen).startswith(("Error:", "Video file not found"))
        extracted = None if provider_error else extract_mcq_answer(gen, valid_letters)

        is_correct = False
        is_failed = False
        if extracted is None or extracted == "":
            is_failed = True
        else:
            is_correct = extracted == answer.strip().upper()

        if is_correct:
            correct += 1
        if is_failed:
            failed += 1

        video_type = job.get("video_type", "unknown")
        question_type = job.get("question_type", "unknown")
        audio_type = job.get("audio_type", "unknown")
        duration = parse_duration_seconds(job.get("duration"))
        bucket = get_duration_bucket(duration)

        for key, group in [
            (video_type, by_video_type),
            (question_type, by_question_type),
            (audio_type, by_audio_type),
        ]:
            group[key]["total"] += 1
            if is_correct:
                group[key]["correct"] += 1
            if is_failed:
                group[key]["failed"] += 1

        if bucket:
            by_duration[bucket]["total"] += 1
            if is_correct:
                by_duration[bucket]["correct"] += 1
            if is_failed:
                by_duration[bucket]["failed"] += 1

    valid = total - failed
    acc = correct / total if total > 0 else 0.0
    acc_valid_only = correct / valid if valid > 0 else 0.0

    core_result = {
        "task_samples": {"omnivideobench": total},
        "omnivideobench/acc": acc,
        "omnivideobench/failed": failed,
    }

    print(f"\n{'='*60}")
    print(f"File: {turn_file}")
    print(f"{'='*60}")
    print(f"Total: {total}, Valid: {valid}, Correct: {correct}, Failed(empty gen): {failed}")
    print(f"Accuracy (all, failed=wrong): {acc:.4f} ({correct}/{total})")
    print(f"Accuracy (valid only):        {acc_valid_only:.4f} ({correct}/{valid})")

    print(f"\n--- By Duration ---")
    for bucket in ["<1min", "1min-5min", "5min-10min", ">10min"]:
        d = by_duration.get(bucket, {"total": 0, "correct": 0, "failed": 0})
        if d["total"] > 0:
            a = d["correct"] / d["total"]
            print(f"  {bucket:>12s}: {a:.4f} ({d['correct']}/{d['total']}, failed={d['failed']})")

    print(f"\n--- By Video Type ---")
    for vt, d in sorted(by_video_type.items(), key=lambda x: -x[1]["total"]):
        a = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        print(f"  {vt:>20s}: {a:.4f} ({d['correct']}/{d['total']})")

    print(f"\n--- By Question Type ---")
    for qt, d in sorted(by_question_type.items(), key=lambda x: -x[1]["total"]):
        a = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        print(f"  {qt:>30s}: {a:.4f} ({d['correct']}/{d['total']})")

    print(f"\n--- By Audio Type ---")
    for at, d in sorted(by_audio_type.items(), key=lambda x: -x[1]["total"]):
        a = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        print(f"  {at:>12s}: {a:.4f} ({d['correct']}/{d['total']})")


    tag_results = {}
    for bucket in ["<1min", "1min-5min", "5min-10min", ">10min"]:
        d = by_duration.get(bucket, {"total": 0, "correct": 0, "failed": 0})
        if d["total"] > 0:
            tag_key = f"core&{bucket}"
            tag_results[tag_key] = {
                "task_samples": {"omnivideobench": d["total"]},
                "omnivideobench/acc": d["correct"] / d["total"],
                "omnivideobench/failed": d["failed"],
            }


    result_dir = os.path.dirname(turn_file)
    metadata = {}
    for sibling in ["worldsense_output_eval_result.json", "dailyomni_output_eval_result.json", "videommmu_output_eval_result.json"]:
        sibling_path = os.path.join(result_dir, sibling)
        if os.path.exists(sibling_path):
            with open(sibling_path, "r", encoding="utf-8") as f:
                sibling_data = json.load(f)
            for field in ["model_name", "model_group", "model_args", "model_history",
                          "model_version", "model_arch", "model_path",
                          "is_sensitive_report", "infer_args", "workers", "labels"]:
                if field in sibling_data and field not in metadata:
                    metadata[field] = sibling_data[field]
            break

    import datetime
    eval_result = {
        "eval_time": datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y"),
        "total": {"num": total},
        "core": core_result,
        "all": core_result,
        "constant": {"zero": 0, "one": 1},
        "eval_dataset_name": "omnivideobench_qwen3_omni",
        "eval_dataset_MD5": "",
    }
    eval_result.update(tag_results)
    eval_result.update(metadata)

    input_path = Path(turn_file)
    if input_path.name.endswith("_output_turn.jsonl"):
        output_name = input_path.name.removesuffix("_output_turn.jsonl") + "_output_eval_result.json"
    else:
        output_name = input_path.stem + "_eval_result.json"
    eval_result_path = input_path.with_name(output_name)
    if (eval_result_path.resolve() == input_path.resolve()
            or (eval_result_path.exists() and os.path.samefile(eval_result_path, input_path))):
        raise ValueError("The score output must not overwrite the prediction file")
    with open(eval_result_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=4, ensure_ascii=False)
    print(f"\nEval result saved to: {eval_result_path}")

    return core_result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Score recorded video MCQ outputs with duration/type breakdowns")
    parser.add_argument("runs", nargs="+", help="Output JSONL files or result directories")
    args = parser.parse_args()
    for value in args.runs:
        path = Path(value)
        turn_file = path / "omnivideobench_output_turn.jsonl" if path.is_dir() else path
        if not turn_file.is_file():
            parser.error("Prediction file does not exist")
        score_file(str(turn_file))


if __name__ == "__main__":
    main()
