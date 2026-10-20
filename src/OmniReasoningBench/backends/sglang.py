import argparse
import base64
import json
import mimetypes
import os
import pathlib
import re
import string
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".MP4")


def encode_video_as_data_url(video_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type:
        mime_type = "video/mp4"
    with open(video_path, "rb") as video_file:
        encoded = base64.b64encode(video_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_video_url(video_path: str, media_url_mode: str) -> str:
    if video_path.startswith(("http://", "https://", "file://", "data:")):
        return video_path
    if media_url_mode == "data":
        return encode_video_as_data_url(video_path)
    if media_url_mode == "path":
        return video_path
    return pathlib.Path(video_path).resolve().as_uri()


def clean_answer(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    original = text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        text = re.sub(r"<think>.*", "", original, flags=re.DOTALL).strip()
        if not text:
            return ""
    answer_tag = re.search(r"<answer>\s*\(?([A-H])\)?\s*</answer>", text, re.IGNORECASE)
    if answer_tag:
        return answer_tag.group(1).upper()
    boxed = re.search(r"\\?boxed\{([^}]+)\}|/box\{([^}]+)\}", text)
    if boxed:
        text = next(group for group in boxed.groups() if group)
    text = text.strip()
    if len(text) == 1 and text.upper() in "ABCDEFGH":
        return text.upper()
    short_match = re.match(r"^\s*\(?([A-H])\)?[\s.、:：;)\]]*$", text, re.IGNORECASE)
    if short_match:
        return short_match.group(1).upper()
    prefix_re = re.compile(
        r"(?:(?:the\s+)?(?:best|correct|final|right)\s+(?:answer|option|choice)\s+is|"
        r"answer\s*:|option\s*:|"
        r"i\s+(?:would\s+)?(?:choose|pick|select|go\s+with))"
        r"\s*:?\s*\(?([A-H])\)?",
        re.IGNORECASE,
    )
    prefix_match = prefix_re.search(text)
    if prefix_match:
        return prefix_match.group(1).upper()
    text_upper = text.upper()
    letters = re.findall(r"\b([A-H])\b", text_upper)
    if len(letters) == 1:
        return letters[0]
    if len(text.split()) <= 10 and letters:
        return letters[-1]
    return ""


def normalize_option_text(option: Any) -> str:
    option = str(option).strip()
    if len(option) >= 2 and option[0].isalpha() and option[1] in ".、)）":
        return option[2:].strip()
    return option


def parse_duration_seconds(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.endswith("s") and text[:-1].isdigit():
        return int(text[:-1])
    if text.startswith("<") and "min" in text:
        return 60
    if text.endswith("min") and text[:-3].isdigit():
        return int(text[:-3]) * 60
    parts = text.split(":")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return None


def resolve_video_path(video_root: str, video_id: str) -> str:
    raw = pathlib.Path(str(video_id))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(pathlib.Path(video_root) / str(video_id))
        if raw.suffix:
            candidates.append(pathlib.Path(video_root) / raw.name)
        else:
            candidates.extend(pathlib.Path(video_root) / f"{video_id}{ext}" for ext in VIDEO_EXTENSIONS)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def build_prompt(question: str, options: List[str]) -> str:
    return (
        "You are given a video. Based on both visual and audio content, answer the question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{os.linesep.join(options)}\n\n"
        "Return only the option letter, such as A, B, C, or D. Do not explain."
    )


def build_messages(video_path: str, prompt: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    video_url: Dict[str, Any] = {"url": get_video_url(video_path, args.media_url_mode)}
    if args.num_frames is not None:
        video_url["num_frames"] = args.num_frames
    return [
        {
            "role": "system",
            "content": "You are a helpful multimodal assistant. Answer multiple-choice video questions concisely.",
        },
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": video_url},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def request_completion(client: OpenAI, messages: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        modalities=["text"],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=False,
        extra_body={
            "mm_processor_kwargs": {
                "use_audio_in_video": not args.no_audio_in_video,
            }
        },
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()
    return (content or "").strip()


def iter_omnivideobench(data: Any, video_root: str) -> Iterable[Dict[str, Any]]:
    if not isinstance(data, list):
        raise TypeError("OmniVideoBench JSON must be a list.")
    for video_index, item in enumerate(data):
        video_id = item["video"]
        video_path = resolve_video_path(video_root, video_id)
        duration = parse_duration_seconds(item.get("duration"))
        for question_index, qa in enumerate(item.get("questions", [])):
            options = [str(option).strip() for option in qa["options"]]
            yield {
                "id": f"{video_id}_{question_index}",
                "benchmark": "omnivideobench",
                "video": video_id,
                "video_path": video_path,
                "duration": duration,
                "question": qa["question"],
                "options": options,
                "option_texts": [normalize_option_text(option) for option in options],
                "correct_answer": str(qa["correct_option"]).strip().upper(),
                "answer_text": qa.get("answer", ""),
                "video_type": item.get("video_type", ""),
                "question_type": qa.get("question_type", ""),
                "audio_type": qa.get("audio_type", ""),
                "source_index": video_index,
            }


def iter_worldsense(data: Any, video_root: str) -> Iterable[Dict[str, Any]]:
    if not isinstance(data, dict):
        raise TypeError("WorldSense JSON must be a dict keyed by video id.")
    for video_key, item in data.items():
        video_id = item.get("video_id", video_key)
        video_path = resolve_video_path(video_root, video_id)
        duration = parse_duration_seconds(item.get("video_duration") or item.get("duration"))
        for task_key, qa in sorted(item.items()):
            if not task_key.startswith("task") or not isinstance(qa, dict):
                continue
            options = [str(option).strip() for option in qa.get("candidates", [])]
            yield {
                "id": f"{video_id}_{task_key}",
                "benchmark": "worldsense",
                "video": video_id,
                "video_path": video_path,
                "duration": duration,
                "question": qa["question"],
                "options": options,
                "option_texts": [normalize_option_text(option) for option in options],
                "correct_answer": str(qa["answer"]).strip().upper(),
                "domain": item.get("domain", ""),
                "sub_category": item.get("sub_category", ""),
                "audio_class": item.get("audio_class", []),
                "task_domain": qa.get("task_domain", ""),
                "task_type": qa.get("task_type", ""),
            }



# ---------------------------------------------------------------------------
# OmniReasoningBench: benchmark record -> request conversion.
# convert_over_video / convert_beyond_video port the benchmark's own converters
# (prompt wrapper byte-identical between the two settings); build_part_messages /
# request_part_completion port how the reference harness turned those prompt parts
# into an OpenAI-compatible SGLang request and read the response back.
# ---------------------------------------------------------------------------

LETTERS = "ABCDEFGHIJ"

_OPT_PREFIX_RE = re.compile(r"^\s*[A-Ja-j]\s*[)）.、:：]\s*")

# System message of the reference evaluation runner (its USE_SYSTEM_STR default).
DEFAULT_SYSTEM_STR = (
    "Please think about this question as if you were a human pondering deeply, carefully "
    "considering the video information before answering, including self-reflection or "
    "verification in the reasoning process, providing the detailed reasoning between the "
    "<think> </think> tags, and finally giving the final answer."
)

# Prompt-token model used by the beyond-video converter to carve a per-row generation
# budget out of the serve's context window (SGLang rejects prompt + max_tokens > context
# instead of clamping):
#   prompt_tokens ~= VIDEO_TOK_PER_FRAME * min(SERVE_FRAME_CAP, SERVE_FPS * D)
#                    + AUDIO_TOK_PER_SEC * D + text
# Fitted against an SGLang-served Qwen3-Omni; both constants are rounded up so the
# estimate errs high. Other serves pin their own frame sampling and need their own cap.
VIDEO_TOK_PER_FRAME = 120.0
AUDIO_TOK_PER_SEC = 13.4
SERVE_FPS = 2.0
SERVE_FRAME_CAP = 256
TEXT_TOK_FLOOR = 320


def strip_opt_prefix(o):
    return _OPT_PREFIX_RE.sub("", str(o)).strip()


def build_benchmark_prompt(rec):
    if rec["answer_format"] == "open_numeric":
        return (rec["question"] + "\n\nCompute the single numeric value that answers the question. "
                "Give ONLY the number (unit optional), in the form 'Final Answer: <number>'.")
    lines = [rec["question"], ""]
    for i, o in enumerate(rec["options"]):
        lines.append(f"{LETTERS[i]}) {strip_opt_prefix(o)}")
    lines += ["", "Answer with the single letter of the correct option, in the form 'Final Answer: X'."]
    return "\n".join(lines)


def probe_duration(path):
    """ffprobe fallback for a row whose meta carries no probe.duration."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=120, check=True).stdout.strip()
        return float(out)
    except Exception:
        return None


def row_duration(rec, video):
    probe = ((rec.get("meta") or {}).get("video_raw") or {}).get("probe") or {}
    d = probe.get("duration")
    if isinstance(d, (int, float)) and d > 0:
        return float(d)
    return probe_duration(video or "")


def est_prompt_tokens(duration, prompt_text, frame_cap=None, tok_per_frame=None,
                      tok_per_sec=None):
    cap = SERVE_FRAME_CAP if frame_cap is None else frame_cap
    per_frame = VIDEO_TOK_PER_FRAME if tok_per_frame is None else tok_per_frame
    per_sec = AUDIO_TOK_PER_SEC if tok_per_sec is None else tok_per_sec
    nframes = min(cap, SERVE_FPS * duration)
    text = TEXT_TOK_FLOOR + len(prompt_text) / 4.0
    return int(per_frame * nframes + per_sec * duration + text)


def record_video(rec, video_root):
    video = str(rec.get("video_path") or rec.get("video") or "")
    if video.startswith(("http://", "https://", "file://", "data:")):
        return video
    return os.path.abspath(resolve_video_path(video_root, video))


def convert_over_video(rec, video_root):
    video = record_video(rec, video_root)
    return {
        "id": rec["id"],
        "source": rec.get("source"),
        "question": rec["question"],
        "options": rec.get("options") or [],
        "answer": rec.get("answer", ""),
        "answer_letters": rec.get("answer_letters") or [],
        "answer_format": rec.get("answer_format"),
        "task_type": rec.get("task_type"),
        "video_path": video,
        "prompt": [[{"type": "video", "video": video},
                    {"type": "text", "text": build_benchmark_prompt(rec)}]],
    }


def convert_beyond_video(rec, video_root, args):
    video = record_video(rec, video_root)
    text = build_benchmark_prompt(rec)
    parts = [{"type": "video", "video": video}]
    image = str(rec.get("image") or rec.get("image_path") or "")
    if image and not image.startswith(("http://", "https://", "file://", "data:")) and not os.path.isabs(image):
        image = os.path.abspath(pathlib.Path(video_root) / image)
    if not args.no_image and rec.get("requires_image"):
        if image and os.path.isfile(image):
            parts.append({"type": "image", "image": image})
        else:
            # Loud, not silent: a dual-dependency row without its figure is a
            # different (unanswerable) question; it is kept and runs video-only.
            print(f"WARNING: {rec['id']} needs a figure but none is on disk: {image or '<empty>'}")
    parts.append({"type": "text", "text": text})
    out = {
        "id": rec["id"],
        "source": rec.get("source"),
        "level": rec.get("level"),
        "question": rec["question"],
        "options": rec.get("options") or [],
        "answer": rec.get("answer", ""),
        "answer_letters": rec.get("answer_letters") or [],
        "answer_format": rec.get("answer_format"),
        "task_type": rec.get("task_type"),
        "form": rec.get("form"),
        "requires_image": bool(rec.get("requires_image")),
        "video_path": video,
        "image_path": image if len(parts) == 3 else "",
        "prompt": [parts],
    }
    if args.context_len:
        duration = row_duration(rec, video)
        if duration is None:
            # No duration => no way to size the budget; give it the floor rather
            # than guessing high and starving a possibly-short video.
            duration = 0.0
        est = est_prompt_tokens(duration, text, args.serve_frame_cap,
                                args.video_tok_per_frame, args.audio_tok_per_sec)
        budget = args.context_len - est - args.reserve
        out_seq = max(args.out_min, min(args.out_max, budget))
        out["video_duration"] = round(duration, 2)
        out["est_prompt_tokens"] = est
        # Per-row generation budget; overrides the global --max_tokens for this row.
        out["eval_args"] = {"out_seq_length": int(out_seq)}
    return out


def iter_benchmark(data, args):
    if isinstance(data, dict):
        data = [data]
    for rec in data:
        if args.benchmark == "over_video":
            yield convert_over_video(rec, args.video_dir)
        else:
            yield convert_beyond_video(rec, args.video_dir, args)


def media_url(value, args):
    if value.startswith(("http://", "https://", "file://", "data:")):
        return value
    return get_video_url(value, args.media_url_mode)


def build_part_messages(sample, args):
    content = []
    for part in sample["prompt"][0]:
        kind = part.get("type")
        if kind == "video":
            item = {"type": "video_url", "video_url": {"url": media_url(part["video"], args)}}
            if args.fps is not None:
                item["fps"] = args.fps
            if args.max_frames is not None:
                item["max_frames"] = args.max_frames
            if not args.no_audio_in_video:
                item["use_audio_in_video"] = True
            content.append(item)
        elif kind == "image":
            content.append({"type": "image_url", "image_url": {"url": media_url(part["image"], args)}})
        elif kind == "text":
            content.append({"type": "text", "text": part["text"]})
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def request_part_completion(client, messages, args, max_tokens):
    extra_body: Dict[str, Any] = {}
    flags = [bool(item["use_audio_in_video"]) for msg in messages if isinstance(msg["content"], list)
             for item in msg["content"] if "video" in item.get("type", "") and "use_audio_in_video" in item]
    if flags:
        extra_body["processor_kwargs"] = {"use_audio_in_video": flags[0] if len(flags) == 1 else flags}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    call_args: Dict[str, Any] = dict(model=args.model, messages=messages, temperature=args.temperature,
                                     stream=False, extra_body=extra_body)
    if args.top_p is not None:
        call_args["top_p"] = args.top_p
    if max_tokens > 0:
        call_args["max_tokens"] = max_tokens
    response = client.chat.completions.create(**call_args)
    choice = response.choices[0]
    result = choice.message.content or ""
    reasoning_content = getattr(choice.message, "reasoning_content", None)
    if reasoning_content:
        result = "<think>\n" + reasoning_content + "\n</think>\n\n" + result.lstrip("\n")
    return result, getattr(choice, "finish_reason", None)


def run_benchmark(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("--model (or SGLANG_MODEL) is required.")
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    with open(args.data_json_file, "r", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin if line.strip()] if args.data_json_file.endswith(".jsonl") else json.load(fin)
    samples = list(iter_benchmark(data, args))
    if args.limit is not None:
        samples = samples[: args.limit]
    results, processed_ids = load_existing_results(args.output_file)
    max_tokens = args.max_tokens if args.max_tokens is not None else 32768
    print(f"Using SGLang/OpenAI endpoint: {args.base_url}")
    print(f"Using model: {args.model}")
    print(f"Starting {args.benchmark} on {len(samples)} samples; resume results={len(results)}")

    for sample in tqdm(samples, desc=f"Evaluating {args.benchmark}"):
        if str(sample["id"]) in processed_ids:
            continue
        result = {k: v for k, v in sample.items() if k != "prompt"}
        video = sample["video_path"]
        if not video.startswith(("http://", "https://", "file://", "data:")) and not os.path.exists(video):
            result.update({"gen": [""], "end_reason": "Video file not found."})
        else:
            try:
                row_max = (sample.get("eval_args") or {}).get("out_seq_length", max_tokens)
                gen, finish_reason = request_part_completion(client, build_part_messages(sample, args), args, row_max)
                result.update({"gen": [gen], "end_reason": finish_reason})
            except Exception as exc:
                result.update({"gen": [""], "end_reason": f"Error: {type(exc).__name__} - {exc}"})
        results.append(result)
        processed_ids.add(str(sample["id"]))
        save_results(args.output_file, results)
        print(f"Progress: {len(results)}/{len(samples)}")

    print(f"Detailed results saved to: {args.output_file}")



def load_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    with open(args.data_json_file, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    if args.benchmark == "omnivideobench":
        iterator = iter_omnivideobench(data, args.video_dir)
    else:
        iterator = iter_worldsense(data, args.video_dir)
    samples = []
    for sample in iterator:
        if args.max_duration is not None and sample.get("duration") is not None and sample["duration"] >= args.max_duration:
            continue
        samples.append(sample)
        if args.limit is not None and len(samples) >= args.limit:
            break
    return samples


def load_existing_results(path: str) -> Tuple[List[Dict[str, Any]], set[str]]:
    if not os.path.exists(path):
        return [], set()
    with open(path, "r", encoding="utf-8") as fin:
        content = fin.read().strip()
    if not content:
        return [], set()
    results = [json.loads(line) for line in content.splitlines() if line.strip()] if path.endswith(".jsonl") else json.loads(content)
    return results, {str(item.get("id")) for item in results if item.get("id")}


def save_results(path: str, results: List[Dict[str, Any]]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        if path.endswith(".jsonl"):
            for row in results:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(results, fout, indent=2, ensure_ascii=False)


def result_template(sample: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "id",
        "benchmark",
        "video",
        "duration",
        "question",
        "options",
        "option_texts",
        "correct_answer",
        "answer_text",
        "video_type",
        "question_type",
        "audio_type",
        "domain",
        "sub_category",
        "audio_class",
        "task_domain",
        "task_type",
    ]
    return {key: sample[key] for key in keys if key in sample}


def print_progress(results: List[Dict[str, Any]], total: int) -> None:
    valid = [item for item in results if not str(item.get("model_answer", "")).startswith(("Error:", "Video file not found"))]
    correct = sum(1 for item in valid if item.get("is_correct"))
    accuracy = correct / len(valid) if valid else 0.0
    print(f"Progress: {len(results)}/{total}, valid={len(valid)}, accuracy={accuracy:.2%} ({correct}/{len(valid)})")


def run_evaluation(args: argparse.Namespace) -> None:
    if args.benchmark is None:
        raise ValueError("--benchmark is required when using the common runner directly.")
    if args.benchmark in ("over_video", "beyond_video"):
        return run_benchmark(args)
    if not args.model:
        raise ValueError("--model (or SGLANG_MODEL) is required.")
    if args.max_tokens is None:
        args.max_tokens = 32
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    samples = load_samples(args)
    results, processed_ids = load_existing_results(args.output_file)
    print(f"Using SGLang/OpenAI endpoint: {args.base_url}")
    print(f"Using model: {args.model}")
    print(f"Starting {args.benchmark} on {len(samples)} samples; resume results={len(results)}")

    for sample in tqdm(samples, desc=f"Evaluating {args.benchmark}"):
        if sample["id"] in processed_ids:
            continue
        result = result_template(sample)
        if not os.path.exists(sample["video_path"]):
            result.update({"model_answer": "Video file not found.", "raw_response": "", "is_correct": False})
        else:
            prompt = build_prompt(sample["question"], sample["options"])
            try:
                response_text = request_completion(client, build_messages(sample["video_path"], prompt, args), args)
                model_answer = clean_answer(response_text)
                result.update(
                    {
                        "model_answer": model_answer,
                        "raw_response": response_text,
                        "is_correct": model_answer == clean_answer(sample["correct_answer"]),
                    }
                )
            except Exception as exc:
                result.update(
                    {
                        "model_answer": f"Error: {type(exc).__name__} - {exc}",
                        "raw_response": "",
                        "is_correct": False,
                    }
                )
        results.append(result)
        processed_ids.add(sample["id"])
        save_results(args.output_file, results)
        print_progress(results, len(samples))

    print_progress(results, len(samples))
    print(f"Detailed results saved to: {args.output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate audiovisual benchmarks through an SGLang OpenAI-compatible server")
    parser.add_argument("--benchmark", choices=("over_video", "beyond_video", "omnivideobench", "worldsense"), default=None,
                        help="over_video / beyond_video: OmniReasoningBench records (JSON list or JSONL)")
    parser.add_argument("--data_json_file", required=True)
    parser.add_argument("--video_dir", required=True, help="root for relative video (and figure) paths")
    parser.add_argument("--output_file", required=True, help="results; .jsonl is written line by line")
    parser.add_argument("--base_url", default=os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1"))
    parser.add_argument("--api_key", default=os.environ.get("SGLANG_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("SGLANG_MODEL"), help="served model name (or SGLANG_MODEL)")
    parser.add_argument("--media_url_mode", choices=("file", "data", "path"), default="file")
    parser.add_argument("--num_frames", type=int, default=32, help="omnivideobench / worldsense only")
    parser.add_argument("--no_audio_in_video", action="store_true")
    parser.add_argument("--max_duration", type=int, default=None, help="omnivideobench / worldsense only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="default: 32768 for over_video / beyond_video, 32 for omnivideobench / worldsense")
    parser.add_argument("--temperature", type=float, default=0.0)
    bench = parser.add_argument_group("OmniReasoningBench (over_video / beyond_video)")
    bench.add_argument("--fps", type=float, default=2.0, help="per-video sampling fps sent with the video part")
    bench.add_argument("--max_frames", type=int, default=512, help="per-video frame cap sent with the video part")
    bench.add_argument("--top_p", type=float, default=None, help="not sent when unset")
    bench.add_argument("--top_k", type=int, default=None, help="not sent when unset")
    bench.add_argument("--system_prompt", default=DEFAULT_SYSTEM_STR,
                       help="system message; pass an empty string to send none")
    bench.add_argument("--no_image", action="store_true", help="beyond_video: drop the figure (video-only arm)")
    bench.add_argument("--context_len", type=int, default=None,
                       help="beyond_video: serve context length; enables the per-row generation budget")
    bench.add_argument("--out_max", type=int, default=16384, help="cap on the per-row budget")
    bench.add_argument("--out_min", type=int, default=1024, help="floor for the per-row budget")
    bench.add_argument("--reserve", type=int, default=512, help="tokens held back on top of the prompt estimate")
    bench.add_argument("--serve_frame_cap", type=int, default=SERVE_FRAME_CAP)
    bench.add_argument("--video_tok_per_frame", type=float, default=VIDEO_TOK_PER_FRAME)
    bench.add_argument("--audio_tok_per_sec", type=float, default=AUDIO_TOK_PER_SEC)
    return parser.parse_args()

if __name__ == "__main__":
    run_evaluation(parse_args())
