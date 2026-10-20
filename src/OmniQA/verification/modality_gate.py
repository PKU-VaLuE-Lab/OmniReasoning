"""Modality-shortcut gate for generated audio-visual multiple-choice questions.

Each question is evaluated under restricted conditions built from its caption
bundle, plus a per-step media check:

  Stage 3.1  full-caption consistency (joint condition): a text model answers from
             the full audio+visual caption; must pick the gold option.
  Stage 3.2  visual-only bias filter: dual test -- (A) N_SINGLE_TRIALS answers from
             single random segments' visual captions (fail if >= SINGLE_MAJORITY
             correct) and (B) one answer from visual captions with the referenced
             segments removed. Passes if EITHER test passes.
  Stage 3.3  audio-only bias filter: the same dual test on audio captions.
  Stage 4    reasoning-step verification against the video: visual steps with the
             text/vision model, audio / audiovisual steps with the AV model.

A question passes overall iff 3.1 AND 3.2 AND 3.3 AND 4 pass. (The question-only
condition is the Stage-2 text-only blind gate and is not repeated here.)

Ported from the original validation script; the sampling-by-task-type driver was
replaced with Stage-2 accepted.jsonl input.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

RETRYABLE_STATUS = {408, 409, 418, 425, 429, 500, 502, 503, 504}

# Text model (Stage 3.1-3.3 and visual steps of Stage 4) and audio-visual model
# (non-visual steps of Stage 4). Both are overridable via env or CLI; one API key
# is used for both, as in the original script.
GPT_MODEL = os.environ.get("GATE_TEXT_MODEL", "gpt-5.5")
GPT_API_BASE = os.environ.get("GATE_TEXT_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

GEMINI_MODEL = os.environ.get("GATE_AV_MODEL", "gemini-3.1-pro-preview")
GEMINI_API_BASE = os.environ.get("GATE_AV_API_BASE", "")

# dataset/source name -> caption raw dir (".../<source>/raw"); filled from --caption-raw-dir.
CAPTION_DIRS: dict[str, str] = {}

# Fallback root for bundle-relative video paths; filled from --video-root.
VIDEO_ROOT = ""

_NEXT_API_START = 0.0
_THROTTLE_LOCK = threading.Lock()

# ─── API helpers ──────────────────────────────────────────────────────────────

def throttle_api(qps: float) -> None:
    global _NEXT_API_START
    if qps <= 0:
        return
    interval = 1.0 / qps
    with _THROTTLE_LOCK:
        now = time.monotonic()
        wait = max(0.0, _NEXT_API_START - now)
        _NEXT_API_START = max(now, _NEXT_API_START) + interval
    if wait > 0:
        time.sleep(wait)


def call_gpt(prompt: str, api_key: str, model: str | None = None,
             max_tokens: int = 256, qps: float = 1.0) -> dict[str, Any]:
    model = model or GPT_MODEL
    url = f"{GPT_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "input": prompt}
    return _do_request(url, headers, payload, qps=qps)


def call_gpt_with_video(video_url: str, prompt_text: str, api_key: str,
                        model: str | None = None, max_tokens: int = 512,
                        qps: float = 1.0) -> dict[str, Any]:
    model = model or GPT_MODEL
    url = f"{GPT_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_video", "video_url": video_url},
                    {"type": "input_text", "text": prompt_text},
                ],
            }
        ],
    }
    return _do_request(url, headers, payload, qps=qps, timeout=600)


def call_gemini_with_video(video_url: str, prompt_text: str, api_key: str,
                           model: str | None = None,
                           qps: float = 0.5) -> dict[str, Any]:
    model = model or GEMINI_MODEL
    url = f"{GEMINI_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "stream": False,
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"fileData": {"mimeType": "video/mp4", "fileUri": video_url}},
                    {"text": prompt_text},
                ],
            },
        ],
    }
    return _do_request(url, headers, payload, qps=qps, timeout=600)


def _do_request(url: str, headers: dict, payload: dict,
                max_retries: int = 5, backoff: float = 3.0,
                qps: float = 1.0, timeout: int = 300) -> dict[str, Any]:
    last_error = ""
    for attempt in range(max(1, max_retries)):
        throttle_api(qps)
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code < 400:
                return resp.json()
            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            if resp.status_code not in RETRYABLE_STATUS:
                break
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        sleep_sec = backoff * (attempt + 1)
        if attempt + 1 < max(1, max_retries):
            time.sleep(sleep_sec)
    raise RuntimeError(last_error)


def extract_response_text(resp: dict[str, Any]) -> str:
    # Responses API: output is a list with {type: "message", content: [{type: "output_text", text: "..."}]}
    output = resp.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            text = part.get("text", "")
                            if isinstance(text, str) and text.strip():
                                return text.strip()
                elif isinstance(content, str) and content.strip():
                    return content.strip()
            # Also try generic content extraction from list items
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    # dashscope input-mode dict format: {"output": {"text": "..."}}
    if isinstance(output, dict):
        text = output.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    # output_text at top level (some Responses API variants)
    output_text = resp.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    # OpenAI chat completions: {"choices": [{"message": {"content": "..."}}]}
    choices = resp.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        c = msg.get("content", "")
        if isinstance(c, str) and c.strip():
            return c.strip()
    # Gemini: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
    candidates = resp.get("candidates") or []
    if candidates:
        parts = (candidates[0].get("content") or {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
        if texts:
            return "\n".join(texts).strip()
    return json.dumps(resp, ensure_ascii=False)[:300]


def parse_option_letter(text: str, num_options: int = 8) -> str:
    valid = {chr(ord("A") + i) for i in range(num_options)}
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    patterns = [
        r"(?i)(?:answer|option|choice|答案|选项)\s*(?:is|:|：)?\s*\(?([A-H])\)?",
        r"^\s*\(?([A-H])\)?\s*$",
        r"^\s*\(?([A-H])\)?[\.)：:)、\s]",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match:
            letter = match.group(1).upper()
            if letter in valid:
                return letter
    return ""


def extract_json_from_text(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        if start < 0:
            raise
        stack, in_str, esc = [], False, False
        for idx, ch in enumerate(text[start:], start=start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if stack:
                        stack.pop()
                    if not stack:
                        return json.loads(text[start:idx + 1])
        raise


# ─── Caption formatting ──────────────────────────────────────────────────────

def format_full_caption(bundle: dict[str, Any]) -> str:
    parts = [f"Video Summary: {bundle.get('whole_video_summary', '')}"]
    for ec in bundle.get("event_captions", []):
        parts.append(f"\n=== Event {ec.get('event_id', '')} ({ec.get('global_timestamp', '')}) ===")
        parts.append(f"Summary: {ec.get('event_summary', '')}")
        if ec.get("audio_captions"):
            parts.append("Audio Captions:")
            for ac in ec["audio_captions"]:
                parts.append(f"  [{ac.get('timestamp', '')}] ({ac.get('audio_type', 'unknown')}): {ac.get('caption', '')}")
        if ec.get("visual_captions"):
            parts.append("Visual Captions:")
            for vc in ec["visual_captions"]:
                parts.append(f"  [{vc.get('timestamp', '')}]: {vc.get('caption', '')}")
        if ec.get("av_sync_pairs"):
            parts.append("AV Sync Pairs:")
            for avp in ec["av_sync_pairs"]:
                parts.append(f"  [{avp.get('timestamp', '')}] Audio: {avp.get('audio_caption', '')} | Visual: {avp.get('visual_caption', '')} ({avp.get('relation', '')})")
    return "\n".join(parts)


def format_visual_only_caption(bundle: dict[str, Any]) -> str:
    parts = []
    for ec in bundle.get("event_captions", []):
        if ec.get("visual_captions"):
            parts.append(f"[{ec.get('global_timestamp', '')}] Visual observations:")
            for vc in ec["visual_captions"]:
                parts.append(f"  [{vc.get('timestamp', '')}]: {vc.get('caption', '')}")
    return "\n".join(parts) if parts else "(no visual information available)"


def format_audio_only_caption(bundle: dict[str, Any]) -> str:
    parts = []
    for ec in bundle.get("event_captions", []):
        if ec.get("audio_captions"):
            parts.append(f"[{ec.get('global_timestamp', '')}] Audio observations:")
            for ac in ec["audio_captions"]:
                parts.append(f"  [{ac.get('timestamp', '')}] ({ac.get('audio_type', 'unknown')}): {ac.get('caption', '')}")
    return "\n".join(parts) if parts else "(no audio information available)"


def format_single_event_visual(event_caption: dict[str, Any]) -> str:
    parts = []
    if event_caption.get("visual_captions"):
        parts.append(f"[{event_caption.get('global_timestamp', '')}] Visual observations:")
        for vc in event_caption["visual_captions"]:
            parts.append(f"  [{vc.get('timestamp', '')}]: {vc.get('caption', '')}")
    return "\n".join(parts) if parts else "(no visual information for this segment)"


def format_single_event_audio(event_caption: dict[str, Any]) -> str:
    parts = []
    if event_caption.get("audio_captions"):
        parts.append(f"[{event_caption.get('global_timestamp', '')}] Audio observations:")
        for ac in event_caption["audio_captions"]:
            parts.append(f"  [{ac.get('timestamp', '')}] ({ac.get('audio_type', 'unknown')}): {ac.get('caption', '')}")
    return "\n".join(parts) if parts else "(no audio information for this segment)"


def format_question_prompt(question: str, options: list[str], hint: str = "A-H") -> str:
    lines = [question, ""]
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        clean = re.sub(r"^\s*\(?[A-Z][\.)：:)、]\s*", "", str(opt)).strip()
        lines.append(f"{letter}. {clean}")
    lines.append(f"\nAnswer with exactly one option letter ({hint}). Do not provide any explanation.")
    return "\n".join(lines)


# ─── Stage 3.1: Caption Consistency ──────────────────────────────────────────

STAGE_3_1_SYSTEM = (
    "You are given a detailed caption of a video (including audio and visual descriptions). "
    "Answer the multiple-choice question based ONLY on the caption information provided."
)


def run_stage_3_1(candidate: dict, bundle: dict, api_key: str) -> dict[str, Any]:
    caption_text = format_full_caption(bundle)
    question_text = format_question_prompt(
        candidate["question"], candidate["options"],
        f"A-{chr(ord('A') + len(candidate['options']) - 1)}"
    )
    prompt = f"{STAGE_3_1_SYSTEM}\n\n--- VIDEO CAPTION ---\n{caption_text}\n\n--- QUESTION ---\n{question_text}"

    try:
        resp = call_gpt(prompt, api_key, qps=8.0)
        raw_text = extract_response_text(resp)
        predicted = parse_option_letter(raw_text, len(candidate["options"]))
        gold = candidate["correct_option"].strip().upper()
        return {
            "stage": "3.1_caption_consistency",
            "predicted": predicted,
            "gold": gold,
            "is_correct": bool(predicted and predicted == gold),
            "pass": bool(predicted and predicted == gold),
            "raw_text": raw_text,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "stage": "3.1_caption_consistency",
            "predicted": "",
            "gold": candidate.get("correct_option", "").strip().upper(),
            "is_correct": False,
            "pass": False,
            "raw_text": "",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


# ─── Stage 3.2/3.3: Dual-test Bias Filters (v5.4) ───────────────────────────

STAGE_SINGLE_SEG_VL = (
    "You are given visual observations from a SHORT SEGMENT of a longer video. "
    "Most of the video's visual content is NOT shown. Audio is completely unavailable. "
    "Answer based SOLELY on what is described below. If insufficient, pick your best guess."
)
STAGE_COMPLEMENT_VL = (
    "You are given visual observations from SOME segments of a video. "
    "Several key segments are MISSING. Audio is completely unavailable. "
    "Answer based SOLELY on what is described below. If insufficient, pick your best guess."
)
STAGE_SINGLE_SEG_AU = (
    "You are given audio observations from a SHORT SEGMENT of a longer video. "
    "Most of the video's audio content is NOT provided. Visual content is completely unavailable. "
    "Answer based SOLELY on what is described below. If insufficient, pick your best guess."
)
STAGE_COMPLEMENT_AU = (
    "You are given audio observations from SOME segments of a video. "
    "Several key segments are MISSING. Visual content is completely unavailable. "
    "Answer based SOLELY on what is described below. If insufficient, pick your best guess."
)

N_SINGLE_TRIALS = 3
SINGLE_MAJORITY = 2


def _get_referenced_segment_ids(candidate: dict) -> set[str]:
    ids = set()
    for step in candidate.get("reasoning_steps", []):
        seg_id = step.get("segment_id", "")
        if seg_id and seg_id != "full_video":
            ids.add(seg_id)
    return ids


def _format_complement_visual(bundle: dict, exclude_ids: set[str]) -> str:
    parts = []
    for ec in bundle.get("event_captions", []):
        if ec.get("event_id", "") in exclude_ids:
            parts.append(f"[{ec.get('global_timestamp', '')}] (segment content not available)")
            continue
        if ec.get("visual_captions"):
            parts.append(f"[{ec.get('global_timestamp', '')}] Visual observations:")
            for vc in ec["visual_captions"]:
                parts.append(f"  [{vc.get('timestamp', '')}]: {vc.get('caption', '')}")
    return "\n".join(parts) if parts else "(no visual information available)"


def _format_complement_audio(bundle: dict, exclude_ids: set[str]) -> str:
    parts = []
    for ec in bundle.get("event_captions", []):
        if ec.get("event_id", "") in exclude_ids:
            parts.append(f"[{ec.get('global_timestamp', '')}] (segment content not available)")
            continue
        if ec.get("audio_captions"):
            parts.append(f"[{ec.get('global_timestamp', '')}] Audio observations:")
            for ac in ec["audio_captions"]:
                parts.append(f"  [{ac.get('timestamp', '')}] ({ac.get('audio_type', 'unknown')}): {ac.get('caption', '')}")
    return "\n".join(parts) if parts else "(no audio information available)"


def _run_single_bias_trial(prompt: str, candidate: dict, api_key: str) -> str:
    resp = call_gpt(prompt, api_key, qps=8.0)
    raw_text = extract_response_text(resp)
    return parse_option_letter(raw_text, len(candidate["options"]))


def run_stage_3_2(candidate: dict, bundle: dict, api_key: str) -> dict[str, Any]:
    import random
    events = bundle.get("event_captions", [])
    gold = candidate["correct_option"].strip().upper()
    question_text = format_question_prompt(
        candidate["question"], candidate["options"],
        f"A-{chr(ord('A') + len(candidate['options']) - 1)}"
    )
    exclude_ids = _get_referenced_segment_ids(candidate)

    try:
        # Test A: single-segment (v5.2 style)
        rng = random.Random(hash(candidate.get("question", "")))
        sampled = rng.sample(events, min(N_SINGLE_TRIALS, len(events)))
        while len(sampled) < N_SINGLE_TRIALS:
            sampled.append(rng.choice(events))
        single_preds = []
        for ev in sampled:
            cap = format_single_event_visual(ev)
            p = _run_single_bias_trial(
                f"{STAGE_SINGLE_SEG_VL}\n\n--- VISUAL (one segment) ---\n{cap}\n\n--- QUESTION ---\n{question_text}",
                candidate, api_key)
            single_preds.append(p)
        single_correct = sum(1 for p in single_preds if p == gold)
        single_pass = single_correct < SINGLE_MAJORITY

        # Test B: complement-segment (v5.3 style)
        comp_cap = _format_complement_visual(bundle, exclude_ids)
        comp_pred = _run_single_bias_trial(
            f"{STAGE_COMPLEMENT_VL}\n\n--- VISUAL (key segments excluded) ---\n{comp_cap}\n\n--- QUESTION ---\n{question_text}",
            candidate, api_key)
        comp_pass = not (comp_pred and comp_pred == gold)

        overall_pass = single_pass or comp_pass
        return {
            "stage": "3.2_vl_bias_filter",
            "single_preds": single_preds, "single_correct": single_correct, "single_pass": single_pass,
            "comp_pred": comp_pred, "comp_pass": comp_pass,
            "gold": gold, "excluded_segments": sorted(exclude_ids),
            "pass": overall_pass, "status": "ok",
        }
    except Exception as exc:
        return {
            "stage": "3.2_vl_bias_filter", "gold": gold,
            "pass": False, "status": "error", "error": f"{type(exc).__name__}: {exc}",
        }


def run_stage_3_3(candidate: dict, bundle: dict, api_key: str) -> dict[str, Any]:
    import random
    events = bundle.get("event_captions", [])
    gold = candidate["correct_option"].strip().upper()
    question_text = format_question_prompt(
        candidate["question"], candidate["options"],
        f"A-{chr(ord('A') + len(candidate['options']) - 1)}"
    )
    exclude_ids = _get_referenced_segment_ids(candidate)

    try:
        # Test A: single-segment
        rng = random.Random(hash(candidate.get("question", "")) + 1)
        sampled = rng.sample(events, min(N_SINGLE_TRIALS, len(events)))
        while len(sampled) < N_SINGLE_TRIALS:
            sampled.append(rng.choice(events))
        single_preds = []
        for ev in sampled:
            cap = format_single_event_audio(ev)
            p = _run_single_bias_trial(
                f"{STAGE_SINGLE_SEG_AU}\n\n--- AUDIO (one segment) ---\n{cap}\n\n--- QUESTION ---\n{question_text}",
                candidate, api_key)
            single_preds.append(p)
        single_correct = sum(1 for p in single_preds if p == gold)
        single_pass = single_correct < SINGLE_MAJORITY

        # Test B: complement-segment
        comp_cap = _format_complement_audio(bundle, exclude_ids)
        comp_pred = _run_single_bias_trial(
            f"{STAGE_COMPLEMENT_AU}\n\n--- AUDIO (key segments excluded) ---\n{comp_cap}\n\n--- QUESTION ---\n{question_text}",
            candidate, api_key)
        comp_pass = not (comp_pred and comp_pred == gold)

        overall_pass = single_pass or comp_pass
        return {
            "stage": "3.3_audio_bias_filter",
            "single_preds": single_preds, "single_correct": single_correct, "single_pass": single_pass,
            "comp_pred": comp_pred, "comp_pass": comp_pass,
            "gold": gold, "excluded_segments": sorted(exclude_ids),
            "pass": overall_pass, "status": "ok",
        }
    except Exception as exc:
        return {
            "stage": "3.3_audio_bias_filter", "gold": gold,
            "pass": False, "status": "error", "error": f"{type(exc).__name__}: {exc}",
        }


# ─── Stage 4: Reasoning Step Post-Verification ──────────────────────────────

def parse_timestamp_range(ts: str) -> tuple[float, float] | None:
    m = re.match(r"(\d+):(\d+)\s*-\s*(\d+):(\d+)", ts.strip())
    if not m:
        return None
    start = int(m.group(1)) * 60 + int(m.group(2))
    end = int(m.group(3)) * 60 + int(m.group(4))
    return float(start), float(end)


def extract_clip(video_path: str, timestamp: str, output_path: str,
                 padding: float = 1.0, max_sec: float = 30.0,
                 silent: bool = False) -> dict[str, Any]:
    parsed = parse_timestamp_range(timestamp)
    if not parsed:
        raise ValueError(f"Cannot parse timestamp: {timestamp}")
    start, end = parsed
    start = max(0.0, start - padding)
    end = max(start + 0.5, end + padding)
    duration = end - start
    if max_sec > 0 and duration > max_sec:
        center = (start + end) / 2.0
        start = max(0.0, center - max_sec / 2.0)
        end = start + max_sec

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{max(0.5, end - start):.3f}",
    ]
    if silent:
        cmd += ["-map", "0:v:0", "-an"]
    cmd += [
        "-c:v", "libopenh264", "-pix_fmt", "yuv420p",
        "-vf", "fps=2,scale='min(640,iw)':-2",
        "-movflags", "+faststart",
        output_path,
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {proc.stderr[:500]}")
    return {"start_sec": start, "end_sec": end, "duration_sec": end - start, "silent": silent}


def video_to_base64_url(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def sign_oss_url(oss_path: str, timeout: int = 3600) -> str:
    cmd = ["ossutil", "sign", oss_path, "--timeout", str(timeout)]
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ossutil sign failed: {proc.stderr or proc.stdout}")
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            return line
    raise RuntimeError(f"ossutil sign no URL: {proc.stdout}")


def _video_url(video_path: str) -> str:
    """oss:// paths are signed with ossutil (original behaviour); a local file is
    sent inline as a base64 data URL instead."""
    if video_path.startswith("oss://"):
        return sign_oss_url(video_path)
    return video_to_base64_url(video_path)


VISUAL_VERIFY_PROMPT = """You are a visual hallucination detector. You will be given a video and a visual description from a reasoning step.

Your task: Check whether the CORE VISUAL CONTENT described actually appears in the video.

IMPORTANT — what to verify:
- Whether described actions, movements, and events actually happen in the video
- Whether described objects, scenes, and spatial layouts are visible
- Whether described on-screen text or graphics actually appear (if claimed)

IMPORTANT — what to IGNORE (do NOT count as hallucinations):
- Person names and identities — models cannot identify people by face, names come from audio/context
- Speaker attribution — who said what is an audio concern, not visual
- Minor differences in exact wording of on-screen text
- Subtle details that are hard to see due to video quality or resolution
- Any audio-related claims (speech, music, sound)

Only flag as hallucination if a described visual event/action/object is COMPLETELY ABSENT or CONTRADICTED by the video.

Visual description to verify:
{content}

Return strict JSON:
{{"pass": true/false, "hallucinations": ["list ONLY fabricated visual content that is completely absent or contradicted"], "rationale": "one sentence"}}"""

AUDIO_AV_VERIFY_PROMPT = """You are an audio-visual verification expert. You will be given a video clip (WITH audio) and a description from a reasoning step.

Your task: Check whether the CORE CONTENT described matches what is shown and heard in the video.

IMPORTANT — what to verify:
- Whether described actions and events actually happen
- Whether described speech content approximately matches what is said (exact wording not required)
- Whether described sounds or music are present
- Whether described visual scenes and objects are visible

IMPORTANT — what to IGNORE (do NOT count as hallucinations):
- Person names and identities — unless the speaker explicitly introduces themselves in audio
- Minor paraphrasing differences in speech (focus on meaning, not exact words)
- Subtle visual details hard to see due to video quality
- Exact timestamps within the segment (small offsets are fine)

Only flag as hallucination if described content is COMPLETELY ABSENT or CONTRADICTED by the video.

Description to verify (modality: {modality}):
{content}

Return strict JSON:
{{"pass": true/false, "hallucinations": ["list ONLY fabricated content completely absent or contradicted"], "rationale": "one sentence"}}"""


def verify_visual_step(step: dict, oss_path: str,
                       api_key: str) -> dict[str, Any]:
    step_id = step.get("step_id", "step")
    timestamp = step.get("timestamp", "")
    content = step.get("content", "")

    try:
        signed_url = _video_url(oss_path)
        prompt = VISUAL_VERIFY_PROMPT.format(content=content)
        prompt += f"\n\nNote: Focus on the video segment at timestamp {timestamp}. Only verify the visual content in this time range."

        resp = call_gpt_with_video(signed_url, prompt, api_key, qps=4.0)
        raw_text = extract_response_text(resp)
        result = extract_json_from_text(raw_text)
        hallucinations = result.get("hallucinations", [])
        if not isinstance(hallucinations, list):
            hallucinations = [str(hallucinations)] if hallucinations else []
        passed = bool(result.get("pass")) and not hallucinations
        return {
            "step_id": step_id, "modality": "visual", "timestamp": timestamp,
            "content": content, "model": GPT_MODEL,
            "pass": passed,
            "hallucinations": hallucinations,
            "rationale": result.get("rationale", ""),
            "raw_text": raw_text,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "step_id": step_id, "modality": "visual", "timestamp": timestamp,
            "content": content, "model": GPT_MODEL,
            "pass": False, "hallucinations": [], "rationale": "",
            "raw_text": "", "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def verify_audio_av_step(step: dict, oss_path: str,
                         api_key: str) -> dict[str, Any]:
    step_id = step.get("step_id", "step")
    timestamp = step.get("timestamp", "")
    content = step.get("content", "")
    modality = step.get("modality", "audiovisual")

    try:
        signed_url = _video_url(oss_path)
        prompt = AUDIO_AV_VERIFY_PROMPT.format(modality=modality, content=content)
        prompt += f"\n\nNote: The relevant segment is at timestamp {timestamp} in the video. Focus your verification on this time range."

        resp = call_gemini_with_video(signed_url, prompt, api_key, qps=2.0)
        raw_text = extract_response_text(resp)
        result = extract_json_from_text(raw_text)
        hallucinations = result.get("hallucinations", [])
        if not isinstance(hallucinations, list):
            hallucinations = [str(hallucinations)] if hallucinations else []
        passed = bool(result.get("pass")) and not hallucinations
        return {
            "step_id": step_id, "modality": modality, "timestamp": timestamp,
            "content": content, "model": GEMINI_MODEL,
            "pass": passed,
            "hallucinations": hallucinations,
            "rationale": result.get("rationale", ""),
            "raw_text": raw_text,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "step_id": step_id, "modality": modality, "timestamp": timestamp,
            "content": content, "model": GEMINI_MODEL,
            "pass": False, "hallucinations": [], "rationale": "",
            "raw_text": "", "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_stage_4(candidate: dict, bundle: dict, oss_path: str,
                api_key: str) -> dict[str, Any]:
    steps = candidate.get("reasoning_steps", [])
    results = []
    for step in steps:
        modality = (step.get("modality") or "").lower()
        if modality == "visual":
            r = verify_visual_step(step, oss_path, api_key)
        else:
            r = verify_audio_av_step(step, oss_path, api_key)
        results.append(r)
    all_pass = bool(results) and all(r.get("pass") for r in results)
    return {
        "stage": "4_reasoning_step_verification",
        "pass": all_pass,
        "step_count": len(results),
        "steps_passed": sum(1 for r in results if r.get("pass")),
        "steps_failed": sum(1 for r in results if not r.get("pass")),
        "step_results": results,
    }




# ─── Caption bundle / video lookup ───────────────────────────────────────────

def load_bundle(dataset: str, source_id: str) -> dict[str, Any]:
    path = os.path.join(CAPTION_DIRS[dataset], source_id, "caption_bundle.json")
    with open(path) as f:
        return json.load(f)


def resolve_video(bundle: dict[str, Any]) -> str:
    for field in ("time_video_local_path", "raw_video_local_path"):
        vp = bundle.get(field, "")
        if vp and os.path.exists(vp):
            return vp
    for field in ("time_video_local_path", "raw_video_local_path"):
        vp = bundle.get(field, "")
        if vp and VIDEO_ROOT:
            abs_path = os.path.join(VIDEO_ROOT, vp)
            if os.path.exists(abs_path):
                return abs_path
    raise FileNotFoundError(f"No video found for {bundle.get('video_id')}")


# ─── Process one candidate ───────────────────────────────────────────────────

def process_candidate(candidate: dict, api_key: str,
                      skip_stage4: bool = False) -> dict[str, Any]:
    dataset = candidate["_dataset"]
    source_id = candidate["source_id"]
    bundle = load_bundle(dataset, source_id)

    result = {
        "dataset": dataset,
        "source_id": source_id,
        "task_type": candidate["task_type"],
        "question": candidate["question"],
        "correct_option": candidate["correct_option"],
    }

    result["stage_3_1"] = run_stage_3_1(candidate, bundle, api_key)
    result["stage_3_2"] = run_stage_3_2(candidate, bundle, api_key)
    result["stage_3_3"] = run_stage_3_3(candidate, bundle, api_key)

    if not skip_stage4:
        oss_path = bundle.get("time_video_oss_path", "")
        if not oss_path:
            # Local run: fall back to the bundle's local video (sent as a data URL).
            try:
                oss_path = resolve_video(bundle)
            except FileNotFoundError:
                oss_path = ""
        if oss_path:
            try:
                result["stage_4"] = run_stage_4(candidate, bundle, oss_path, api_key)
            except Exception as exc:
                result["stage_4"] = {"stage": "4", "pass": False, "error": str(exc)}
        else:
            result["stage_4"] = {"stage": "4", "pass": False, "error": "no oss path"}

    return result


def overall_pass(result: dict[str, Any]) -> bool:
    return bool(result["stage_3_1"]["pass"] and result["stage_3_2"]["pass"]
                and result["stage_3_3"]["pass"] and result.get("stage_4", {}).get("pass"))


# ─── Stage-2 record -> candidate ─────────────────────────────────────────────

def candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map one Stage-2 accepted.jsonl row onto the candidate fields used above.

    question / options / correct_option / reasoning_steps are read from
    row["candidate"] unchanged; the caption-bundle key is (source, video_id)."""
    cand = dict(row.get("candidate") or {})
    cand["_dataset"] = str(row.get("source", "")).split("::")[0]
    cand["source_id"] = row.get("video_id") or cand.get("video_id", "")
    cand["task_type"] = cand.get("task_type") or row.get("task_type", "")
    return cand


def build_caption_dirs(spec: str) -> dict[str, str]:
    """--caption-raw-dir: comma-separated '<...>/<source>/raw' dirs, keyed by <source>
    exactly as Stage 2 names its sources."""
    dirs: dict[str, str] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        d = Path(part)
        dirs[d.parent.name if d.name == "raw" else d.name] = str(d)
    return dirs


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global GPT_MODEL, GPT_API_BASE, GEMINI_MODEL, GEMINI_API_BASE, VIDEO_ROOT
    parser = argparse.ArgumentParser(
        description="Modality-shortcut gate: full-caption consistency (3.1), visual-only (3.2) "
                    "and audio-only (3.3) bias filters, per-step media verification (4)")
    parser.add_argument("--input", type=Path, required=True,
                        help="accepted.jsonl from Stage 2 (OmniQA.stage2_generate)")
    parser.add_argument("--caption-raw-dir", required=True,
                        help="comma-separated caption raw dirs ('<...>/<source>/raw'), same as Stage 2")
    parser.add_argument("--video-root", default="",
                        help="optional root for bundle-relative time/raw video paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-stage4", action="store_true", help="Skip stage 4 (requires video)")
    parser.add_argument("--workers", "--max-workers", dest="max_workers", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all")
    parser.add_argument("--text-model", default=GPT_MODEL)
    parser.add_argument("--text-api-base", default=GPT_API_BASE)
    parser.add_argument("--av-model", default=GEMINI_MODEL)
    parser.add_argument("--av-api-base", default=GEMINI_API_BASE,
                        help="endpoint accepting Gemini-style 'contents' payloads at /chat/completions")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    GPT_MODEL, GPT_API_BASE = args.text_model, args.text_api_base
    GEMINI_MODEL, GEMINI_API_BASE = args.av_model, args.av_api_base
    VIDEO_ROOT = args.video_root
    CAPTION_DIRS.update(build_caption_dirs(args.caption_raw_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Modality-shortcut gate (dual-test bias filters)")
    print("=" * 80)
    print(f"GPT Model: {GPT_MODEL}")
    print(f"Gemini Model: {GEMINI_MODEL}")
    print(f"Output: {args.output_dir}")

    run_rows(args, api_key)


def run_rows(args, api_key: str):
    from collections import defaultdict

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.max_rows > 0:
        rows = rows[:args.max_rows]
    total = len(rows)
    print(f"\nLoaded {total} questions from {args.input}\n")

    all_results = []
    task_results: dict[str, list] = defaultdict(list)
    completed = 0
    all_path = args.output_dir / "all_results.jsonl"
    acc_path = args.output_dir / "accepted_gated.jsonl"

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor, \
            open(all_path, "w", encoding="utf-8") as fa, open(acc_path, "w", encoding="utf-8") as fo:
        futures = {}
        for row in rows:
            item = candidate_from_row(row)
            future = executor.submit(process_candidate, item, api_key, args.skip_stage4)
            futures[future] = (item["task_type"], item, row)

        for future in as_completed(futures):
            tt, item, row = futures[future]
            completed += 1
            try:
                result = future.result()
                result["overall_pass"] = overall_pass(result)
                all_results.append(result)
                task_results[tt].append(result)
                fa.write(json.dumps(result, ensure_ascii=False) + "\n")
                if result["overall_pass"]:
                    fo.write(json.dumps({**row, "modality_gate": result}, ensure_ascii=False) + "\n")
                s31 = "P" if result["stage_3_1"]["pass"] else "F"
                s32 = "P" if result["stage_3_2"]["pass"] else "F"
                s33 = "P" if result["stage_3_3"]["pass"] else "F"
                s4 = "P" if result.get("stage_4", {}).get("pass") else "F"
                print(f"  [{completed}/{total}] {tt[:25]:25s} | 3.1={s31} 3.2={s32} 3.3={s33} 4={s4} | {item['source_id'][:30]}")
            except Exception as exc:
                fa.write(json.dumps({"dataset": item.get("_dataset"), "source_id": item.get("source_id"),
                                     "task_type": tt, "question": item.get("question"),
                                     "status": "error", "error": f"{type(exc).__name__}: {exc}",
                                     "overall_pass": False}, ensure_ascii=False) + "\n")
                print(f"  [{completed}/{total}] {tt[:25]:25s} | ERROR: {exc}")

    # Per-task-type summary
    print(f"\n{'=' * 90}")
    print("PER-TASK-TYPE RESULTS")
    print(f"{'=' * 90}")
    header = f"{'Task Type':<30s} | {'N':>3s} | {'3.1 Cap':>7s} | {'3.2 VL':>7s} | {'3.3 Aud':>7s} | {'4 Verify':>8s} | {'All Pass':>8s}"
    print(header)
    print("─" * len(header))

    grand_totals = {"n": 0, "s31": 0, "s32": 0, "s33": 0, "s4": 0, "all": 0}
    for tt in sorted(task_results.keys()):
        results = task_results[tt]
        n = len(results)
        s31 = sum(1 for r in results if r["stage_3_1"]["pass"])
        s32 = sum(1 for r in results if r["stage_3_2"]["pass"])
        s33 = sum(1 for r in results if r["stage_3_3"]["pass"])
        s4 = sum(1 for r in results if r.get("stage_4", {}).get("pass"))
        all_pass = sum(1 for r in results
                       if r["stage_3_1"]["pass"] and r["stage_3_2"]["pass"]
                       and r["stage_3_3"]["pass"] and r.get("stage_4", {}).get("pass"))
        print(f"{tt:<30s} | {n:>3d} | {s31:>3d}/{n:<3d} | {s32:>3d}/{n:<3d} | {s33:>3d}/{n:<3d} | {s4:>4d}/{n:<3d} | {all_pass:>4d}/{n:<3d}")
        grand_totals["n"] += n
        grand_totals["s31"] += s31
        grand_totals["s32"] += s32
        grand_totals["s33"] += s33
        grand_totals["s4"] += s4
        grand_totals["all"] += all_pass

    n = grand_totals["n"]
    print("─" * len(header))
    print(f"{'TOTAL':<30s} | {n:>3d} | {grand_totals['s31']:>3d}/{n:<3d} | {grand_totals['s32']:>3d}/{n:<3d} | "
          f"{grand_totals['s33']:>3d}/{n:<3d} | {grand_totals['s4']:>4d}/{n:<3d} | {grand_totals['all']:>4d}/{n:<3d}")

    # Pass rate summary
    print(f"\n{'=' * 60}")
    print("PASS RATE SUMMARY")
    print(f"{'=' * 60}")
    for stage, key in [("3.1 Caption Consistency", "s31"), ("3.2 VL Bias Filter", "s32"),
                       ("3.3 Audio Bias Filter", "s33"), ("4   Reasoning Verify", "s4")]:
        rate = grand_totals[key] / n * 100 if n else 0
        print(f"  Stage {stage}: {grand_totals[key]}/{n} ({rate:.1f}%)")
    rate = grand_totals["all"] / n * 100 if n else 0
    print(f"  Overall Pipeline Pass:    {grand_totals['all']}/{n} ({rate:.1f}%)")

    summary = {
        "input_rows": total,
        "errors": total - len(all_results),
        "per_task_type": {
            tt: {
                "total": len(results),
                "stage_3_1_pass": sum(1 for r in results if r["stage_3_1"]["pass"]),
                "stage_3_2_pass": sum(1 for r in results if r["stage_3_2"]["pass"]),
                "stage_3_3_pass": sum(1 for r in results if r["stage_3_3"]["pass"]),
                "stage_4_pass": sum(1 for r in results if r.get("stage_4", {}).get("pass")),
            }
            for tt, results in task_results.items()
        },
        "totals": grand_totals,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {all_path}")
    print(f"Accepted (overall pass): {acc_path}")


if __name__ == "__main__":
    main()
