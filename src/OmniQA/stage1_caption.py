"""Generate timestamped audio-visual captions and event plans."""
from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import dataclasses
import functools
import threading
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import requests
from tqdm import tqdm


VIDEO_STRUCTURE_PROMPT_TEMPLATE = """You are analyzing one full video before fine-grained omni captioning.

Task:
Watch the whole video once and produce only the lightweight structure needed for detailed captioning:
1. a concise whole-video summary
2. a specific video type/category
3. major event segmentation with timestamps

Requirements:
1. Segment by semantic event changes, not fixed time windows.
2. Use original global timestamps.
3. `video_type` must be specific, not generic. Prefer concrete labels such as "NBA basketball game highlight analysis", "Harry Potter movie lore explainer", "makeup tutorial", "first-person cooking tutorial", "Minecraft gameplay commentary", "classroom math lecture", "product unboxing review", or "news interview". Include the sport/game/topic/domain when identifiable.
4. Every event summary must mention important visual evidence and important audio evidence.
5. Each event segment must be at most 3 minutes long. If a semantic event lasts longer than 3 minutes, split it into consecutive sub-segments at natural boundaries near the 3-minute limit.
6. Prefer 3 to {max_events} events, but respect the 3-minute maximum segment length.
7. Do not generate question profiles, question hints, candidate questions, labels, scores, or reasoning tasks in Stage 1.
8. Prioritize caption completeness over task planning.

Return strict JSON only:
{{
  "video_summary": "video summary",
  "video_type": "specific video type/category with domain/topic/sport/game when identifiable",
  "event_segments": [
    {{
      "event_id": "event_01",
      "start": "MM:SS",
      "end": "MM:SS",
      "summary": "key visual and audio details, including speech/music/sound if present"
    }}
  ]
}}
"""

EVENT_OMNI_CAPTION_PROMPT_TEMPLATE = """You are captioning one semantically coherent event clip from a longer video. Act as a high-recall audiovisual sensor: decompose what you see and hear into fine-grained, retrieval-ready facts. Do not summarize and do not use evaluative framing such as "emphasizes", "showcases", "conveys", or "creates a sense of".

Context:
 - Global event range: {event_start_text}-{event_end_text}
 - Previous event captions for reference resolution: {prior_event_caption_context}

Task:
Return one event-level omni caption plus fine-grained audio-only and visual-only timestamped captions. Use original global timestamps.

Required output:
- `global_timestamp`: full event range.
- `global_caption`: a rich event-level omni caption integrating what is seen and heard in temporal order — scene changes, entities, actions, visible text, audio, and audio-visual relations. Do not merely concatenate the modality captions.
- `audio_captions`: time-ordered audio-only items, split when speech/speaker/music/sound/silence/unclear audio changes.
- `visual_captions`: time-ordered visual-only items, split when shot/scene/subject/action/object/text/layout/state changes.

Visual detail requirements (every visual caption must be fine-grained and specific):
1. Subjects/entities: give the exact count (state the number, never "several"/"some"). For each person: gender, approximate age, build, hair, clothing with precise colors and style, accessories, and position in frame.
2. Colors: name the precise shade of key items (reddish-orange, olive, teal, fuchsia, coral, navy, maroon) — never the base color alone or "light-colored".
3. Objects: exact color/shade, shape/geometry, material/finish, and sub-type/brand/model when identifiable. Use domain-specific names ("fedora" not "hat", "ape-hanger handlebars" not "handlebars", "chevron-shaped gold pendant" not "necklace", "Nike swoosh logo" not "logo").
4. Actions: body-part specifics (which hand/foot/side and direction), temporal phases (preparation/execution/follow-through), action-object interaction (grip, contact surface, tool, object state before -> after), speed/dynamics, multi-person interaction (who initiates/responds, gaze direction, proximity), communicative gestures (pointing, nodding, waving, thumbs-up), and exact repetition counts.
5. Spatial layout: left/right arrangement, foreground/midground/background, facing directions, depth relations.
6. Camera/editing: shot size, camera movement, and the transition type between shots (hard cut/jump cut/dissolve/fade/wipe).
7. On-screen text/graphics/UI/signs/logos: transcribe verbatim in the original language, with position and appearance/disappearance timing. Transcribe formulas, code, tables, and numbers exactly.
8. State changes: objects/scenes that transform, appear, or disappear.

Audio detail requirements (every audio caption must be fine-grained and specific):
1. If the sound source can be identified (person/object/animal/place), start the caption with a specific referring description of that source, e.g. "The host in the blue shirt says...", "The off-screen narrator says...", "The car engine revs...", "The phone on the table rings...". Avoid generic labels unless nothing more specific is supported.
2. Speech: transcribe only the words audible in that window, verbatim in the original language; give speaker identity, language/accent, and tone/voice character (baritone, soprano, raspy, breathy) when perceivable. Mark unclear words as `[unclear]`.
3. Music: never just "background music" — give genre, tempo/mood, and name distinguishable instruments or synth voices (do not generalize to "drums" if a hi-hat or snare is distinguishable).
4. Sound effects: name each effect with its texture (whine, hiss, thud, room echo).
5. Silence: report silence intervals explicitly.
6. Do not hallucinate audio: do not assume music from visible instruments, or speech from mouth movement. If you cannot clearly hear it, mark it `[unclear]`; if the clip is silent, say so.

Rules:
1. Keep audio captions audio-only and visual captions visual-only.
2. Audio captions must be fine-grained enough for audio-visual reasoning. Hard constraint: no `speech` item may span more than 10 seconds unless it contains only music/background/silence. If one narrator/character speaks continuously for longer than 10 seconds, split it into consecutive 3-6 second phrase-level chunks with exact original global timestamps.
3. If speech crosses the event boundary, mark only that boundary status.
4. Use the overlaid upper-left timestamp only to improve timestamp accuracy; never mention or transcribe that timestamp overlay in any caption.
5. Use previous event captions only to resolve references, pronouns, recurring people/objects, names, roles, places, quoted phrases, or repeated audio/visual anchors. Do not copy unsupported details into this event.
6. When a person/object/place/sound source has already been identified in previous event captions, reuse that specific identifier instead of vague phrases such as "someone", "the person", "it", or "the object" when the current event evidence supports the same referent.
7. Prefer precision over generalization; when precision is uncertain, give your best approximation and mark it `[unclear]` rather than staying silent. Do not add fields, summaries, hints, scores, labels, or markdown outside the JSON. Caption strings should start directly with observed content.

Return strict JSON only:
{{
  "global_timestamp": "MM:SS-MM:SS",
  "global_caption": "comprehensive event-level omni caption integrating important audio and visual information",
  "audio_captions": [
    {{
      "timestamp": "MM:SS-MM:SS",
      "audio_type": "speech|music|sound|background|silence|unclear",
      "caption": "audio-only caption or faithful speech transcript"
    }}
  ],
  "visual_captions": [
    {{
      "timestamp": "MM:SS-MM:SS",
      "caption": "fine-grained visual-only caption with entities/actions/objects/text/attributes/positions"
    }}
  ]
}}
"""

ADAPTATION_VIDEO_STRUCTURE_PROMPT_TEMPLATE = """You are analyzing one full video as a knowledge base for Adaptation questions.

Adaptation task definition:
The video is always the knowledge base. A later question will present a new text/audio/image case, and the solver must transfer rules, procedures, constraints, formulas, preferences, or demonstrated techniques from the video to solve that new case.

Task:
Watch the whole video once and produce the lightweight structure needed for detailed adaptation-aware omni captioning:
1. a concise whole-video summary
2. a specific video type/category
3. major event segmentation with timestamps
4. a lightweight inventory of explicit solved problems/examples if the video contains questions, exercises, examples, quizzes, code tasks, diagrams, or case studies

Requirements:
1. Segment by semantic event changes, not fixed time windows.
2. Use original global timestamps.
3. `video_type` must be specific, not generic. Include the domain/topic when identifiable, such as "chemistry worked-example lecture", "instrument maintenance tutorial", "coding/debugging screen recording", "recipe technique demonstration", "game strategy explanation", or "math problem-solving lesson".
4. Every event summary must mention important visual evidence and important audio evidence.
5. Each event segment must be at most 3 minutes long. If a semantic event lasts longer than 3 minutes, split it into consecutive sub-segments at natural boundaries near the 3-minute limit.
6. Prefer 3 to {max_events} events, but respect the 3-minute maximum segment length.
7. Do not generate final questions, options, answers, scores, or reasoning tasks in Stage 1.
8. Keep this plan lightweight. Do not extract transferable knowledge items, teaching-flow items, or detailed solution steps here.
9. If the video contains a concrete problem/question/case and then solves or explains it, identify it only as a lightweight candidate. Preserve the original problem statement and final answer when available, but leave detailed steps to the later event captions.

Return strict JSON only:
{{
  "video_summary": "video summary",
  "video_type": "specific video type/category with domain/topic when identifiable",
  "adaptation_domain": "domain/topic of transferable knowledge",
  "knowledge_base_summary": "short summary of the subject matter and worked-example style",
  "solved_problem_candidates": [
    {{
      "problem_id": "prob_01",
      "timestamp": "MM:SS-MM:SS",
      "problem_type": "math|science|coding|diagram|case_study|tool_procedure|quiz|other",
      "problem_statement": "the explicit question/problem/case shown or spoken in the video",
      "question_modality_in_video": "text|image|audio|audiovisual",
      "given_conditions": ["known values, constraints, code inputs, visible diagram facts, or spoken conditions"],
      "final_answer_or_conclusion": "answer/conclusion if the video provides one"
    }}
  ],
  "event_segments": [
    {{
      "event_id": "event_01",
      "start": "MM:SS",
      "end": "MM:SS",
      "summary": "key visual and audio details, including speech/music/sound if present"
    }}
  ]
}}
"""

ADAPTATION_EVENT_OMNI_CAPTION_PROMPT_TEMPLATE = """You are captioning one semantically coherent event clip from a longer video for Adaptation task construction.

Context:
 - Global event range: {event_start_text}-{event_end_text}
 - Previous event captions for reference resolution: {prior_event_caption_context}

Adaptation task definition:
The video is the knowledge base. A later question may be text, audio, or image based, and the solver must apply knowledge learned from this video's omni evidence to a new but analogous case.

Task:
Return the same event-level omni caption structure as the normal caption pipeline. In addition, if this event contains an explicit problem/example/case, add a lightweight `solved_problem_items` index that references the already produced caption items.

Required output:
- `global_timestamp`: full event range.
- `global_caption`: a rich event-level omni caption covering all important audio and visual facts in the event, with special care for teaching content such as visible problem text, formulas, diagrams, code, examples, spoken explanation, intermediate values, and conclusions.
- `audio_captions`: time-ordered audio-only items, split when speech/speaker/music/sound/silence/unclear audio changes.
- `visual_captions`: time-ordered visual-only items, split when shot/scene/subject/action/object/text/layout/state changes.
- `solved_problem_items`: lightweight indexes for explicit problems/examples/cases. Do not rewrite the solution process; reference relevant caption items by index and timestamp.

Visual detail requirements (every visual caption must be fine-grained and specific):
1. Subjects/entities: exact count (never "several"/"some"); for each person give gender, approximate age, build, hair, clothing colors/style, accessories, and position in frame.
2. Colors: name the precise shade of key items (reddish-orange, olive, teal, fuchsia, navy, maroon) — never the base color alone.
3. Objects: exact color/shade, shape/geometry, material/finish, and sub-type/brand/model when identifiable; use domain-specific names ("fedora" not "hat").
4. Actions: body-part specifics (which hand/foot/side and direction), temporal phases, action-object interaction (grip, contact, tool, object state before -> after), speed, multi-person interaction, communicative gestures, and exact repetition counts.
5. Spatial layout: left/right, foreground/midground/background, facing directions, depth relations.
6. Camera/editing: shot size, camera movement, transition type (hard cut/jump cut/dissolve/fade/wipe).
7. On-screen text, formulas, code, diagrams, labels, tables, UI, and numbers: transcribe verbatim and exactly (LaTeX for math; enumerate every labeled vertex/side/angle/length/coordinate for geometry figures), with position.
8. State changes: objects/scenes that transform, appear, or disappear.

Audio detail requirements (every audio caption must be fine-grained and specific):
1. If the sound source can be identified (person/object/place), start with a specific referring description of that source.
2. Speech: transcribe only words audible in that window, verbatim in the original language; give speaker identity, language/accent, and tone/voice character when perceivable. Mark unclear words as `[unclear]`.
3. Music: never just "background music" — give genre, tempo/mood, and name distinguishable instruments; sound effects: name each with its texture; report silence intervals explicitly.
4. Do not hallucinate audio: do not assume music from visible instruments or speech from mouth movement; if you cannot clearly hear it, mark it `[unclear]`.

Rules:
1. Keep audio captions audio-only and visual captions visual-only.
2. Audio captions must be fine-grained enough for reasoning. No `speech` item may span more than 10 seconds unless it contains only music/background/silence; split long continuous speech into consecutive 3-6 second phrase-level chunks.
3. Use the overlaid upper-left timestamp only to improve timestamp accuracy; never mention or transcribe that timestamp overlay.
4. Do not output `transferable_knowledge_items` or `teaching_flow_items`.
5. Extract lightweight solved-problem indexes only when the video contains any of these:
   - a displayed or spoken question, exercise, quiz, exam problem, case study, coding task, diagram question, formula substitution, table/graph problem, or troubleshooting case
   - a worked example where the narrator/teacher solves step by step
   - a before/after procedure where a specific input case leads to a concrete output or conclusion
6. For each solved problem, preserve the original problem statement as faithfully as possible. If it is visual, transcribe visible text/formulas/code and describe the relevant diagram/image. If it is spoken, transcribe the spoken problem setup. If both are needed, mark `question_modality_in_video="audiovisual"`.
7. Do not rewrite solution steps inside `solved_problem_items`. Instead, fill `problem_caption_refs`, `solution_caption_refs`, and `answer_caption_refs` with references to the `audio_captions` or `visual_captions` items you already output in this same JSON.
8. Do not generate new final questions, options, scores, or markdown outside the JSON. You may extract answers only when the video itself provides or derives them. Prefer precision over generalization; caption strings should start directly with observed content.

Return strict JSON only:
{{
  "global_timestamp": "MM:SS-MM:SS",
  "global_caption": "comprehensive event-level omni caption integrating important audio and visual information",
  "audio_captions": [
    {{
      "timestamp": "MM:SS-MM:SS",
      "audio_type": "speech|music|sound|background|silence|unclear",
      "caption": "audio-only caption or faithful speech transcript"
    }}
  ],
  "visual_captions": [
    {{
      "timestamp": "MM:SS-MM:SS",
      "caption": "fine-grained visual-only caption with entities/actions/objects/text/formulas/code/attributes/positions"
    }}
  ],
  "solved_problem_items": [
    {{
      "problem_id": "event_01_prob_01",
      "timestamp": "MM:SS-MM:SS",
      "problem_type": "math|science|coding|diagram|case_study|tool_procedure|quiz|other",
      "problem_statement": "faithful extracted problem/question/case statement",
      "question_modality_in_video": "text|image|audio|audiovisual",
      "visual_problem_text": "visible problem text/formula/code/table/diagram details, if any",
      "audio_problem_text": "spoken problem setup, if any",
      "given_conditions": ["known values, constraints, code inputs, visible diagram facts, or spoken conditions"],
      "problem_caption_refs": [
        {{
          "ref_type": "audio|visual",
          "index": 0,
          "timestamp": "MM:SS-MM:SS",
          "reason": "problem_statement|given_condition|diagram|code|table|setup"
        }}
      ],
      "solution_caption_refs": [
        {{
          "ref_type": "audio|visual",
          "index": 0,
          "timestamp": "MM:SS-MM:SS",
          "reason": "solution_step|formula_substitution|diagram_operation|code_execution|verification"
        }}
      ],
      "answer_caption_refs": [
        {{
          "ref_type": "audio|visual",
          "index": 0,
          "timestamp": "MM:SS-MM:SS",
          "reason": "final_answer|conclusion"
        }}
      ],
      "final_answer_or_conclusion": "answer/conclusion if provided",
      "transfer_potential": "short note on what kind of analogous new case this example could support"
    }}
  ]
}}
"""

@dataclasses.dataclass
class VideoItem:
    video_id: str
    raw_video_path: Path
    time_video_path: Path
    time_video_oss_path: str
    source_relative_path: str

@dataclasses.dataclass
class SegmentSpec:
    event_id: str
    start_sec: float
    end_sec: float
    summary: str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniQA timestamped audio-visual caption generation.")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory or one mp4 file to read raw videos from.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-glob", action="append", default=None, help="Glob under data-path; default: **/*.mp4")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    parser.add_argument("--api-host", default=os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com"), help="Gemini-compatible API base. Gemini video captioning goes through {base}/v1beta/models/{model}:generateContent.")
    parser.add_argument("--gemini-model", default="gemini-3.1-pro-preview", help="Gemini model for caption stage. A leading vertex_ai./vertex. prefix is stripped before building the Vertex URL.")
    parser.add_argument("--gemini-thinking-level", choices=["default", "low", "medium", "high"], default="high", help="Gemini 3 thinking effort for the fine-grained EVENT CAPTION step (generationConfig.thinkingConfig.thinkingLevel). 'default' omits the key so the model uses its own default. 3.1-pro uses high.")
    parser.add_argument("--gemini-media-resolution", choices=["default", "low", "medium", "high"], default="high", help="Per-frame media resolution for the EVENT CAPTION step (generationConfig.mediaResolution). 'default' omits the key (model default, ~70 tokens/frame for video). For VIDEO on gemini-3.1-pro: low/medium are identical (~70 tokens/frame); high gives ~266 tokens/frame (the ~256 target).")
    parser.add_argument("--gemini-plan-thinking-level", choices=["default", "low", "medium", "high"], default="low", help="Gemini 3 thinking effort for the PLAN step (whole-video structure/segmentation). Defaults to low: the plan only needs coarse segmentation, so it is kept cheap. 'default' omits the key. Only used when --plan-segmenter gemini.")
    parser.add_argument("--gemini-plan-media-resolution", choices=["default", "low", "medium", "high"], default="low", help="Per-frame media resolution for the PLAN step. Defaults to low: coarse whole-video structure does not need high single-frame detail. 'default' omits the key. Only used when --plan-segmenter gemini.")
    parser.add_argument("--gemini-max-output-tokens", type=int, default=32768, help="maxOutputTokens for Gemini generateContent. Give headroom because high thinking consumes output-token budget.")
    parser.add_argument("--gemini-temperature", type=float, default=0.0, help="Sampling temperature for Gemini generateContent.")
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--openai-api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--openai-model", default="openai.gpt-5.4-2026-03-05-response")
    parser.add_argument("--oss-prefix", default="")
    parser.add_argument("--oss-http-endpoint", default="")
    parser.add_argument("--oss-sign-url", action="store_true", default=False, help="Always use ossutil signed URLs for Gemini. Auto mode also signs when the plain HTTPS URL is not public.")
    parser.add_argument("--no-oss-sign-url", action="store_false", dest="oss_sign_url", help="Use plain HTTPS URL instead of signed URL.")
    parser.add_argument("--oss-public-read", action="store_true", default=False, help="Set uploaded object ACL to public-read for Gemini URL probing. Disabled by default because some buckets reject object ACL changes.")
    parser.add_argument("--no-oss-public-read", action="store_false", dest="oss_public_read", help="Do not change uploaded object ACL.")
    parser.add_argument("--oss-sign-timeout", type=int, default=86400, help="Signed OSS URL TTL in seconds for Gemini reads.")
    parser.add_argument("--ossutil-bin", default="ossutil")
    parser.add_argument("--ossutil-config", type=Path, default=Path(".ossutilconfig"))
    parser.add_argument("--oss-parallel", type=int, default=8, help="ossutil --parallel for single-file multipart upload.")
    parser.add_argument("--oss-part-size", default="16777216", help="ossutil --part-size in bytes for multipart upload.")
    parser.add_argument("--oss-checkpoint-dir", type=Path, default=None, help="Enable resumable upload checkpoint directory.")
    parser.add_argument("--upload-timeout", type=int, default=300, help="Timeout seconds for each ossutil upload; 0 disables timeout.")
    parser.add_argument("--upload-workers", type=int, default=1, help="Maximum concurrent ossutil uploads.")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--use-raw-video", action="store_true", help="Use the original video for Gemini instead of generating a timestamp overlay video.")
    parser.add_argument("--force-time-video", action="store_true")
    parser.add_argument("--time-video-width", type=int, default=640, help="Scale timestamp video width to keep Gemini inline fallback small; 0 keeps original width.")
    parser.add_argument("--time-video-fps", type=float, default=2.0, help="FPS for timestamp overlay videos.")
    parser.add_argument("--time-video-crf", default="30", help="CRF for timestamp video encoding.")
    parser.add_argument("--time-video-fontfile", type=Path, default=None, help="Explicit TTF/OTF/TTC font for ffmpeg drawtext timestamp overlay. Also supports TIME_VIDEO_FONTFILE env var.")
    parser.add_argument("--gemini-video-source", choices=["auto", "url", "inline"], default="auto", help="auto uses URL when available; it only inlines very small local videos because the chat API has a small JSON request-body limit.")
    parser.add_argument("--max-inline-video-mb", type=float, default=3.5, help="Maximum local video size allowed for inline Gemini fallback. Keep below the API JSON body limit because base64 expands video bytes.")
    parser.add_argument("--force-upload", action="store_true")
    parser.add_argument("--num-candidate-videos", type=int, default=0, help="0 means all matched videos.")
    parser.add_argument("--max-events", type=int, default=10)
    parser.add_argument("--plan-segmenter", choices=["gemini", "scenedetect"], default="gemini", help="How to split a video into event segments for the plan step. 'gemini' calls Gemini to produce a semantic structure (default); 'scenedetect' runs PySceneDetect locally on the video (no plan API call) and derives segments from detected scene cuts.")
    parser.add_argument("--scenedetect-detector", choices=["content", "adaptive", "threshold"], default="content", help="PySceneDetect detector when --plan-segmenter scenedetect: 'content' (HSV content change, best general default), 'adaptive' (rolling-average content, robust to fast motion), 'threshold' (fade to/from black, luma-based).")
    parser.add_argument("--scenedetect-threshold", type=float, default=27.0, help="PySceneDetect cut threshold. For 'content'/'adaptive' this is the content/adaptive threshold (~27 default; lower = more cuts). For 'threshold' it is the 0-255 luma fade threshold (try ~12).")
    parser.add_argument("--scenedetect-min-scene-len-sec", type=float, default=2.0, help="Minimum scene length in seconds for PySceneDetect (converted to frames via the video fps); shorter candidate scenes are merged away.")
    parser.add_argument("--min-video-duration-sec", type=float, default=1.0)
    parser.add_argument("--max-video-duration-sec", type=float, default=0.0, help="0 disables max duration filter.")
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--stage1-caption-profile", choices=["default", "adaptation"], default="default", help="Caption prompt profile. adaptation extracts video-as-knowledge-base transfer rules for Adaptation tasks.")
    parser.add_argument("--api-qps", type=float, default=2.0, help="Default maximum API request starts per second across all workers.")
    parser.add_argument("--max-inflight-queries", type=int, default=64, help="Default maximum concurrent API requests across all workers.")
    parser.add_argument("--stage1-api-qps", type=float, default=None, help="Caption stage API QPS; defaults to --api-qps.")
    parser.add_argument("--stage1-max-inflight-queries", type=int, default=None, help="Caption stage max inflight API requests; defaults to --max-inflight-queries.")
    parser.add_argument("--video-workers", type=int, default=64, help="Maximum concurrently processed videos.")
    parser.add_argument("--min-api-interval-sec", type=float, default=0.0, help="Deprecated; use --api-qps instead.")
    parser.add_argument("--api-max-retries", type=int, default=10)
    parser.add_argument("--api-retry-backoff-sec", type=float, default=3.0)
    parser.add_argument("--api-retry-max-backoff-sec", type=float, default=120.0, help="Maximum exponential backoff sleep for transient API retries.")
    parser.add_argument("--json-parse-max-retries", type=int, default=2, help="JSON repair attempts after one successful API response; avoids repeating expensive video calls.")
    parser.add_argument("--json-parse-retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true", default=True, help="Reuse existing stage outputs and do not call models again when cache exists.")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Disable stage cache reuse.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

_JSONL_WRITE_LOCK = threading.Lock()

_UPLOAD_SEMAPHORE: threading.BoundedSemaphore | None = None

_UPLOAD_LIMIT: int | None = None

def acquire_upload_slot(max_uploads: int) -> threading.BoundedSemaphore | None:
    global _UPLOAD_SEMAPHORE, _UPLOAD_LIMIT
    if max_uploads <= 0:
        return None
    with _JSONL_WRITE_LOCK:
        if _UPLOAD_SEMAPHORE is None or _UPLOAD_LIMIT != max_uploads:
            _UPLOAD_LIMIT = max_uploads
            _UPLOAD_SEMAPHORE = threading.BoundedSemaphore(max_uploads)
        semaphore = _UPLOAD_SEMAPHORE
    semaphore.acquire()
    return semaphore

def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def cached_json(path: Path) -> Any | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        return read_json(path)
    except Exception:
        return None

def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()

AUDIO_TYPES = {"speech", "music", "sound", "background", "silence", "unclear"}

AUDIO_TYPE_ALIASES = {
    "sound effect": "sound",
    "sound effects": "sound",
    "sfx": "sound",
    "background sound": "background",
    "background sounds": "background",
    "background noise": "background",
    "no audio": "silence",
    "silent": "silence",
    "none": "silence",
    "muffled": "unclear",
    "unclear audio": "unclear",
}

def normalize_audio_type(value: Any) -> str:
    text = normalize_text(value).lower()
    text = AUDIO_TYPE_ALIASES.get(text, text)
    return text if text in AUDIO_TYPES else ""

def infer_audio_type(audio_caption: str) -> str:
    text = audio_caption.lower()
    if not text:
        return "unclear"
    if any(token in text for token in ("no speech", "no salient sound", "silence", "silent")):
        return "silence"
    if any(token in text for token in ("unclear", "muffled", "inaudible")):
        return "unclear"
    if any(token in text for token in ("music", "song", "singing", "instrument", "guitar", "piano", "drum", "melody")):
        return "music"
    if any(token in text for token in ("says", "said", "speaks", "speech", "talks", "asks", "replies", "announces", '"')):
        return "speech"
    if any(token in text for token in ("background", "ambient", "noise", "crowd", "wind", "traffic")):
        return "background"
    return "sound"

def clean_caption_text(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    prefix_patterns = [
        r"^(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)?)\s+(?:hint|caption|clue)\s*[:：\-–—,;]?\s*",
        r"^(?:hint|caption|clue)\s*(?:#?\d+|[A-Z])?\s*[:：\-–—,;]?\s*",
        r"^\(?\d+\)?[\.)、]\s*(?:hint|caption|clue)\s*[:：\-–—,;]?\s*",
    ]
    previous = None
    while previous != text:
        previous = text
        for pattern in prefix_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text

def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:160] or hashlib.md5(value.encode("utf-8")).hexdigest()[:16]

def mmss(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds or 0))))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"

def parse_mmss(value: str) -> float | None:
    raw = normalize_text(value)
    raw = re.sub(r"[^0-9:.]", "", raw)
    if not raw:
        return None
    parts = [p for p in raw.split(":") if p != ""]
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return float(int(parts[0]) * 60 + float(parts[1]))
        if len(parts) == 3:
            return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
    except ValueError:
        return None
    return None

def parse_timestamp_range(value: str) -> tuple[float, float] | None:
    timestamp_pattern = r"[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?(?:\.[0-9]+)?"
    match = re.search(rf"({timestamp_pattern})\s*-\s*({timestamp_pattern})", value or "")
    if not match:
        one = re.search(rf"({timestamp_pattern})", value or "")
        if not one:
            return None
        start = parse_mmss(one.group(1))
        if start is None:
            return None
        return start, start + 1.0
    start = parse_mmss(match.group(1))
    end = parse_mmss(match.group(2))
    if start is None or end is None:
        return None
    return start, max(start + 0.5, end)

def shift_range(timestamp: str, offset_sec: float) -> str:
    parsed = parse_timestamp_range(timestamp)
    if not parsed:
        return normalize_text(timestamp)
    start, end = parsed
    return f"{mmss(start + offset_sec)}-{mmss(end + offset_sec)}"


def add_caption_indices(items: list[dict[str, Any]], index_key: str) -> list[dict[str, Any]]:
    indexed_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        indexed_item = dict(item)
        indexed_item[index_key] = index
        indexed_items.append(indexed_item)
    return indexed_items

def probe_video(video_path: Path) -> dict[str, Any]:
    result = run([
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video_path)
    ], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr}")
    return json.loads(result.stdout or "{}")

def safe_probe_video(video_path: Path) -> dict[str, Any] | None:
    try:
        return probe_video(video_path)
    except Exception as exc:
        tqdm.write(f"[warn] skip unreadable video {video_path}: {exc}", file=sys.stderr)
        return None

def duration_from_probe(probe: dict[str, Any]) -> float:
    try:
        return float((probe.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0

def find_videos(data_path: Path, globs: list[str] | None) -> list[Path]:
    video_suffixes = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
    if data_path.is_file():
        return [data_path] if data_path.suffix.lower() in video_suffixes else []
    patterns = globs or ["**/*.mp4", "**/*.webm", "**/*.mov", "**/*.mkv", "**/*.avi", "**/*.m4v"]
    videos: list[Path] = []
    for pattern in tqdm(patterns, desc="Loading video sources", unit="glob"):
        videos.extend(path for path in data_path.glob(pattern) if path.is_file() and path.suffix.lower() in video_suffixes)
    return sorted(set(videos), key=lambda p: str(p))

def build_oss_path(oss_prefix: str, relative_path: str, time_name: str) -> str:
    prefix = oss_prefix.rstrip("/")
    parent = str(Path(relative_path).parent).replace(".", "").strip("/")
    if parent:
        return f"{prefix}/{parent}/{time_name}"
    return f"{prefix}/{time_name}"

def oss_path_to_https(oss_path: str, http_endpoint: str) -> str:
    match = re.match(r"^oss://([^/]+)/(.*)$", oss_path)
    if not match:
        return oss_path
    endpoint = http_endpoint.rstrip("/")
    bucket = match.group(1)
    object_key = match.group(2).lstrip("/")
    if bucket not in endpoint and os.environ.get("OSS_HTTP_ENDPOINT_TEMPLATE"):
        endpoint = os.environ["OSS_HTTP_ENDPOINT_TEMPLATE"].format(bucket=bucket).rstrip("/")
    return f"{endpoint}/{object_key}"

def sign_oss_url(oss_path: str, args: argparse.Namespace) -> str:
    cmd = [
        args.ossutil_bin,
        "sign",
        oss_path,
        "--timeout",
        str(max(60, int(args.oss_sign_timeout or 60))),
        "-c",
        str(args.ossutil_config),
    ]
    result = run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ossutil sign failed: {' '.join(cmd)}\n{result.stderr or result.stdout}")
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            return line
    raise RuntimeError(f"ossutil sign did not return a URL: {' '.join(cmd)}\n{result.stdout}")

def public_https_head_ok(url: str) -> bool:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10)
        return 200 <= resp.status_code < 400
    except requests.RequestException:
        return False

def choose_gemini_video_ref(oss_path: str, time_video_path: Path, args: argparse.Namespace) -> str | Path:
    size_mb = time_video_path.stat().st_size / 1024 / 1024
    if args.gemini_video_source == "inline":
        if size_mb > float(args.max_inline_video_mb):
            raise RuntimeError(
                f"Inline Gemini video is too large: {size_mb:.1f} MB > {args.max_inline_video_mb} MB. "
                "Use URL mode with a signed/public OSS URL."
            )
        return time_video_path

    plain_url = oss_path_to_https(oss_path, args.oss_http_endpoint)
    if not args.oss_sign_url and public_https_head_ok(plain_url):
        return plain_url

    try:
        signed_url = sign_oss_url(oss_path, args)
        if args.oss_sign_url or args.gemini_video_source == "url":
            tqdm.write(f"Using signed OSS URL for Gemini: {oss_path}", file=sys.stderr)
            return signed_url
        if public_https_head_ok(signed_url):
            tqdm.write(f"Plain OSS URL is not public; using signed OSS URL for Gemini: {oss_path}", file=sys.stderr)
            return signed_url
        tqdm.write(f"Signed OSS URL HEAD check failed; using signed URL anyway: {oss_path}", file=sys.stderr)
        return signed_url
    except Exception as exc:
        if args.gemini_video_source == "url":
            raise RuntimeError(f"OSS URL is not public and signing failed for {oss_path}: {type(exc).__name__}: {exc}") from exc
        tqdm.write(f"OSS URL is not public and signing failed ({type(exc).__name__}: {exc}); considering inline fallback", file=sys.stderr)

    if size_mb > float(args.max_inline_video_mb):
        raise RuntimeError(
            f"Gemini cannot access OSS URL and inline fallback is too large: {size_mb:.1f} MB > {args.max_inline_video_mb} MB. "
            "Use a readable OSS URL or fix ossutil signing credentials."
        )
    tqdm.write(f"Gemini URL is not accessible; using inline local time video ({size_mb:.1f} MB): {time_video_path}", file=sys.stderr)
    return time_video_path

def error_dir_for_video(args: argparse.Namespace, video: Path, data_root: Path) -> Path:
    relative = str(video.relative_to(data_root)) if data_root.is_dir() and video.is_relative_to(data_root) else video.name
    return args.output_dir / "raw" / safe_slug(Path(relative).with_suffix("").as_posix())

def ffmpeg_video_encode_args(crf: str) -> list[str]:
    result = run(["ffmpeg", "-hide_banner", "-encoders"], check=False)
    encoders = result.stdout + result.stderr
    if " libx264 " in encoders:
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", crf, "-pix_fmt", "yuv420p"]
    if " libopenh264 " in encoders:
        return ["-c:v", "libopenh264", "-b:v", "1500k", "-pix_fmt", "yuv420p"]
    if " mpeg4 " in encoders:
        return ["-c:v", "mpeg4", "-q:v", "5", "-pix_fmt", "yuv420p"]
    return ["-c:v", "h264", "-pix_fmt", "yuv420p"]

def escape_ffmpeg_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

@functools.cache
def discover_time_video_fontfile() -> Path | None:
    env_font = os.environ.get("TIME_VIDEO_FONTFILE", "").strip()
    if env_font and Path(env_font).is_file():
        return Path(env_font)

    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path

    roots = [
        Path.home() / ".cache" / "uv",
        Path("/root/.cache/uv"),
        Path("/opt/conda"),
        Path("/usr/local/lib"),
        Path("/usr/lib"),
    ]
    patterns = [
        "**/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
        "**/DejaVuSans.ttf",
        "**/LiberationSans-Regular.ttf",
        "**/NotoSans-Regular.ttf",
        "**/NotoSansCJK-Regular.ttc",
    ]
    seen_roots: set[Path] = set()
    for root in roots:
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        for pattern in patterns:
            try:
                for path in root.glob(pattern):
                    if path.is_file():
                        return path
            except OSError:
                continue
    return None

def resolve_time_video_fontfile(fontfile: Path | None = None) -> Path | None:
    if fontfile is not None:
        if fontfile.is_file():
            return fontfile
        raise FileNotFoundError(f"Timestamp overlay fontfile does not exist: {fontfile}")
    return discover_time_video_fontfile()

def build_timestamp_drawtext_filter(fontfile: Path | None) -> str:
    drawtext = "drawtext=text='%{pts\\:hms}':x=12:y=12:fontcolor=white:fontsize=28:box=1:boxcolor=black@0.65:boxborderw=6"
    if fontfile is not None:
        drawtext += f":fontfile={escape_ffmpeg_filter_value(str(fontfile))}"
    return drawtext

def make_time_video(
    raw_video_path: Path,
    output_path: Path,
    force: bool = False,
    width: int = 640,
    fps: float = 2.0,
    crf: str = "30",
    fontfile: Path | None = None,
) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        return output_path
    ensure_dir(output_path.parent)
    timestamp_fps = max(float(fps or 0), 0.1)
    filters = [f"fps={timestamp_fps:g}"]
    if int(width or 0) > 0:
        filters.append(f"scale={int(width)}:-2")
    resolved_fontfile = resolve_time_video_fontfile(fontfile)
    drawtext = build_timestamp_drawtext_filter(resolved_fontfile)
    filters.append(drawtext)
    vf = ",".join(filters)
    cmd = [
        "ffmpeg", "-y", "-i", str(raw_video_path), "-vf", vf, "-map", "0:v:0", "-map", "0:a:0?",
    ] + ffmpeg_video_encode_args(str(crf)) + ["-c:a", "aac", "-b:a", "48k", "-ac", "1", "-movflags", "+faststart", str(output_path)]
    result = run(cmd, check=False)
    if result.returncode != 0:
        font_hint = (
            f"Resolved timestamp fontfile: {resolved_fontfile}"
            if resolved_fontfile is not None
            else "No timestamp fontfile was found. Set --time-video-fontfile or TIME_VIDEO_FONTFILE to a .ttf/.otf/.ttc file."
        )
        raise RuntimeError(f"ffmpeg timestamp overlay failed for {raw_video_path}.\n{font_hint}\n{result.stderr}")
    return output_path

def set_oss_object_acl(oss_path: str, acl: str, args: argparse.Namespace) -> None:
    cmd = [args.ossutil_bin, "set-acl", oss_path, acl, "-c", str(args.ossutil_config)]
    result = run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ossutil set-acl failed: {' '.join(cmd)}\n{result.stderr or result.stdout}")

def upload_to_oss(local_path: Path, oss_path: str, args: argparse.Namespace, force: bool = False, skip: bool = False) -> str:
    if skip:
        return oss_path
    checkpoint_dir = args.oss_checkpoint_dir or (local_path.parent / ".ossutil_checkpoint")
    ensure_dir(checkpoint_dir)
    cmd = [
        args.ossutil_bin,
        "cp",
        str(local_path),
        oss_path,
        "-c",
        str(args.ossutil_config),
        "-f",
        "--parallel",
        str(max(1, int(args.oss_parallel or 1))),
        "--part-size",
        str(args.oss_part_size),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if force:
        pass
    semaphore = acquire_upload_slot(args.upload_workers)
    try:
        size_mb = local_path.stat().st_size / 1024 / 1024
        tqdm.write(f"UPLOAD start {size_mb:.1f} MB: {' '.join(cmd)}", file=sys.stderr)
        try:
            result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=args.upload_timeout if args.upload_timeout > 0 else None)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ossutil upload timed out after {args.upload_timeout}s: {' '.join(cmd)}") from exc
        if result.returncode != 0:
            raise RuntimeError(f"ossutil upload failed: {' '.join(cmd)}\n{result.stderr or result.stdout}")
        tqdm.write(f"UPLOAD done: {oss_path}", file=sys.stderr)
        return oss_path
    finally:
        if semaphore is not None:
            semaphore.release()

_API_RATE_LOCK = threading.Lock()

_NEXT_API_START_BY_STAGE: dict[str, float] = {}

_API_INFLIGHT_SEMAPHORES_BY_STAGE: dict[str, tuple[int, threading.BoundedSemaphore]] = {}

_STAGE_LOCAL = threading.local()

@contextlib.contextmanager
def stage_context(stage_name: str) -> Iterable[None]:
    previous = getattr(_STAGE_LOCAL, "stage_name", None)
    _STAGE_LOCAL.stage_name = stage_name
    try:
        yield
    finally:
        _STAGE_LOCAL.stage_name = previous

def current_api_limits(args: argparse.Namespace) -> tuple[float, int]:
    stage_name = getattr(_STAGE_LOCAL, "stage_name", "")
    if stage_name == "caption":
        qps = args.stage1_api_qps if args.stage1_api_qps is not None else args.api_qps
        max_inflight = args.stage1_max_inflight_queries if args.stage1_max_inflight_queries is not None else args.max_inflight_queries
    else:
        qps = args.api_qps
        max_inflight = args.max_inflight_queries
    return float(qps), int(max_inflight)

def throttle_api(qps: float, legacy_min_interval_sec: float) -> None:
    interval = 1.0 / qps if qps > 0 else max(0.0, legacy_min_interval_sec)
    if interval <= 0:
        return

    stage_name = getattr(_STAGE_LOCAL, "stage_name", "") or "default"
    with _API_RATE_LOCK:
        next_start = _NEXT_API_START_BY_STAGE.get(stage_name, 0.0)
        now = time.monotonic()
        wait = max(0.0, next_start - now)
        _NEXT_API_START_BY_STAGE[stage_name] = max(now, next_start) + interval
    if wait > 0:
        time.sleep(wait)


    rate_limit_file = normalize_text(os.environ.get("OMNIQA_GLOBAL_API_RATE_LIMIT_FILE"))
    if not rate_limit_file:
        return
    lock_path = Path(rate_limit_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            lock_file.seek(0)
            raw = lock_file.read().strip()
            try:
                next_start = float(raw) if raw else 0.0
            except ValueError:
                next_start = 0.0
            now = time.monotonic()
            wait = max(0.0, next_start - now)
            reserved_start = max(now, next_start)
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(f"{reserved_start + interval:.9f}")
            lock_file.flush()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    if wait > 0:
        time.sleep(wait)

def acquire_api_slot(max_inflight: int) -> threading.BoundedSemaphore | None:
    if max_inflight <= 0:
        return None
    stage_name = getattr(_STAGE_LOCAL, "stage_name", "") or "default"
    with _API_RATE_LOCK:
        current = _API_INFLIGHT_SEMAPHORES_BY_STAGE.get(stage_name)
        if current is None or current[0] != max_inflight:
            current = (max_inflight, threading.BoundedSemaphore(max_inflight))
            _API_INFLIGHT_SEMAPHORES_BY_STAGE[stage_name] = current
        semaphore = current[1]
    semaphore.acquire()
    return semaphore

def _request_with_retries(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int, qps: float, max_inflight: int, min_interval_sec: float, max_retries: int, backoff: float, max_backoff: float = 120.0) -> dict[str, Any]:
    last_error = ""
    for attempt in range(max(1, max_retries)):
        throttle_api(qps, min_interval_sec)
        semaphore = acquire_api_slot(max_inflight)
        try:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code < 400:
                    return resp.json()
                last_error = f"HTTP {resp.status_code}: {resp.text[:1000]}"
                if resp.status_code not in {408, 409, 418, 425, 429, 500, 502, 503, 504}:
                    break
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        finally:
            if semaphore is not None:
                semaphore.release()
        if attempt >= max(1, max_retries) - 1:
            break
        sleep_cap = max(0.0, float(max_backoff or 0.0))
        sleep_sec = min(sleep_cap, float(backoff or 0.0) * (2 ** attempt))
        if sleep_sec > 0:
            sleep_sec = random.uniform(0.5 * sleep_sec, 1.5 * sleep_sec)
        tqdm.write(f"API retry {attempt + 1}/{max(1, max_retries)} after error: {last_error[:300]}; sleep={sleep_sec:.1f}s", file=sys.stderr)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    raise RuntimeError(last_error)

def call_chat(api_base: str, api_key: str, payload: dict[str, Any], timeout: int, qps: float, max_inflight: int, min_interval_sec: float, max_retries: int, backoff: float, max_backoff: float = 120.0) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    return _request_with_retries(url, headers, payload, timeout, qps, max_inflight, min_interval_sec, max_retries, backoff, max_backoff)

def call_vertex_generate_content(api_base: str, api_key: str, model: str, payload: dict[str, Any], timeout: int, qps: float, max_inflight: int, min_interval_sec: float, max_retries: int, backoff: float, max_backoff: float = 120.0) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": os.environ.get("GEMINI_AUTH_PREFIX", "Bearer ") + api_key}
    return _request_with_retries(url, headers, payload, timeout, qps, max_inflight, min_interval_sec, max_retries, backoff, max_backoff)

_VERTEX_MODEL_PREFIX_RE = re.compile(r"^(vertex_ai|vertex)\.")
_MEDIA_RESOLUTION_MAP = {"low": "MEDIA_RESOLUTION_LOW", "medium": "MEDIA_RESOLUTION_MEDIUM", "high": "MEDIA_RESOLUTION_HIGH"}
_THINKING_LEVEL_MAP = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}

def vertex_model_name(model: str) -> str:
    """Strip a gateway routing prefix so the name is a bare Vertex model id.

    The URL path is ``.../models/{model}:generateContent`` and the model id
    itself contains dots (``gemini-3.1-pro-preview``), so only the leading
    ``vertex_ai.`` / ``vertex.`` router prefix is removed.
    """
    return _VERTEX_MODEL_PREFIX_RE.sub("", str(model or "").strip())

def call_gemini_video(args: argparse.Namespace, prompt: str, video_ref: str | Path, *, max_output_tokens: int | None = None, media_resolution: str | None = None, thinking_level: str | None = None) -> dict[str, Any]:
    source = str(video_ref)
    if re.match(r"^(https?|oss)://", source):
        video_part: dict[str, Any] = {"fileData": {"mimeType": "video/mp4", "fileUri": source}}
    else:
        data = base64.b64encode(Path(source).read_bytes()).decode("ascii")
        video_part = {"inlineData": {"mimeType": "video/mp4", "data": data}}
    fps = getattr(args, "time_video_fps", None)
    if fps:
        video_part["videoMetadata"] = {"fps": float(fps)}


    generation_config: dict[str, Any] = {
        "temperature": float(getattr(args, "gemini_temperature", 0.0) or 0.0),
        "maxOutputTokens": int(max_output_tokens or getattr(args, "gemini_max_output_tokens", 32768)),
    }
    resolution_key = media_resolution if media_resolution is not None else getattr(args, "gemini_media_resolution", "")
    media_resolution_value = _MEDIA_RESOLUTION_MAP.get((resolution_key or "").lower())
    if media_resolution_value:
        generation_config["mediaResolution"] = media_resolution_value
    thinking_key = thinking_level if thinking_level is not None else getattr(args, "gemini_thinking_level", "")
    thinking_level_value = _THINKING_LEVEL_MAP.get((thinking_key or "").lower())
    if thinking_level_value:
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level_value}

    payload = {
        "contents": [{"role": "user", "parts": [video_part, {"text": prompt}]}],
        "generationConfig": generation_config,
    }
    model = vertex_model_name(args.gemini_model)
    qps, max_inflight = current_api_limits(args)
    return call_vertex_generate_content(args.api_host, args.api_key, model, payload, args.request_timeout, qps, max_inflight, args.min_api_interval_sec, min(args.api_max_retries, 10), args.api_retry_backoff_sec, args.api_retry_max_backoff_sec)

def call_openai_text(args: argparse.Namespace, prompt: str, *, model: str | None = None, max_output_tokens: int = 4096) -> dict[str, Any]:
    del max_output_tokens
    payload = {
        "model": model or args.openai_model,
        "input": prompt,
    }
    qps, max_inflight = current_api_limits(args)
    return call_chat(args.openai_api_base, args.openai_api_key, payload, args.request_timeout, qps, max_inflight, args.min_api_interval_sec, min(args.api_max_retries, 10), args.api_retry_backoff_sec, args.api_retry_max_backoff_sec)

def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if isinstance(content.get("parts"), list):
            return extract_text_from_content(content.get("parts"))
        text = content.get("text", "")
        if isinstance(text, dict):
            text = text.get("value", "")
        return text.strip() if isinstance(text, str) else ""
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        text = extract_text_from_content(item)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()

def response_text(resp: dict[str, Any]) -> str:
    output_text = resp.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for item in resp.get("output") or []:
        text = extract_text_from_content((item or {}).get("content"))
        if text:
            chunks.append(text)
    if chunks:
        return "".join(chunks).strip()

    choices = resp.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        text = extract_text_from_content(message.get("content", ""))
        if text:
            return text

    candidates = resp.get("candidates") or []
    if candidates:
        parts = (candidates[0].get("content") or {}).get("parts") or []

        answer_parts = [p for p in parts if not (isinstance(p, dict) and p.get("thought"))]
        text = extract_text_from_content(answer_parts or parts)
        if text:
            return text

    return json.dumps(resp, ensure_ascii=False)

def extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.replace("\u0000", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([i for i in [text.find("{"), text.find("[")] if i >= 0], default=-1)
        if start < 0:
            raise
        stack = []
        in_str = False
        esc = False
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

def repair_json_text(args: argparse.Namespace, broken_text: str, context: str) -> Any:
    prompt = f"""Repair the following model output into strict valid JSON only.
Do not add explanations, markdown, or fields that are not already implied.
Context: {context}

Broken output:
{broken_text[:12000]}
"""
    resp = call_openai_text(args, prompt, model=args.openai_model, max_output_tokens=4096)
    return extract_json(response_text(resp))

def call_json(request_fn: Any, context: str, args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    resp = request_fn()
    text = response_text(resp)
    try:
        return resp, extract_json(text)
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}; text={text[:500]}"
    repair_attempts = max(0, int(args.json_parse_max_retries or 0))
    for attempt in range(repair_attempts):
        sleep_sec = args.json_parse_retry_backoff_sec * attempt
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        try:
            data = repair_json_text(args, text, context)
            tqdm.write(f"JSON repair succeeded {attempt + 1}/{repair_attempts} for {context}", file=sys.stderr)
            return resp, data
        except Exception as exc:
            last_error = f"repair {attempt + 1}/{repair_attempts} failed: {type(exc).__name__}: {exc}; original={text[:500]}"
            tqdm.write(f"JSON repair retry {attempt + 1}/{repair_attempts} for {context}: {last_error[:300]}", file=sys.stderr)
    raise RuntimeError(f"Failed to parse JSON for {context}: {last_error}")

def _merge_scenes_to_max(scenes: list[tuple[float, float]], max_events: int) -> list[tuple[float, float]]:
    """Greedily merge the shortest scene into its shorter neighbour until at most
    ``max_events`` scenes remain, preserving the longest (most salient) cuts."""
    scenes = list(scenes)
    while len(scenes) > max(1, max_events):
        i = min(range(len(scenes)), key=lambda k: scenes[k][1] - scenes[k][0])
        if i == 0:
            j = 1
        elif i == len(scenes) - 1:
            j = i - 1
        else:
            left = scenes[i - 1][1] - scenes[i - 1][0]
            right = scenes[i + 1][1] - scenes[i + 1][0]
            j = i - 1 if left <= right else i + 1
        lo, hi = min(i, j), max(i, j)
        scenes[lo:hi + 1] = [(scenes[lo][0], scenes[hi][1])]
    return scenes


def _absorb_short_scenes(scenes: list[tuple[float, float]], min_len: float) -> list[tuple[float, float]]:
    """Merge any scene shorter than ``min_len`` seconds into an adjacent scene so
    tail/boundary artifacts (e.g. a cut detected right at the video end) don't
    become near-empty event-caption calls."""
    scenes = list(scenes)
    while len(scenes) > 1:
        i = next((k for k in range(len(scenes)) if scenes[k][1] - scenes[k][0] < min_len), None)
        if i is None:
            break
        j = i - 1 if i == len(scenes) - 1 else i + 1
        lo, hi = min(i, j), max(i, j)
        scenes[lo:hi + 1] = [(scenes[lo][0], scenes[hi][1])]
    return scenes


def detect_scenes_pyscenedetect(video_path: Path, args: argparse.Namespace, duration_sec: float) -> list[tuple[float, float]]:
    """Detect scene boundaries with PySceneDetect and return (start_sec, end_sec)
    ranges covering the whole video. Falls back to a single whole-video scene when
    no cuts are found. Scenes are merged down to ``--max-events`` if needed."""
    try:
        from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector, ThresholdDetector
    except ImportError as exc:
        raise RuntimeError(
            "PySceneDetect is required for --plan-segmenter scenedetect; install it with `pip install scenedetect`."
        ) from exc

    last_err: Exception | None = None
    video = None
    for backend in ("pyav", "opencv"):
        try:
            video = open_video(str(video_path), backend=backend)
            break
        except Exception as exc:
            last_err = exc
    if video is None:
        raise RuntimeError(f"PySceneDetect could not open {video_path}: {type(last_err).__name__}: {last_err}")

    fps = float(video.frame_rate or 0.0) or 25.0
    min_scene_len = max(1, round(args.scenedetect_min_scene_len_sec * fps))
    if args.scenedetect_detector == "adaptive":
        detector = AdaptiveDetector(adaptive_threshold=args.scenedetect_threshold, min_scene_len=min_scene_len)
    elif args.scenedetect_detector == "threshold":
        detector = ThresholdDetector(threshold=args.scenedetect_threshold, min_scene_len=min_scene_len)
    else:
        detector = ContentDetector(threshold=args.scenedetect_threshold, min_scene_len=min_scene_len)

    manager = SceneManager()
    manager.add_detector(detector)
    manager.detect_scenes(video, show_progress=False)
    scene_list = manager.get_scene_list()

    scenes = [(float(start.get_seconds()), float(end.get_seconds())) for start, end in scene_list]
    if not scenes:

        scenes = [(0.0, float(duration_sec))]

    scenes[-1] = (scenes[-1][0], float(duration_sec))
    scenes = _absorb_short_scenes(scenes, max(0.5, args.scenedetect_min_scene_len_sec))
    return _merge_scenes_to_max(scenes, args.max_events)


def build_plan_from_scenedetect(video_path: Path, duration_sec: float, args: argparse.Namespace, schema_version: str) -> dict[str, Any]:
    """Build a plan dict (same shape as the Gemini plan) from PySceneDetect scene
    cuts. There is no LLM call, so ``video_summary``/``video_type`` are left empty
    and per-event summaries are blank; boundaries come purely from scene detection.
    Reuses ``normalize_video_plan`` for the 180s split, clamping and schema fields."""
    scenes = detect_scenes_pyscenedetect(video_path, args, duration_sec)
    raw_segments = [
        {"event_id": f"event_{idx:02d}", "start": mmss(start), "end": mmss(end), "summary": ""}
        for idx, (start, end) in enumerate(scenes, start=1)
    ]
    plan = normalize_video_plan(
        {"event_segments": raw_segments, "video_summary": "", "video_type": ""},
        duration_sec, args.max_events, schema_version,
    )
    plan["segmenter"] = "scenedetect"
    plan["scenedetect_detector"] = args.scenedetect_detector
    plan["scenedetect_threshold"] = args.scenedetect_threshold
    plan["scenedetect_num_scenes"] = len(scenes)
    return plan


def normalize_video_plan(data: dict[str, Any], duration_sec: float, max_events: int, schema_version: str = "lightweight_caption_v3_no_audio_inventory") -> dict[str, Any]:
    max_segment_sec = 180.0
    segments: list[dict[str, Any]] = []
    raw_segments = data.get("event_segments") or []
    for idx, item in enumerate(raw_segments[:max_events], start=1):
        start = parse_mmss(str(item.get("start", "")))
        end = parse_mmss(str(item.get("end", "")))
        if start is None:
            start = 0.0 if not segments else segments[-1]["end_sec"]
        if end is None or end <= start:
            end = min(duration_sec, start + max(1.0, duration_sec / max(1, len(raw_segments))))
        start = max(0.0, min(float(start), duration_sec))
        end = max(start + 0.5, min(float(end), duration_sec))
        base_event_id = normalize_text(item.get("event_id")) or f"event_{idx:02d}"
        summary = normalize_text(item.get("summary"))
        split_index = 1
        cursor = start
        while cursor < end and len(segments) < max_events:
            chunk_end = min(end, cursor + max_segment_sec)
            suffix = f"_part{split_index:02d}" if end - start > max_segment_sec else ""
            segments.append({
                "event_id": f"{base_event_id}{suffix}",
                "start": mmss(cursor),
                "end": mmss(chunk_end),
                "start_sec": float(cursor),
                "end_sec": float(chunk_end),
                "summary": summary,
            })
            cursor = chunk_end
            split_index += 1
    if not segments:
        end = min(duration_sec, max_segment_sec)
        segments = [{"event_id": "event_01", "start": "00:00", "end": mmss(end), "start_sec": 0.0, "end_sec": end, "summary": normalize_text(data.get("video_summary"))}]
    normalized = {
        "video_summary": normalize_text(data.get("video_summary")),
        "video_type": normalize_text(data.get("video_type") or data.get("type") or data.get("category") or data.get("video_category")),
        "event_segments": segments,
        "structure_schema_version": schema_version,
    }
    if schema_version.startswith("adaptation_"):
        normalized["adaptation_domain"] = normalize_text(data.get("adaptation_domain"))
        normalized["knowledge_base_summary"] = normalize_text(data.get("knowledge_base_summary"))
        normalized["solved_problem_candidates"] = normalize_solved_problem_items(data.get("solved_problem_candidates"), "")
    return normalized

def format_prior_event_caption_context(event_captions: list[dict[str, Any]], max_chars: int = 12000) -> str:
    if not event_captions:
        return "None"
    lines = []
    for event in event_captions[-20:]:
        lines.append(f"{event.get('event_id')} {event.get('event_range', '')}:")
        global_caption = event.get("global_caption") or ""
        global_ts = event.get("global_timestamp") or ""
        if global_caption:
            lines.append(f"- global {global_ts}: {global_caption}")
        for audio in (event.get("audio_captions") or [])[:12]:
            if isinstance(audio, dict):
                lines.append(f"  audio {audio.get('timestamp', '')} · {audio.get('audio_type', '')}: {audio.get('caption', '')}")
        for visual in (event.get("visual_captions") or [])[:12]:
            if isinstance(visual, dict):
                lines.append(f"  visual {visual.get('timestamp', '')}: {visual.get('caption', '')}")
    text = "\n".join(lines)
    return text[-max_chars:]

def normalize_global_timestamp(value: Any, segment: SegmentSpec) -> str:
    timestamp = normalize_text(value)
    if timestamp and parse_timestamp_range(timestamp):
        return timestamp
    if timestamp:
        shifted = shift_range(timestamp, segment.start_sec)
        if parse_timestamp_range(shifted):
            return shifted
    return f"{mmss(segment.start_sec)}-{mmss(segment.end_sec)}"

def normalize_modality_timestamp(value: Any, segment: SegmentSpec, fallback: str) -> str:
    timestamp = normalize_text(value)
    if timestamp and parse_timestamp_range(timestamp):
        return timestamp
    if timestamp:
        shifted = shift_range(timestamp, segment.start_sec)
        if parse_timestamp_range(shifted):
            return shifted
    return fallback

def normalize_audio_caption_items(raw_items: Any, segment: SegmentSpec, fallback_ts: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(raw_items, list):
        return items
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        timestamp = normalize_modality_timestamp(
            item.get("timestamp") or item.get("audio_timestamp") or item.get("audio_time") or item.get("speech_timestamp"),
            segment,
            fallback_ts,
        )
        caption = clean_caption_text(
            item.get("caption")
            or item.get("audio_caption")
            or item.get("audio")
            or item.get("speech_caption")
            or item.get("sound_caption")
            or item.get("music_caption")
            or item.get("audio_description")
        )
        audio_type = normalize_audio_type(item.get("audio_type")) or infer_audio_type(caption)
        if timestamp or caption:
            items.append({
                "timestamp": timestamp,
                "audio_type": audio_type,
                "caption": caption or "Audio is present but not separately described.",
            })
    return items

def normalize_visual_caption_items(raw_items: Any, segment: SegmentSpec, fallback_ts: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(raw_items, list):
        return items
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        timestamp = normalize_modality_timestamp(
            item.get("timestamp") or item.get("visual_timestamp") or item.get("vision_timestamp") or item.get("visual_time"),
            segment,
            fallback_ts,
        )
        caption = clean_caption_text(item.get("caption") or item.get("visual_caption") or item.get("vision_caption") or item.get("visual") or item.get("visual_description"))
        if timestamp or caption:
            items.append({
                "timestamp": timestamp,
                "caption": caption or "Visual content is present but not separately described.",
            })
    return items

def event_caption_has_fine_grained_audio_visual(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    global_caption = normalize_text(data.get("global_caption"))
    audio_items = data.get("audio_captions")
    visual_items = data.get("visual_captions")
    if not global_caption or not isinstance(audio_items, list) or not isinstance(visual_items, list):
        return False
    if not audio_items or not visual_items:
        return False
    for item in audio_items:
        if not isinstance(item, dict) or not normalize_text(item.get("timestamp")) or not normalize_text(item.get("caption")):
            return False
    for item in visual_items:
        if not isinstance(item, dict) or not normalize_text(item.get("timestamp")) or not normalize_text(item.get("caption")):
            return False
    return True

def stage1_plan_schema_version(args: argparse.Namespace) -> str:
    if getattr(args, "stage1_caption_profile", "default") == "adaptation":
        base = "adaptation_lightweight_caption_v4_solved_problem_refs"
    else:
        base = "lightweight_caption_v3_no_audio_inventory"
    if getattr(args, "plan_segmenter", "gemini") == "scenedetect":
        base += f"_scenedetect_{getattr(args, 'scenedetect_detector', 'content')}"
    return base

def event_caption_schema_version(args: argparse.Namespace) -> str:
    if getattr(args, "stage1_caption_profile", "default") == "adaptation":
        return "event_omni_v6_adaptation_no_avsync_detail"
    return "event_omni_v4_no_avsync_detail"

def event_caption_matches_profile(data: Any, args: argparse.Namespace) -> bool:
    return (
        isinstance(data, dict)
        and data.get("event_caption_schema_version") == event_caption_schema_version(args)
        and event_caption_has_fine_grained_audio_visual(data)
    )

def normalize_event_caption(data: Any, segment: SegmentSpec) -> dict[str, Any]:
    event_range = {"start": mmss(segment.start_sec), "end": mmss(segment.end_sec)}
    fallback_global_ts = f"{event_range['start']}-{event_range['end']}"
    global_timestamp = normalize_global_timestamp(data.get("global_timestamp") or data.get("timestamp"), segment) if isinstance(data, dict) else fallback_global_ts
    global_caption = clean_caption_text(data.get("global_caption") or data.get("caption") or data.get("event_caption")) if isinstance(data, dict) else ""
    audio_captions = normalize_audio_caption_items(data.get("audio_captions"), segment, global_timestamp) if isinstance(data, dict) else []
    visual_captions = normalize_visual_caption_items(data.get("visual_captions"), segment, global_timestamp) if isinstance(data, dict) else []
    audio_captions = add_caption_indices(audio_captions, "audio_index")
    visual_captions = add_caption_indices(visual_captions, "visual_index")
    if not global_caption:
        parts = [item.get("caption", "") for item in visual_captions[:4]] + [item.get("caption", "") for item in audio_captions[:4]]
        global_caption = " ".join(part for part in parts if part) or segment.summary
    return {
        "event_id": segment.event_id,
        "event_range": event_range,
        "event_summary": segment.summary,
        "global_timestamp": global_timestamp or fallback_global_ts,
        "global_caption": global_caption,
        "audio_captions": audio_captions,
        "visual_captions": visual_captions,
        "event_caption_schema_version": "event_omni_v4_no_avsync_detail",
    }

def normalize_caption_refs(raw_refs: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if not isinstance(raw_refs, list):
        return refs
    for ref in raw_refs:
        if not isinstance(ref, dict):
            continue
        ref_type = normalize_text(ref.get("ref_type") or ref.get("type") or ref.get("modality")).lower()
        if ref_type not in {"audio", "visual"}:
            ref_type = "visual"
        try:
            index = int(ref.get("index"))
        except (TypeError, ValueError):
            index = 0
        refs.append({
            "ref_type": ref_type,
            "index": max(0, index),
            "timestamp": normalize_text(ref.get("timestamp")),
            "reason": normalize_text(ref.get("reason")),
        })
    return refs

def normalize_solved_problem_items(raw_items: Any, event_id: str = "") -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(raw_items, list):
        return items
    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        problem_id = normalize_text(item.get("problem_id")) or "_".join(part for part in (event_id, f"prob_{idx:02d}") if part) or f"prob_{idx:02d}"
        conditions = item.get("given_conditions")
        if not isinstance(conditions, list):
            conditions = [conditions] if normalize_text(conditions) else []
        normalized = {
            "problem_id": problem_id,
            "timestamp": normalize_text(item.get("timestamp")),
            "problem_type": normalize_text(item.get("problem_type")) or "other",
            "problem_statement": normalize_text(item.get("problem_statement") or item.get("question") or item.get("case_statement")),
            "question_modality_in_video": normalize_text(item.get("question_modality_in_video")) or "audiovisual",
            "visual_problem_text": normalize_text(item.get("visual_problem_text")),
            "audio_problem_text": normalize_text(item.get("audio_problem_text")),
            "given_conditions": [normalize_text(x) for x in conditions if normalize_text(x)],
            "solution_overview": normalize_text(item.get("solution_overview")),
            "problem_caption_refs": normalize_caption_refs(item.get("problem_caption_refs")),
            "solution_caption_refs": normalize_caption_refs(item.get("solution_caption_refs")),
            "answer_caption_refs": normalize_caption_refs(item.get("answer_caption_refs")),
            "final_answer_or_conclusion": normalize_text(item.get("final_answer_or_conclusion") or item.get("answer") or item.get("conclusion")),
            "transfer_potential": normalize_text(item.get("transfer_potential")),
            "source_event_id": event_id,
        }
        items.append(normalized)
    return [
        item for item in items
        if (
            item["problem_statement"]
            or item["visual_problem_text"]
            or item["audio_problem_text"]
            or item["problem_caption_refs"]
            or item["solution_caption_refs"]
            or item["answer_caption_refs"]
        )
    ]

def normalize_event_caption_for_profile(data: Any, segment: SegmentSpec, args: argparse.Namespace) -> dict[str, Any]:
    normalized = normalize_event_caption(data, segment)
    normalized["event_caption_schema_version"] = event_caption_schema_version(args)
    if getattr(args, "stage1_caption_profile", "default") == "adaptation":
        normalized["solved_problem_items"] = normalize_solved_problem_items(
            data.get("solved_problem_items") if isinstance(data, dict) else None,
            segment.event_id,
        )
    return normalized

def collect_solved_problems(full_video_caption: dict[str, Any], event_captions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in full_video_caption.get("solved_problem_candidates") or []:
        if isinstance(item, dict):
            plan_item = dict(item)
            plan_item.setdefault("source", "video_plan")
            items.append(plan_item)
    for event in event_captions:
        if not isinstance(event, dict):
            continue
        for item in event.get("solved_problem_items") or []:
            if isinstance(item, dict):
                event_item = dict(item)
                event_item.setdefault("source", "event_caption")
                items.append(event_item)
    return [
        item for item in items
        if (
            normalize_text(item.get("problem_statement"))
            or normalize_text(item.get("visual_problem_text"))
            or normalize_text(item.get("audio_problem_text"))
            or item.get("problem_caption_refs")
            or item.get("solution_caption_refs")
            or item.get("answer_caption_refs")
        )
    ]

def build_caption_bundle(item: VideoItem, duration_sec: float, full_video_caption: dict[str, Any], event_captions: list[dict[str, Any]], gemini_video_url: str = "", caption_profile: str = "default") -> dict[str, Any]:
    bundle = {
        "video_id": item.video_id,
        "raw_video_local_path": str(item.raw_video_path),
        "time_video_local_path": str(item.time_video_path),
        "time_video_oss_path": item.time_video_oss_path,
        "gemini_video_url": gemini_video_url or item.time_video_oss_path,
        "video_duration_sec": duration_sec,
        "caption_profile": caption_profile,
        "whole_video_summary": full_video_caption.get("video_summary", ""),
        "video_type": full_video_caption.get("video_type", ""),
        "event_segments": full_video_caption.get("event_segments", []),
        "event_captions": event_captions,
    }
    if caption_profile == "adaptation":
        bundle["adaptation_domain"] = normalize_text(full_video_caption.get("adaptation_domain"))
        bundle["knowledge_base_summary"] = normalize_text(full_video_caption.get("knowledge_base_summary"))
        bundle["solved_problem_items"] = collect_solved_problems(full_video_caption, event_captions)
    return bundle

def video_workspace(args: argparse.Namespace, raw_video_path: Path, data_root: Path) -> tuple[str, str, Path]:
    relative, video_id = video_identity(raw_video_path, data_root)
    raw_item_dir = args.output_dir / "raw" / video_id
    ensure_dir(raw_item_dir)
    return relative, video_id, raw_item_dir

def video_identity(raw_video_path: Path, data_root: Path) -> tuple[str, str]:
    relative = str(raw_video_path.relative_to(data_root)) if data_root.is_dir() and raw_video_path.is_relative_to(data_root) else raw_video_path.name
    video_id = safe_slug(Path(relative).with_suffix("").as_posix())
    return relative, video_id

def caption_bundle_is_complete(bundle: Any) -> bool:
    if not isinstance(bundle, dict):
        return False
    event_captions = bundle.get("event_captions")
    if not isinstance(event_captions, list) or not event_captions:
        return False
    return all(event_caption_has_fine_grained_audio_visual(event) for event in event_captions if isinstance(event, dict))

def probe_video_duration_for_selection(args: argparse.Namespace, raw_video_path: Path, data_root: Path) -> tuple[str, str, Path, float | None]:
    relative, video_id = video_identity(raw_video_path, data_root)
    raw_item_dir = args.output_dir / "raw" / video_id
    ensure_dir(raw_item_dir)
    probe_path = raw_item_dir / "probe.json"
    if args.resume and (cached := cached_json(probe_path)) is not None:
        probe = cached
    else:
        log_video_stage(video_id, "probe video for duration filter")
        probe = safe_probe_video(raw_video_path)
        if probe is None:
            write_json(raw_item_dir / "skip.json", {
                "reason": "probe_failed",
                "video_path": str(raw_video_path),
            })
            return relative, video_id, raw_item_dir, None
        write_json(probe_path, probe)
    has_audio = any((stream.get("codec_type") or "") == "audio" for stream in probe.get("streams") or [])
    if not has_audio:
        write_json(raw_item_dir / "skip.json", {
            "reason": "audio_filter",
            "has_audio": False,
        })
        return relative, video_id, raw_item_dir, None
    duration_sec = duration_from_probe(probe)
    if duration_sec < args.min_video_duration_sec or (args.max_video_duration_sec > 0 and duration_sec > args.max_video_duration_sec):
        write_json(raw_item_dir / "skip.json", {
            "reason": "duration_filter",
            "duration_sec": duration_sec,
            "min_video_duration_sec": args.min_video_duration_sec,
            "max_video_duration_sec": args.max_video_duration_sec,
        })
        return relative, video_id, raw_item_dir, None
    return relative, video_id, raw_item_dir, duration_sec

def select_caption_stage_videos(args: argparse.Namespace, videos: list[Path], data_root: Path) -> tuple[list[Path], int]:
    target = int(args.num_candidate_videos or 0)
    if target <= 0:
        return videos, 0
    existing_complete_ids: set[str] = set()
    if args.resume:
        for context in discover_caption_bundle_contexts(args.output_dir / "raw"):
            if caption_bundle_is_complete(context.get("caption_bundle")):
                existing_complete_ids.add(context["video_id"])
    remaining_slots = max(0, target - len(existing_complete_ids))
    selected: list[Path] = []
    for video in videos:
        _, video_id = video_identity(video, data_root)
        if video_id in existing_complete_ids:
            continue
        _, _, _, duration_sec = probe_video_duration_for_selection(args, video, data_root)
        if duration_sec is None:
            continue
        selected.append(video)
        if len(selected) >= remaining_slots:
            break
    return selected, len(existing_complete_ids)

def log_video_stage(video_id: str, stage: str) -> None:
    tqdm.write(f"[{video_id}] {stage}", file=sys.stderr)


def run_caption_stage(args: argparse.Namespace, raw_video_path: Path, data_root: Path) -> dict[str, Any] | None:
    relative, video_id, raw_item_dir = video_workspace(args, raw_video_path, data_root)
    with stage_context("caption"):
        probe_path = raw_item_dir / "probe.json"
        if args.resume and (cached := cached_json(probe_path)) is not None:
            log_video_stage(video_id, "reuse probe")
            probe = cached
        else:
            log_video_stage(video_id, "probe video")
            probe = safe_probe_video(raw_video_path)
            if probe is None:
                write_json(raw_item_dir / "skip.json", {
                    "reason": "probe_failed",
                    "video_path": str(raw_video_path),
                })
                return None
            write_json(probe_path, probe)
        duration_sec = duration_from_probe(probe)
        if duration_sec < args.min_video_duration_sec or (args.max_video_duration_sec > 0 and duration_sec > args.max_video_duration_sec):
            write_json(raw_item_dir / "skip.json", {"reason": "duration_filter", "duration_sec": duration_sec})
            return None

        time_video_path = raw_video_path if args.use_raw_video else raw_item_dir / f"time_{raw_video_path.stem}.mp4"
        video_paths_path = raw_item_dir / "video_paths.json"
        cached_paths = cached_json(video_paths_path) if args.resume else None
        if args.use_raw_video:
            log_video_stage(video_id, "use raw video")
            oss_path = build_oss_path(args.oss_prefix, relative, raw_video_path.name)
        elif cached_paths and time_video_path.exists() and time_video_path.stat().st_size > 0 and not args.force_time_video and not args.force_upload:
            log_video_stage(video_id, "reuse timestamp video/upload paths")
            oss_path = str(cached_paths.get("time_video_oss_path") or build_oss_path(args.oss_prefix, relative, time_video_path.name))
        else:
            log_video_stage(video_id, "make timestamp video")
            make_time_video(
                raw_video_path,
                time_video_path,
                force=args.force_time_video,
                width=args.time_video_width,
                fps=args.time_video_fps,
                crf=args.time_video_crf,
                fontfile=args.time_video_fontfile,
            )
            oss_path = build_oss_path(args.oss_prefix, relative, time_video_path.name)
            log_video_stage(video_id, "upload timestamp video")
            upload_to_oss(time_video_path, oss_path, args, force=args.force_upload, skip=args.skip_upload)
            if args.oss_public_read and not args.skip_upload:
                try:
                    log_video_stage(video_id, "set timestamp video public-read")
                    set_oss_object_acl(oss_path, "public-read", args)
                except Exception as exc:
                    tqdm.write(f"[{video_id}] set public-read skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
            log_video_stage(video_id, "uploaded timestamp video")
        gemini_video_ref = choose_gemini_video_ref(oss_path, time_video_path, args)
        gemini_video_url = str(gemini_video_ref)
        item = VideoItem(video_id, raw_video_path, time_video_path, oss_path, relative)
        video_paths = dataclasses.asdict(item)
        video_paths["gemini_video_url"] = gemini_video_url
        write_json(video_paths_path, video_paths)

        plan_path = raw_item_dir / "video_plan.json"
        plan_resp_path = raw_item_dir / "video_plan_response.json"
        plan_schema_version = stage1_plan_schema_version(args)
        if args.resume and (cached_plan := cached_json(plan_path)) is not None and cached_plan.get("event_segments") and cached_plan.get("structure_schema_version") == plan_schema_version:
            log_video_stage(video_id, "reuse lightweight Gemini video plan")
            full_video_caption = cached_plan
        elif args.plan_segmenter == "scenedetect":
            log_video_stage(video_id, f"segment video with PySceneDetect ({args.scenedetect_detector})")
            full_video_caption = build_plan_from_scenedetect(time_video_path, duration_sec, args, plan_schema_version)
            log_video_stage(video_id, f"PySceneDetect produced {len(full_video_caption['event_segments'])} segment(s)")
            write_json(plan_path, full_video_caption)
        else:
            plan_template = ADAPTATION_VIDEO_STRUCTURE_PROMPT_TEMPLATE if args.stage1_caption_profile == "adaptation" else VIDEO_STRUCTURE_PROMPT_TEMPLATE
            plan_prompt = plan_template.format(max_events=args.max_events)
            if args.resume and (cached_resp := cached_json(plan_resp_path)) is not None:
                log_video_stage(video_id, "reparse cached Gemini video plan response")
                plan_data = extract_json(response_text(cached_resp))
            else:
                log_video_stage(video_id, "call Gemini video plan")
                plan_resp, plan_data = call_json(lambda: call_gemini_video(args, plan_prompt, gemini_video_ref, media_resolution=args.gemini_plan_media_resolution, thinking_level=args.gemini_plan_thinking_level), f"video_plan_{video_id}", args)
                write_json(plan_resp_path, plan_resp)
            full_video_caption = normalize_video_plan(plan_data, duration_sec, args.max_events, plan_schema_version)
            write_json(plan_path, full_video_caption)

        event_captions = []
        for seg_data in tqdm(full_video_caption["event_segments"], desc=f"Caption {video_id}", unit="event", leave=False):
            segment = SegmentSpec(seg_data["event_id"], seg_data["start_sec"], seg_data["end_sec"], seg_data.get("summary", ""))
            event_template = ADAPTATION_EVENT_OMNI_CAPTION_PROMPT_TEMPLATE if args.stage1_caption_profile == "adaptation" else EVENT_OMNI_CAPTION_PROMPT_TEMPLATE
            prompt = event_template.format(
                event_start_text=mmss(segment.start_sec),
                event_end_text=mmss(segment.end_sec),
                prior_event_caption_context=format_prior_event_caption_context(event_captions),
            )
            cap_path = raw_item_dir / f"{segment.event_id}_caption.json"
            cap_resp_path = raw_item_dir / f"{segment.event_id}_caption_response.json"
            if args.resume and (cached_cap := cached_json(cap_path)) is not None and event_caption_matches_profile(cached_cap, args):
                normalized = cached_cap
            else:
                if args.resume and (cached_resp := cached_json(cap_resp_path)) is not None:
                    event_data = extract_json(response_text(cached_resp))
                    if not event_caption_has_fine_grained_audio_visual(event_data) or not event_caption_matches_profile(cached_cap, args):
                        log_video_stage(video_id, f"refresh {segment.event_id} caption for {args.stage1_caption_profile} schema")
                        resp, event_data = call_json(lambda: call_gemini_video(args, prompt, gemini_video_ref), f"{video_id}_{segment.event_id}_caption", args)
                        write_json(cap_resp_path, resp)
                else:
                    resp, event_data = call_json(lambda: call_gemini_video(args, prompt, gemini_video_ref), f"{video_id}_{segment.event_id}_caption", args)
                    write_json(cap_resp_path, resp)
                normalized = normalize_event_caption_for_profile(event_data, segment, args)
                write_json(cap_path, normalized)
            event_captions.append(normalized)

        log_video_stage(video_id, "build caption bundle")
        caption_bundle = build_caption_bundle(item, duration_sec, full_video_caption, event_captions, gemini_video_url, args.stage1_caption_profile)
        caption_bundle_path = raw_item_dir / "caption_bundle.json"
        write_json(caption_bundle_path, caption_bundle)
        return {"video_id": video_id, "item": item, "duration_sec": duration_sec, "caption_bundle": caption_bundle, "caption_bundle_path": caption_bundle_path}


def discover_caption_bundle_contexts(caption_raw_dir: Path) -> list[dict[str, Any]]:
    contexts = []
    for caption_bundle_path in sorted(caption_raw_dir.glob("*/caption_bundle.json")):
        raw_item_dir = caption_bundle_path.parent
        contexts.append({
            "video_id": raw_item_dir.name,
            "raw_item_dir": raw_item_dir,
            "caption_bundle_path": caption_bundle_path,
        })
    return contexts

def load_caption_bundle_context(context: dict[str, Any]) -> dict[str, Any] | None:
    caption_bundle_path = context.get("caption_bundle_path")
    if not isinstance(caption_bundle_path, Path):
        caption_bundle_path = Path(caption_bundle_path)
    caption_bundle = cached_json(caption_bundle_path)
    if not isinstance(caption_bundle, dict):
        return None
    loaded = dict(context)
    video_id = normalize_text(caption_bundle.get("video_id")) or normalize_text(context.get("video_id")) or caption_bundle_path.parent.name
    loaded["video_id"] = video_id
    loaded["raw_item_dir"] = caption_bundle_path.parent
    loaded["caption_bundle"] = caption_bundle
    loaded["caption_bundle_path"] = caption_bundle_path
    return loaded

def collect_caption_bundles(output_dir: Path) -> list[dict[str, Any]]:
    bundles = []
    for context in discover_caption_bundle_contexts(output_dir / "raw"):
        loaded_context = load_caption_bundle_context(context)
        if loaded_context is None:
            continue
        bundle = dict(loaded_context["caption_bundle"])
        bundle.setdefault("video_id", context["video_id"])
        bundle["caption_bundle_path"] = str(loaded_context["caption_bundle_path"].relative_to(output_dir))
        bundles.append(bundle)
    return bundles


def process_video(args: argparse.Namespace, raw_video_path: Path, data_root: Path) -> list[dict[str, Any]]:
    result = run_caption_stage(args, raw_video_path, data_root)
    return [result["caption_bundle"]] if result else []


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    videos = find_videos(args.data_path, args.source_glob)
    existing_caption_bundles_before = 0
    if args.num_candidate_videos > 0:
        videos, existing_caption_bundles_before = select_caption_stage_videos(args, videos, args.data_path)
    if args.dry_run:
        print(json.dumps({
            "matched_videos": [str(v) for v in videos],
            "stage": "caption",
            "existing_caption_bundles_before": existing_caption_bundles_before,
            "caption_top_up_target": args.num_candidate_videos if args.num_candidate_videos > 0 else None,
        }, ensure_ascii=False, indent=2))
        return 0
    stage_results: list[dict[str, Any]] = []
    max_workers = max(1, min(args.video_workers, len(videos) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(process_video, args, video, args.data_path): video
            for video in videos
        }
        for future in tqdm(as_completed(future_to_video), total=len(future_to_video), desc="Videos", unit="video"):
            video = future_to_video[future]
            try:
                stage_results.extend(future.result())
            except Exception as exc:
                error_dir = error_dir_for_video(args, video, args.data_path)
                write_json(error_dir / "error.json", {"video": str(video), "error": f"{type(exc).__name__}: {exc}"})
                tqdm.write(f"ERROR {video}: {type(exc).__name__}: {exc}", file=sys.stderr)
    output_results = collect_caption_bundles(args.output_dir)
    manifest = {
        "version": "simple_v2_staged",
        "stage": "caption",
        "data_path": str(args.data_path) if args.data_path is not None else "",
        "output_dir": str(args.output_dir),
        "stage_output": "caption_bundles.json",
        "oss_prefix": args.oss_prefix,
        "num_videos": len(videos),
        "existing_caption_bundles_before": existing_caption_bundles_before,
        "caption_top_up_target": args.num_candidate_videos if args.num_candidate_videos > 0 else 0,
        "num_caption_bundles": len(output_results),
        "num_stage_results": len(stage_results),
        "num_stage_output_results": len(output_results),
        "stage_api_limits": {
            "caption": {
                "qps": args.stage1_api_qps if args.stage1_api_qps is not None else args.api_qps,
                "max_inflight_queries": args.stage1_max_inflight_queries if args.stage1_max_inflight_queries is not None else args.max_inflight_queries,
            },
        },
    }
    write_json(args.output_dir / "caption_bundles.json", output_results)
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if stage_results or videos:
        return 0
    if args.num_candidate_videos > 0 and existing_caption_bundles_before >= args.num_candidate_videos:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
