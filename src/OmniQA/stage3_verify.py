"""Stage 3 · VLM clue verification for generated AV questions (anti-hallucination).

Verifies, against the ACTUAL video, the decisive clues that each generated question
was built from (accepted.jsonl from Stage 2, OmniQA.stage2_generate). Captions may contain
hallucinated content, so a question grounded in a hallucinated clue must be rejected.

Default VLM: qwen3.6-flash (vision-only)
  • visual clue → ffmpeg-cut clip around the clue timestamp, verified with the VLM.
  • audio clue  → cannot be verified by a vision-only model, marked
    "skipped_vision_only_vlm" (not counted for accept/reject).
With --vlm-model qwen3.5-omni-plus (audio+video):
  • audio clue → ffmpeg-extracted audio clip verified the same way.

A clue PASSES iff: pass=true AND support_strength >= --support-threshold AND
unsupported_claims is empty. A row is REJECTED iff any verifiable clue FAILS.

Input rows carry the video paths / duration that Stage 2 attaches to every row, so
this stage runs without re-joining caption bundles. Video path resolution: the
optional caption<->video map (--video-map) first, then the row's own
video_path / raw_video_local_path / time_video_local_path.

Concurrency/resume: RateLimiter + ThreadPoolExecutor + append_jsonl resume keyed by
(source, video_id, row idx).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

WRITE_LOCK = threading.Lock()

# ══════════════════════════ Prompts ══════════════════════════

VISUAL_VERIFY_PROMPT = r"""You are verifying whether a video clip supports a specific VISUAL claim.

Claim: {fact}

(The claim comes from an automatically generated video caption, quoted as: "{quote}".
Do NOT use this quote as ground truth — captions may be hallucinated. Judge ONLY from
the video frames you actually see. Be strict.)

Rules:
- pass = true ONLY if the clip clearly and unambiguously shows the claimed content.
- If the claim is only partially visible, ambiguous, or contradicted, pass = false.
- List every part of the claim you could NOT confirm in unsupported_claims.

Return ONLY valid JSON (no other text):
{{"contains_visual_claim": true, "pass": true, "support_strength": 0.0,
  "visible_evidence": "what you actually see",
  "unsupported_claims": ["..."], "rationale": "short reason"}}
"""

AUDIO_VERIFY_PROMPT = r"""You are verifying whether an audio clip supports a specific AUDIO claim.

Claim: {fact}

(The claim comes from an automatically generated audio caption, quoted as: "{quote}".
Do NOT use this quote as ground truth — captions may be hallucinated. Judge ONLY from
what you actually hear. Be strict.)

Rules:
- pass = true ONLY if the audio clearly contains the claimed speech/sound/music content.
- If the claim is only partially audible, ambiguous, or contradicted, pass = false.
- List every part of the claim you could NOT confirm in unsupported_claims.

Return ONLY valid JSON (no other text):
{{"contains_audio_claim": true, "pass": true, "support_strength": 0.0,
  "audible_evidence": "what you actually hear",
  "unsupported_claims": ["..."], "rationale": "short reason"}}
"""


# ══════════════════════════ Rate limiter ══════════════════════════

class RateLimiter:
    def __init__(self, qps: float):
        self.interval = 1.0 / qps if qps > 0 else 0.0
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait_sec = max(0.0, self.next_start - now)
            self.next_start = max(now, self.next_start) + self.interval
        if wait_sec > 0:
            time.sleep(wait_sec)


# ══════════════════════════ Timestamp / clip helpers ══════════════════════════

def _to_sec(ts: str) -> float | None:
    ts = str(ts).strip()
    if not ts:
        return None
    try:
        parts = [float(p) for p in ts.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 1:
        return parts[0]
    return None


def parse_ts_range(ts: str) -> tuple[float, float] | None:
    """'MM:SS-MM:SS' or 'MM:SS' -> (start, end) seconds; None if unparseable."""
    ts = str(ts or "").strip()
    if not ts:
        return None
    parts = re.split(r"\s*-\s*", ts)
    if len(parts) >= 2:
        a, b = _to_sec(parts[0]), _to_sec(parts[1])
        if a is not None and b is not None and b > a:
            return a, b
        if a is not None:
            return a, a
    a = _to_sec(parts[0])
    return (a, a) if a is not None else None


def clip_window(clue_ts: str, duration: float | None, padding: float,
                max_len: float) -> tuple[float, float] | None:
    """Return (start, length) for the verification clip; None = use whole video."""
    rng = parse_ts_range(clue_ts)
    if rng is None:
        return None
    a, b = rng
    mid = (a + b) / 2.0
    length = min(max(b - a + 2 * padding, 2 * padding + 2.0), max_len)
    start = max(0.0, mid - length / 2.0)
    if duration and start + length > duration:
        start = max(0.0, duration - length)
    return start, length


_VCODEC: str | None = None


def _pick_vcodec() -> str:
    """This box's ffmpeg may lack libx264 — probe once and fall back."""
    global _VCODEC
    if _VCODEC:
        return _VCODEC
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30).stdout
        for c in ("libx264", "libopenh264", "mpeg4"):
            if c in out:
                _VCODEC = c
                break
    except Exception:
        pass
    _VCODEC = _VCODEC or "mpeg4"
    return _VCODEC


def cut_clip(video_path: str, start: float, length: float, cache_dir: Path,
             audio_only: bool = False) -> str:
    """Cut (and cache) a verification clip. Vision clips are downscaled & muted;
    audio clips are mp3. Cached by content hash of (path, size, mtime, window)."""
    p = Path(video_path)
    st = p.stat()
    key = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|{start:.2f}|{length:.2f}|{audio_only}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    ext = "mp3" if audio_only else "mp4"
    out = cache_dir / f"{h}.{ext}"
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    tmp = out.with_suffix(f".tmp.{ext}")
    if audio_only:
        cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}",
               "-i", str(p), "-vn", "-acodec", "libmp3lame", "-q:a", "4", str(tmp)]
    else:
        codec = _pick_vcodec()
        if codec == "libx264":
            vargs = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30"]
        elif codec == "libopenh264":
            vargs = ["-c:v", "libopenh264", "-b:v", "1M"]
        else:
            vargs = ["-c:v", "mpeg4", "-q:v", "5"]
        cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}",
               "-i", str(p), "-an", "-vf", "scale=640:-2", *vargs, str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True)
    tmp.replace(out)
    return str(out)


# ══════════════════════════ VLM call ══════════════════════════

def extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    try:
        output = response.output
        if hasattr(output, "choices") and output.choices:
            content = output.choices[0].message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return "".join(item.get("text", "") if isinstance(item, dict) else str(item)
                               for item in content).strip()
        if hasattr(output, "text") and output.text:
            return output.text.strip()
    except (AttributeError, IndexError, TypeError):
        pass
    return str(response)


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def call_vlm(media_path: str, media_kind: str, prompt: str, args: argparse.Namespace,
             limiter: RateLimiter) -> tuple[dict | None, str, str]:
    """Returns (verdict_json|None, raw_text, error). media_kind: 'video'|'audio'."""
    from dashscope import MultiModalConversation

    if media_kind == "audio":
        media_item = {"audio": f"file://{media_path}"}
    else:
        media_item = {"video": f"file://{media_path}", "fps": 2.0}
    messages = [{"role": "user", "content": [media_item, {"text": prompt}]}]

    last_error = ""
    for attempt in range(max(1, args.max_retries)):
        limiter.wait()
        try:
            response = MultiModalConversation.call(
                model=args.vlm_model, messages=messages, api_key=args.api_key)
            if response.status_code != 200:
                last_error = f"API error {response.status_code}: {response.code} - {response.message}"
                if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < args.max_retries:
                    time.sleep(args.retry_backoff * (attempt + 1))
                    continue
                return None, "", last_error
            raw = extract_response_text(response)
            verdict = extract_json(raw)
            if verdict is None:
                return None, raw, f"could not parse verdict JSON: {raw[:200]}"
            return verdict, raw, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < args.max_retries:
                time.sleep(args.retry_backoff * (attempt + 1))
                continue
    return None, "", last_error


def judge_verdict(verdict: dict, threshold: float) -> bool:
    """STRICT (quality-first): pass iff the VLM passes the claim, is confident enough,
    and reports NO unsupported part of the claim."""
    if not verdict.get("pass"):
        return False
    try:
        strength = float(verdict.get("support_strength", 0.0))
    except (TypeError, ValueError):
        strength = 0.0
    if strength < threshold:
        return False
    if verdict.get("unsupported_claims"):
        return False
    return True


# ══════════════════════════ Per-row verification ══════════════════════════

def load_video_map(path: str) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Load outputs/video_map.jsonl -> ((source, video_id) -> path, video_id -> path).
    Only 'mapped' rows are kept."""
    by_pair: dict[tuple[str, str], str] = {}
    by_id: dict[str, str] = {}
    if not path or not Path(path).exists():
        return by_pair, by_id
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "mapped" or not row.get("video_path"):
                continue
            by_pair[(row.get("source", ""), row.get("video_id", ""))] = row["video_path"]
            by_id.setdefault(row.get("video_id", ""), row["video_path"])
    return by_pair, by_id


def resolve_video_path(row: dict, video_root: str = "",
                       map_by_pair: dict | None = None,
                       map_by_id: dict | None = None) -> str:
    # 0. the prebuilt caption<->video map (most reliable)
    src = str(row.get("source", "")).split("::")[0]   # strip path-hash disambiguator
    vid = str(row.get("video_id", ""))
    for p in ((map_by_pair or {}).get((src, vid)), (map_by_id or {}).get(vid)):
        if p:
            try:
                if Path(p).exists():
                    return p
            except OSError:
                pass
    cands: list[Path] = []
    vp = str(row.get("video_path") or "")
    if vp:
        cands.append(Path(vp))
    for k in ("raw_video_local_path", "time_video_local_path"):
        p = str(row.get(k) or "")
        if p:
            cands.append(Path(p))
    tp = str(row.get("time_video_local_path") or "")
    if tp and not Path(tp).is_absolute() and video_root:
        cands.append(Path(video_root) / tp)
    for c in cands:
        try:
            if c.exists():
                return str(c)
        except OSError:
            continue
    return ""


def verify_one(row: dict, idx: int, args: argparse.Namespace,
               limiter: RateLimiter, cache_dir: Path) -> dict:
    video_id = row.get("video_id", "")
    source = row.get("source", "")
    candidate = row.get("candidate") or {}
    duration = row.get("video_duration_sec")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None

    result: dict[str, Any] = {
        "idx": idx, "video_id": video_id, "source": source,
        "task_type_id": row.get("task_type_id", ""),
        "visual": {"status": "no_visual_clue"},
        "audio": {"status": "no_audio_clue"},
        "status": "ok", "error": "",
    }

    video_path = resolve_video_path(row, args.video_root, args._map_by_pair, args._map_by_id)
    if not video_path:
        result["status"] = "error"
        result["error"] = "video file not found (video_path / raw_video_local_path / time_video_local_path)"
        result["decision"] = "error"
        result["reject_reason"] = result["error"]
        out = dict(row)
        out["clue_verification"] = result
        return out
    result["video_path"] = video_path

    vlm_supports_audio = "omni" in args.vlm_model.lower()

    # ---- visual clue ----
    vc = candidate.get("visual_clue") or {}
    visual_claim = vc.get("fact") or vc.get("quote")   # quote alone is also checkable
    if visual_claim:
        try:
            window = clip_window(vc.get("timestamp", ""), duration,
                                 args.clip_padding, args.clip_max_len)
            if window is None:
                # no usable timestamp: sample the whole video at low fps instead
                start, length = 0.0, min(duration or args.clip_max_len, args.clip_max_len)
            else:
                start, length = window
            prompt = VISUAL_VERIFY_PROMPT.format(fact=visual_claim, quote=vc.get("quote", ""))
            clip = cut_clip(video_path, start, length, cache_dir, audio_only=False)
            verdict, raw, err = call_vlm(clip, "video", prompt, args, limiter)
            if verdict is None:
                result["visual"] = {"status": "error", "error": err, "raw": (raw or "")[:300]}
            else:
                passed = judge_verdict(verdict, args.support_threshold)
                result["visual"] = {"status": "pass" if passed else "fail",
                                    "clue": vc, "clip_window": [round(start, 2), round(length, 2)],
                                    "verdict": verdict}
        except Exception as exc:
            result["visual"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    # ---- audio clue ----
    ac = candidate.get("audio_clue") or {}
    audio_claim = ac.get("fact") or ac.get("quote")
    if audio_claim:
        if not vlm_supports_audio:
            result["audio"] = {"status": "skipped_vision_only_vlm", "clue": ac}
        else:
            try:
                window = clip_window(ac.get("timestamp", ""), duration,
                                     args.clip_padding, args.clip_max_len)
                if window is None:
                    start, length = 0.0, min(duration or args.clip_max_len, args.clip_max_len)
                else:
                    start, length = window
                clip = cut_clip(video_path, start, length, cache_dir, audio_only=True)
                prompt = AUDIO_VERIFY_PROMPT.format(fact=audio_claim, quote=ac.get("quote", ""))
                verdict, raw, err = call_vlm(clip, "audio", prompt, args, limiter)
                if verdict is None:
                    result["audio"] = {"status": "error", "error": err, "raw": raw[:300]}
                else:
                    passed = judge_verdict(verdict, args.support_threshold)
                    result["audio"] = {"status": "pass" if passed else "fail",
                                       "clue": ac, "clip_window": [round(start, 2), round(length, 2)],
                                       "verdict": verdict}
            except Exception as exc:
                result["audio"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    return finalize(row, result, args)


def finalize(row: dict, result: dict, args: argparse.Namespace) -> dict:
    """Row-level accept/reject: rejected iff any VERIFIABLE clue fails."""
    if result.get("status") == "error" and "decision" not in result:
        result["decision"] = "error"
        result["reject_reason"] = result.get("error", "")
        out = dict(row)
        out["clue_verification"] = result
        return out
    v, a = result["visual"]["status"], result["audio"]["status"]
    verifiable = [s for s in (v, a) if s in ("pass", "fail")]
    failed = [s for s in (v, a) if s == "fail"]
    errors = [s for s in (v, a) if s == "error"]
    if failed:
        result["decision"] = "rejected"
        result["reject_reason"] = (
            f"visual clue unsupported: {result['visual'].get('verdict', {}).get('rationale', '')[:200]}"
            if v == "fail" else
            f"audio clue unsupported: {result['audio'].get('verdict', {}).get('rationale', '')[:200]}")
    elif errors and not verifiable:
        # nothing could be verified at all — keep out of the accepted pool, but not a
        # content rejection (transient API / IO errors are retryable on rerun).
        result["decision"] = "error"
        result["reject_reason"] = "no clue could be verified (errors)"
    else:
        result["decision"] = "accepted"
        result["reject_reason"] = ""
    out = dict(row)
    out["clue_verification"] = result
    return out


# ══════════════════════════ IO / runner ══════════════════════════

def append_jsonl(path: Path, row: dict) -> None:
    with WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _row_key(row: dict) -> str:
    ver = row.get("clue_verification") or {}
    return f"{row.get('source', '')}::{row.get('video_id', '')}::{ver.get('idx', '')}"


def load_latest_rows(path: Path) -> dict[str, dict]:
    """verified_all.jsonl may contain the same key several times (errored rows are
    retried and re-appended) — the LAST occurrence wins."""
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[_row_key(row)] = row
    return latest


def load_done_keys(path: Path) -> set[str]:
    done: set[str] = set()
    for key, row in load_latest_rows(path).items():
        ver = row.get("clue_verification") or {}
        # retry rows whose verification itself errored
        if ver.get("status") == "error" or ver.get("decision") == "error":
            continue
        done.add(key)
    return done


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 3 · VLM clue verification (anti-hallucination)")
    p.add_argument("--input", type=Path, required=True, help="accepted.jsonl from Stage 2 (OmniQA.stage2_generate)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--vlm-model", default="qwen3.6-flash",
                   help="qwen3.6-flash (vision-only) or qwen3.5-omni-plus (audio+video)")
    p.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY",
                                                       os.environ.get("QWEN_API_KEY", "")))
    p.add_argument("--api-base", default=os.environ.get("DASHSCOPE_BASE_URL", ""),
                   help="optional DashScope HTTP API base URL (SDK default when empty)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--qps", type=float, default=2.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-backoff", type=float, default=5.0)
    p.add_argument("--clip-padding", type=float, default=2.0,
                   help="seconds of context on each side of the clue timestamp")
    p.add_argument("--clip-max-len", type=float, default=45.0)
    p.add_argument("--support-threshold", type=float, default=0.75)
    p.add_argument("--video-root", default="",
                   help="optional fallback root for resolving RELATIVE time_video_local_path "
                        "in rows that lack an absolute path (usually unneeded — Stage 2 rows "
                        "carry absolute paths)")
    p.add_argument("--video-map",
                   default=str(Path.cwd() / "data" / "video_map.jsonl"),
                   help="optional caption<->video map (JSONL: {source, video_id, video_path, "
                        "status:'mapped'}); used first if present, otherwise path resolution "
                        "falls back to the paths recorded in each row")
    p.add_argument("--max-rows", type=int, default=0, help="0 = all (use small for smoke tests)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: Set DASHSCOPE_API_KEY or pass --api-key.", file=sys.stderr)
        return 1
    if args.api_base:
        import dashscope
        dashscope.base_http_api_url = args.api_base
    args._map_by_pair, args._map_by_id = load_video_map(args.video_map)
    if args._map_by_id:
        print(f"Video map: {len(args._map_by_pair)} mapped entries from {args.video_map}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "clip_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.max_rows > 0:
        rows = rows[:args.max_rows]
    print(f"Loaded {len(rows)} rows from {args.input}", flush=True)
    print(f"VLM: {args.vlm_model} (audio clues: "
          f"{'verified' if 'omni' in args.vlm_model.lower() else 'skipped — vision-only model'})",
          flush=True)

    all_path = args.output_dir / "verified_all.jsonl"
    done = load_done_keys(all_path) if args.resume else set()
    pending = [(i, r) for i, r in enumerate(rows)
               if f"{r.get('source', '')}::{r.get('video_id', '')}::{i}" not in done]
    print(f"Resumable done: {len(done)}, pending: {len(pending)}, "
          f"workers={args.workers}, qps={args.qps}", flush=True)

    limiter = RateLimiter(args.qps)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(verify_one, row, idx, args, limiter, cache_dir): idx
                   for idx, row in pending}
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures), desc="Verifying", unit="q"):
            append_jsonl(all_path, fut.result())

    # ---- split pools + summary (dedup by key, last occurrence wins) ----
    verified: list[dict] = list(load_latest_rows(all_path).values())
    acc_path = args.output_dir / "accepted_verified.jsonl"
    rej_path = args.output_dir / "rejected_verified.jsonl"
    n_acc = n_rej = n_err = 0
    per_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    clue_status: dict[str, int] = defaultdict(int)
    with open(acc_path, "w", encoding="utf-8") as fa, open(rej_path, "w", encoding="utf-8") as fr:
        for row in verified:
            ver = row.get("clue_verification") or {}
            dec = ver.get("decision", "error")
            src = row.get("source", "?")
            tid = row.get("task_type_id") or "?"
            for modality in ("visual", "audio"):
                clue_status[f"{modality}:{(ver.get(modality) or {}).get('status', '?')}"] += 1
            per_source[src][dec] += 1
            per_type[tid][dec] += 1
            if dec == "accepted":
                n_acc += 1
                fa.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif dec == "rejected":
                n_rej += 1
                fr.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                n_err += 1
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vlm_model": args.vlm_model,
        "support_threshold": args.support_threshold,
        "total": len(verified),
        "accepted": n_acc, "rejected": n_rej, "error": n_err,
        "clue_status": dict(clue_status),
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "per_task_type": {k: dict(v) for k, v in per_type.items()},
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(f"\naccepted: {n_acc}  rejected: {n_rej}  error: {n_err}", flush=True)
    print(f"clue status: {dict(clue_status)}", flush=True)
    print(f"-> {acc_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
