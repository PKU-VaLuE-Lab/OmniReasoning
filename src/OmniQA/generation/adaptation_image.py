"""Generate knowledge-transfer questions and supporting figures."""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path.cwd()


def _default_caption_dirs() -> list[str]:
    """Stage-2 reasoning-beyond-video caption sources, resolved from the Omni-QA env, not baked in.

    Priority: CAPTION_RAW_DIR (comma/newline list, may contain globs) → the shared
    OMNIQA_CAPTION_STORE (`<store>/*/raw`) → the repo's data/caption. Each resolved
    dir either holds <video_id>/caption_bundle.json directly or a `raw/` subdir with
    that layout (auto-detected in find_bundles). `--caption-dir` overrides all of this.
    """
    env = os.environ.get("CAPTION_RAW_DIR", "").strip()
    if env:
        return [p.strip() for p in re.split(r"[,\n]", env) if p.strip()]
    store = os.environ.get("OMNIQA_CAPTION_STORE", "").strip()
    if store:
        return [f"{store.rstrip('/')}/*/raw"]
    return [str(_REPO_ROOT / "data" / "caption") + "/*/raw"]


DEFAULT_CAPTION_DIRS = _default_caption_dirs()

DEFAULT_CAPTION_DIR = DEFAULT_CAPTION_DIRS[0]
DEFAULT_OUTDIR = str(Path("outputs/adaptation"))


BASE = (os.environ.get("OPENAI_API_BASE")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY") or ""
if not API_KEY:
    print("WARN: no OPENAI_API_KEY set — API calls will fail.",
          file=sys.stderr)
GEN_MODEL = os.environ.get("GEN_MODEL", "deepseek-v4-flash")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gpt-image-2")

CHAT_URL = f"{BASE}/chat/completions"
IMAGE_URL = f"{BASE}/images/generations"


class RateLimiter:
    """Thread-safe sliding-window limiter: at most `rpm` acquisitions per 60s.

    Guards the shared account RPM caps (ds-v4 = 600, gpt-image-2 = 200) so we can
    run high worker concurrency without ever bursting past the per-minute ceiling.
    A small safety margin is applied by the caller (e.g. 570 / 190).
    """
    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self._lock = threading.Lock()
        self._events: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0] >= 60.0:
                    self._events.popleft()
                if len(self._events) < self.rpm:
                    self._events.append(now)
                    return
                sleep_for = 60.0 - (now - self._events[0]) + 0.01
            time.sleep(max(0.01, sleep_for))


CHAT_LIMITER = RateLimiter(int(os.environ.get("CHAT_RPM", "570")))
IMAGE_LIMITER = RateLimiter(int(os.environ.get("IMAGE_RPM", "190")))

LETTERS = "ABCDEFGHIJ"


def _post(url: str, payload: dict, timeout: int) -> dict:

    (IMAGE_LIMITER if url == IMAGE_URL else CHAT_LIMITER).acquire()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post_retry(url: str, payload: dict, timeout: int, tries: int = 4) -> dict:
    last = None
    for attempt in range(tries):
        try:
            return _post(url, payload, timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            last = f"HTTP {e.code}: {detail}"


            transient = (e.code == 429 or e.code >= 500
                         or "billing_hard_limit" in detail
                         or "rate" in detail.lower())
            if not transient:
                break
            time.sleep(min(30, 3 * (attempt + 1) ** 2))
            continue
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last or "unknown error")


def extract_json(text: str | None) -> dict | None:
    """Tolerant JSON extraction (fenced blocks, leading prose, first balanced object)."""
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
    depth, start = 0, -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


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


def strip_option_prefix(opt: str) -> str:
    return re.sub(r"^\s*[A-Ja-j]\s*[).:]\s*", "", str(opt)).strip()


def shuffle_plain_options(options: list[str], correct_letter: str,
                          seed_text: str) -> tuple[list[str], str]:
    """Deterministically shuffle plain (no-letter) option strings; remap the answer.

    Returns (new_options, new_correct_letter).  Determinism keyed on the question so a
    re-run produces the same order.
    """
    bodies = [strip_option_prefix(o) for o in options]
    n = len(bodies)
    idx = LETTERS.find(correct_letter)
    if n < 2 or idx < 0 or idx >= n:
        return bodies, correct_letter if idx >= 0 else "A"
    correct_body = bodies[idx]
    h = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16)
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = h % (i + 1)
        h //= (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    new_bodies = [bodies[perm[i]] for i in range(n)]
    new_correct = LETTERS[new_bodies.index(correct_body)]
    return new_bodies, new_correct


GEN_MAX_CHARS = int(os.environ.get("GEN_MAX_CHARS", "28000"))


def build_generation_bundle_text(bundle: dict, max_chars: int | None = None) -> str:
    """Compact JSON view fed to the generator.

    Includes summaries + per-event audio/visual captions.  av_sync_pairs is EXCLUDED
    (redundant with the summaries + audio/visual captions).

    max_chars None -> use GEN_MAX_CHARS env default; 0 or negative -> no truncation.
    """
    if max_chars is None:
        max_chars = GEN_MAX_CHARS
    subset: dict[str, Any] = {"video_id": bundle.get("video_id")}
    for k in ("adaptation_domain", "video_type", "whole_video_summary",
              "knowledge_base_summary"):
        if bundle.get(k):
            subset[k] = bundle[k]
    events = []
    for ev in bundle.get("event_captions") or []:
        events.append({
            "event_range": ev.get("event_range"),
            "event_summary": ev.get("event_summary"),
            "audio_captions": ev.get("audio_captions", []),
            "visual_captions": ev.get("visual_captions", []),

        })
    subset["event_captions"] = events
    text = json.dumps(subset, ensure_ascii=False, indent=1)
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n... (truncated)"
    return text


GENERATION_PROMPT = r"""You are an expert exam writer creating exactly ONE high-difficulty multimodal
"Adaptation reasoning" question for an educational STEM video that serves as a KNOWLEDGE BASE.

## THE CORE IDEA — DUAL DEPENDENCY (lecture-specific knowledge ⊗ a generated image)
The finished item has a REAL image that we will render from your `figure_spec`. Design the
question so the answer requires BOTH of the following, and — critically — so that the IMAGE
ALONE (plus a world-class expert's general knowledge) is NOT enough:

  • VIDEO-KNOWLEDGE FACT — a piece of knowledge that is SPECIFIC TO THIS LECTURE and that a
    general expert who did NOT watch it would NOT know or would get WRONG. It must NOT be drawn
    in the image and must NOT be restated in the question: the student recalls it from the video.

  • IMAGE SCENARIO — a brand-NEW instance the student has never seen (fresh data / chart / table
    / plot / labelled diagram / geometry). It is rendered INTO the image and must NOT appear in
    the video and must NOT be restated in the question: the student reads it off the figure.

## WHAT COUNTS AS A LECTURE-SPECIFIC FACT (this is the whole point — read carefully)
GOOD (a general expert could NOT supply this from the image alone):
  ✓ a SPECIFIC NUMERIC VALUE / parameter / constant / measurement / coefficient the instructor
    STATES or DERIVES in this lecture (e.g. "the instructor gives the decay constant as 0.037",
    "the lecture computes this system's condition number as 12.4") — not derivable from the figure.
  ✓ a NON-STANDARD NOTATION / NAMING / SCOPE the instructor introduces for this course (e.g. a
    renamed quantity, a symbol reused for something specific, "in this course 'stability' means…",
    "we count time from the first rebound, not the release") — a labelling/scope choice, NOT a
    claim that the underlying math/physics behaves differently.
  ✓ a SPECIFIC ASSUMPTION / SIMPLIFICATION / STATED APPROXIMATION the instructor imposes that
    OVERRIDES the general/default approach (e.g. "ignore the second-order term below 50°C",
    "here we use the 3-term approximation, not the exact formula", "approximate Mars' period as
    1.5× Earth's"). This must be an EXPLICITLY STATED simplifying choice, not a wrong formula.
  ✓ a SPECIFIC RESULT / THRESHOLD / ORDERING the instructor establishes for this example that a
    student must recall and transfer.

## ⚠️ FACTUAL-CORRECTNESS RULE (violating this makes the whole item WRONG — read twice)
The lecture-specific fact makes the answer NON-GENERAL only through the INPUTS to a correct
derivation — a specific value, a stated approximation, an assumption/scope, or notation. The
DERIVATION ITSELF must still obey correct, universal mathematics and physics.

You must NEVER build the "correct" answer on a claim that CONTRADICTS a universal, course-independent
truth — a formula, derivative, integral, identity, theorem, law, or sign rule that is simply true
or false regardless of any lecture. Such a claim in a transcript is almost always an INSTRUCTOR
MISSPEAK or an ASR / caption TRANSCRIPTION ERROR, not a real "convention". If you enshrine it, your
"correct" answer becomes mathematically FALSE and the real answer ends up mislabeled as a distractor.

  ✗ EXAMPLE OF THE TRAP (do NOT do this): a transcript says "the gradient of f(x)=½‖Ax−b‖² is
    2·Aᵀ(Ax−b)". The true gradient is Aᵀ(Ax−b); the "2·" is a misspeak/mis-transcription. Treating
    "2·Aᵀ(Ax−b)" as a lecture convention yields a WRONG correct-answer. → This item is INVALID.
  Rule of thumb: if a competent expert would call the lecture claim not merely "unusual" but
  OUTRIGHT WRONG, it is a factual error — do NOT use it; pick another fact or return "rejected".
  The `naive_general_answer` must be wrong because the general expert LACKS a specific input
  (value / approximation / scope), NEVER because the general expert did correct math and you did not.

FORBIDDEN (these are GENERAL KNOWLEDGE — a general expert answers from the image alone, so the
item is too easy — DO NOT use them as the video fact):
  ✗ standard theorems, formulas, identities (determinant = product of pivots, trace = sum of
    eigenvalues, |det| = volume, normal equations AᵀAx̂=Aᵀb, P=QQᵀ, dim(S+U)=dimS+dimU−dim(S∩U),
    Pythagoras, etc.)
  ✗ standard methods / algorithms / procedures taught in every textbook (Gaussian elimination,
    least squares, finding a steady state, diagonalization, ...).
  ✗ universal physical/mathematical constants.
  ✗ a "fact" that MISSTATES a universal truth (a wrong formula/derivative/identity/law/sign rule).
    This is not a convention — it is a factual error (see the FACTUAL-CORRECTNESS RULE above) and
    will be rejected downstream. Never use it as the basis of the correct answer.
If your candidate video fact is something a competent student already knows without this specific
lecture, it is FORBIDDEN — pick a different, genuinely lecture-specific fact, or return "rejected".

## THE ANTI-SHORTCUT TEST (your item MUST pass this by construction)
Imagine a world-class subject expert who sees the figure but did NOT watch the lecture. Using only
general knowledge, they will apply the STANDARD approach and reach a SPECIFIC answer — call it the
`naive_general_answer`. Your item must be built so that:
  • `naive_general_answer` is WRONG, and it IS one of the distractor options.
  • the CORRECT answer is reachable ONLY by additionally using the lecture-specific fact.
If a general expert with the figure would already get the correct answer, the item is INVALID —
redesign it (change the fact, or change how the scenario depends on it) or return "rejected".

## REQUIREMENTS (revise your own draft until all are met)
R1. GROUND THE VIDEO FACT verbatim in the caption bundle below and quote the exact line + timestamp.
R2. THE IMAGE CARRIES THE SCENARIO. `figure_spec` describes ONLY the new scenario's figure — every
    label, value, axis, row/column, point, shape EXACTLY. It must NOT contain the question, the
    answer, the video fact, or any hint. It is raw scenario data only.
R3. DO NOT LEAK THE FIGURE — CRITICAL. The question MUST be UNANSWERABLE without actually looking at
    <image 1>. It must NOT restate, describe, list, or paraphrase ANY concrete content that is DEPICTED in
    the figure (values, numbers, labels, categories, settings/states, checkbox states, coordinates, rows,
    data points). REFER to them instead ("the values in <image 1>", "the two configurations shown", "the
    highlighted region"). ✗ LEAK (forbidden): "The figure shows two configs where 'Link to Drive Mode' is
    enabled in both, and the driver switches Eco→Sport…" ✓ OK: "For the two configurations in <image 1>,
    the driver switches Eco→Sport…". Also do NOT state the video fact's value (the student recalls it).
R4. THE QUESTION MUST BEGIN WITH the exact token "<image 1>" (once, at the very start); in the body refer
    to the generated figure as "the figure", never repeating the token.
R5. SENSITIVITY. The correct answer must change if the lecture fact were different AND if the
    figure's data were different.
R6. UNIQUE ANSWER + INSTRUCTIVE DISTRACTORS. Provide AT LEAST 4 options (4 is the norm; never fewer
    than 4). Exactly one option correct; one distractor MUST be `naive_general_answer`; the others are
    specific slips (used the default value, off-by-one, used only one channel). Options are PLAIN strings
    WITHOUT "A)"/"B)" prefixes. NEVER a binary 2-option (yes/no, A-vs-B) question.
R7. NON-TRIVIAL TRANSFER — beyond reading one value and one multiply.
R8. GROUNDED DERIVATION — `derivation` is an ORDERED LIST of steps. EVERY step that uses a fact/value from
    the lecture MUST carry the exact `source_timestamp` and a verbatim `source_quote` (caption line) it is
    grounded in — REAL times/lines only, never invented. Steps that only apply those to the figure's data
    (pure computation) leave `source_timestamp`/`source_quote` as "" and state which figure data they use.

## OUTPUT — return ONLY valid JSON, no prose:
{{
  "status": "ok",
  "candidate": {{
    "video_id": "{video_id}",
    "adaptation_domain": "{adaptation_domain}",
    "qa_type": "Problem-solving Adaptation" or "Case Study Analysis",
    "question_type": "multiple-choice" or "open",
    "video_knowledge_fact": {{
      "what": "the LECTURE-SPECIFIC fact (value/convention/assumption/result), not a standard method",
      "value": "the specific value/rule as stated in the lecture",
      "why_not_general": "why a general expert who did NOT watch this lecture could NOT know this / would get it wrong",
      "source_quote": "verbatim text from the caption bundle where the instructor states it",
      "source_timestamp": "MM:SS or range"
    }},
    "image_scenario": "1-2 sentences: the NEW scenario the figure depicts",
    "figure_type": "scatter|line|bar|table|coordinate_map|geometry|number_line|flow_diagram|other",
    "figure_spec": "a single self-contained image-generation prompt for the figure ONLY. Clean minimal textbook/exam style; plain white background; large legible black sans-serif text; include EVERY label and numeric value EXACTLY with correct relative placement; no question text, no answer, no watermark, no clutter.",
    "question": "the full question. BEGINS with the token <image 1>; refer to the figure as \"the figure\" in the body. Answerable ONLY with the lecture-specific fact AND the figure. Does NOT restate the figure's data or the fact's value.",
    "options": ["plain option text 1", "plain option text 2", "..."],
    "correct_option": "A|B|... (letter of the correct option; A = first). null for open.",
    "answer": "for multiple-choice: the correct option's body text. for open: the final answer.",
    "naive_general_answer": "the WRONG answer a general expert reaches from the figure + standard knowledge WITHOUT the lecture (must equal one of the distractor options)",
    "why_general_fails": "1-2 sentences: why the figure + general knowledge alone lead to the wrong answer here",
    "derivation": [{{"step": "what this step establishes/computes", "source_timestamp": "MM:SS or range IF this step uses a fact from the video, else \"\"", "source_quote": "the verbatim caption line (spoken or on-screen) this step is grounded in, else \"\""}}],
    "why_both_needed": "1-2 sentences: what the lecture supplies and what the figure supplies",
    "distractor_rationale": {{"B": "the specific mistake this option represents", "...": "..."}}
  }}
}}

If no valid lecture-specific dual-dependency question can be made from this bundle (e.g. the lecture
only teaches standard methods with no course-specific value/convention/assumption), return:
{{ "status": "rejected", "reason": "explanation" }}

## CAPTION BUNDLE (audio_captions = SPOKEN by instructor, visual_captions = SHOWN on screen)
{caption_bundle_json}
"""


CASE_STUDY_PROMPT = r"""You are an expert exam writer creating exactly ONE high-difficulty multimodal
"Case Study Analysis" question for an educational STEM video that serves as a KNOWLEDGE BASE.

## THE CORE IDEA — DUAL DEPENDENCY (a lecture-specific ANALYTICAL FRAMEWORK ⊗ a NEW labelled figure)
The finished item has a REAL image rendered from your `figure_spec`. It shows a NEW specific case
as a LABELLED figure (a diagram / map / cross-section / schematic / annotated plot with regions,
arrows, zones, layers, or points labelled A, B, C … or 1, 2, 3 …). Design the question so the
answer requires BOTH, and — critically — so that the IMAGE ALONE (plus a world-class expert's
general knowledge) is NOT enough:

  • VIDEO-KNOWLEDGE FRAMEWORK — an ANALYTICAL FRAMEWORK SPECIFIC TO THIS LECTURE that a general
    expert who did NOT watch it would NOT know or would apply differently. Examples of the KIND of
    thing this is: a classification scheme / a model with named zones or stages / a decision or
    diagnostic criterion / an ordering or ranking rule / a mapping from features to categories that
    the instructor introduces for this material. It is NOT drawn in the image and NOT restated in the
    question: the student recalls it from the video and APPLIES it.

  • IMAGE CASE — a brand-NEW labelled instance the student has never seen. It is rendered INTO the
    image and must NOT appear in the video and must NOT be restated in the question: the student
    reads the labelled figure and decides which label the framework picks out.

The task is to APPLY the taught framework to the labelled figure to IDENTIFY / CLASSIFY a labelled
element (e.g. "which labelled region is the zone of X", "area Y is most likely ___", "which arrow
indicates ___", "classify labelled stage Z"). The answer is a LABEL or a CLASSIFICATION.

## WHAT COUNTS AS A LECTURE-SPECIFIC FRAMEWORK (this is the whole point — read carefully)
GOOD (a general expert could NOT reproduce this mapping from the labelled figure alone):
  ✓ a CLASSIFICATION / ZONAL / STAGING model the instructor defines with SPECIFIC criteria or
    boundaries (e.g. "in this model the accretion zone is where X exceeds Y", "we split the section
    into these four named bands by criterion C") — where the naive/default classification differs.
  ✓ a DECISION or DIAGNOSTIC RULE that PRIORITISES one feature over another in a way that is not the
    textbook default ("when features conflict, this course classifies by feature A, not feature B").
  ✓ a NON-STANDARD NAMING / SCOPE the instructor assigns to a category or region for this course.
  ✓ a SPECIFIC ORDERING / THRESHOLD / correspondence the lecture establishes that a student must
    recall and transfer to a new labelled case.

## ⚠️ FACTUAL-CORRECTNESS RULE (violating this makes the whole item WRONG — read twice)
The framework makes the answer NON-GENERAL only by supplying a lecture-specific CRITERION / MAPPING /
NAMING that a general expert lacks. Applying it to the figure must still be internally correct.

You must NEVER build the "correct" answer on a claim that CONTRADICTS a universal, course-independent
truth — a mislabelled universally-defined structure, a reversed direction of a well-established
process, a misstated law or definition that holds regardless of any lecture. Such a claim in a
transcript is almost always an INSTRUCTOR MISSPEAK or an ASR / caption TRANSCRIPTION ERROR, not a
real framework. If a competent expert would call the lecture claim not merely "a specific convention"
but OUTRIGHT WRONG, it is a factual error — do NOT use it; pick another framework or return "rejected".
The `naive_general_answer` must be wrong because the general expert applies the DEFAULT/most-common
classification (lacking the lecture's specific criterion), NEVER because the general expert reasoned
correctly and you did not.

  ✗ EXAMPLE OF THE OVER-GENERALISATION TRAP (do NOT do this): while working ONE example a transcript
    says "the u-test here is a left-tailed test". Inflating this into a course-wide framework — "the
    u-test is ALWAYS left-tailed regardless of the alternative hypothesis" — is a FALSE
    over-generalisation: the tail of a u/z-test is set by the alternative hypothesis, so the inflated
    rule is simply wrong. → INVALID.
  A legitimate framework is EITHER (a) a pure LABELLING / NAMING / SCOPE choice, OR (b) a TRUE but
  non-obvious domain fact the lecture emphasises (e.g. "for a REPULSIVE force the scattering focus is
  the EXTERIOR focus" — rarely stated, but actually correct). It is NEVER a one-off, context-specific
  statement inflated with "always / regardless / in all cases" into a universal rule that is false in
  general. Do NOT add "always" / "regardless" unless the lecture genuinely establishes a course-wide rule.

FORBIDDEN (these are GENERAL KNOWLEDGE — a general expert classifies correctly from the image alone,
so the item is too easy — DO NOT use them as the framework):
  ✗ universally-taught taxonomies / anatomy / standard part-labels that any expert knows on sight
    (e.g. "which arrow points to the nucleus", "label the mitochondrion") — unless the lecture
    imposes a genuinely non-standard criterion that changes the answer.
  ✗ standard models every textbook teaches identically, applied in the standard way.
If your candidate framework is something a competent student already applies without this specific
lecture, it is FORBIDDEN — pick a genuinely lecture-specific framework, or return "rejected".

## THE ANTI-SHORTCUT TEST (your item MUST pass this by construction)
Imagine a world-class subject expert who sees the labelled figure but did NOT watch the lecture.
Using only general knowledge, they will apply the DEFAULT classification and pick a SPECIFIC label —
the `naive_general_answer`. Your item must be built so that:
  • `naive_general_answer` is WRONG, and it IS one of the distractor options.
  • the CORRECT label is reachable ONLY by additionally using the lecture-specific framework.
If a general expert with the figure would already pick the correct label, the item is INVALID —
redesign it (change the framework, or how the case depends on it) or return "rejected".

## REQUIREMENTS (revise your own draft until all are met)
R1. GROUND THE FRAMEWORK verbatim in the caption bundle below and quote the exact line + timestamp.
R2. THE IMAGE CARRIES THE LABELLED CASE. `figure_spec` describes ONLY the new labelled figure — every
    region/arrow/zone/point and its LABEL (A, B, C … or 1, 2, 3 …) EXACTLY, with correct relative
    placement. It must NOT contain the question, the framework, the answer, or any hint. The options
    must refer to labels/elements that actually appear in `figure_spec`.
R3. DO NOT LEAK THE FIGURE — CRITICAL. The question MUST be UNANSWERABLE without looking at <image 1>.
    It must NOT restate, describe, or enumerate what the labelled figure depicts (which label is what, the
    concrete data/values/regions) — REFER to them instead ("the labelled elements in <image 1>", "region
    A shown"). It must also NOT restate the framework's criteria (the student recalls them from the video).
R4. THE QUESTION MUST BEGIN WITH the exact token "<image 1>" (once, at the very start); in the body refer
    to the generated figure as "the figure", never repeating the token.
R5. SENSITIVITY. The correct label must change if the framework's criterion were different AND if the
    figure's labelled arrangement were different.
R6. UNIQUE ANSWER + INSTRUCTIVE DISTRACTORS. Provide AT LEAST 4 options (4 is the norm; never fewer than
    4, never a binary 2-option question). Exactly one option correct; one distractor MUST be
    `naive_general_answer` (the default-classification label); others are specific misapplications.
    Options are PLAIN strings WITHOUT "A)"/"B)" prefixes (they may name a label, e.g. "Region C", or
    a category, e.g. "in situ accretion zone").
R7. NON-TRIVIAL TRANSFER — the student must genuinely apply the criterion, not read a printed label.
R8. GROUNDED DERIVATION — `derivation` is an ORDERED LIST of steps. EVERY step that uses the lecture's
    framework/criterion MUST carry the exact `source_timestamp` and a verbatim `source_quote` (caption
    line) it is grounded in — REAL times/lines only, never invented. Steps that only apply it to the
    figure's labelled elements leave `source_timestamp`/`source_quote` as "" and name which elements.

## OUTPUT — return ONLY valid JSON, no prose:
{{
  "status": "ok",
  "candidate": {{
    "video_id": "{video_id}",
    "adaptation_domain": "{adaptation_domain}",
    "qa_type": "Case Study Analysis",
    "question_type": "multiple-choice",
    "video_knowledge_fact": {{
      "what": "the LECTURE-SPECIFIC analytical framework / classification scheme / criterion / model",
      "value": "the specific criterion / mapping / boundary / naming as the lecture states it",
      "why_not_general": "why a general expert who did NOT watch this lecture would apply a DIFFERENT (default) classification and get it wrong",
      "source_quote": "verbatim text from the caption bundle where the instructor states the framework",
      "source_timestamp": "MM:SS or range"
    }},
    "image_scenario": "1-2 sentences: the NEW labelled case the figure depicts and what its labels are",
    "figure_type": "labeled_diagram|coordinate_map|cross_section|flow_diagram|schematic|zonal_map|annotated_plot|geometry|other",
    "figure_spec": "a single self-contained image-generation prompt for the labelled figure ONLY. Clean minimal textbook/exam style; plain white background; large legible black sans-serif LABELS (A, B, C … or 1, 2, 3 …) placed EXACTLY on the correct regions/arrows/points; include every element with correct relative placement; no question text, no answer, no framework, no watermark, no clutter.",
    "question": "the full question. BEGINS with the token <image 1>; refer to the figure as \"the figure\" in the body. Answerable ONLY by applying the lecture-specific framework to the labelled figure. Does NOT restate the framework's criteria.",
    "options": ["plain option text 1", "plain option text 2", "..."],
    "correct_option": "A|B|... (letter of the correct option; A = first).",
    "answer": "the correct option's body text.",
    "naive_general_answer": "the WRONG label/category a general expert picks by applying the DEFAULT classification WITHOUT the lecture framework (must equal one of the distractor options)",
    "why_general_fails": "1-2 sentences: why the default classification leads to the wrong label here",
    "derivation": [{{"step": "what this step establishes/applies", "source_timestamp": "MM:SS or range IF this step uses a fact from the video, else \"\"", "source_quote": "the verbatim caption line this step is grounded in, else \"\""}}],
    "why_both_needed": "1-2 sentences: what the lecture supplies (the criterion) and what the figure supplies (the new labelled case)",
    "distractor_rationale": {{"B": "the specific misapplication this option represents", "...": "..."}}
  }}
}}

If no valid lecture-specific Case Study Analysis question can be made from this bundle (e.g. the
lecture only teaches standard, universally-known classifications), return:
{{ "status": "rejected", "reason": "explanation" }}

## CAPTION BUNDLE (audio_captions = SPOKEN by instructor, visual_captions = SHOWN on screen)
{caption_bundle_json}
"""

REGEN_FEEDBACK = {
    "factual_error": (
        "\n\nREVISION NEEDED: your previous 'correct' answer was FACTUALLY WRONG. The lecture fact you "
        "used contradicts a universal, course-independent truth (a formula / derivative / identity / "
        "law / sign rule) — almost certainly an instructor misspeak or an ASR/caption transcription "
        "error, NOT a real convention. The answer you labelled 'naive' was actually the correct one. "
        "Do NOT build the answer on a false universal claim. Either choose a DIFFERENT lecture fact "
        "whose non-generality comes only from a specific value / stated approximation / assumption / "
        "scope / notation (so the derivation still uses correct math), or return 'rejected'."),
    "factual_uncertain": (
        "\n\nREVISION NEEDED: the factual auditor could not confirm your correct answer. Make the "
        "derivation explicit and self-contained, ensure every step uses correct universal math, and "
        "state the lecture fact as a concrete value / approximation / scope (not a re-derived formula)."),
    "shortcut": (
        "\n\nREVISION NEEDED: your previous draft was answerable from the FIGURE + general knowledge "
        "ALONE, without the lecture (a general expert recovered the correct answer). The video fact "
        "you chose was standard textbook knowledge. Choose a genuinely LECTURE-SPECIFIC fact (a "
        "specific value/convention/assumption/result the instructor states that a general expert "
        "would NOT know or would get wrong), and rebuild the scenario so the naive general-knowledge "
        "answer is WRONG (and is a distractor)."),
    "ungrounded": (
        "\n\nREVISION NEEDED: even WITH the lecture-specific fact, the stated correct answer could not "
        "be recovered from the figure. Re-ground the fact in an exact caption quote, re-check your "
        "derivation, and make the answer uniquely determined by (fact ⊗ figure)."),
    "structure": (
        "\n\nREVISION NEEDED: the previous draft was missing required fields. Return all of: question "
        "(with <image 1>), options, correct_option, answer, video_knowledge_fact, figure_spec, "
        "naive_general_answer."),
    "image_token_in_text_only": (
        "\n\nREVISION NEEDED: this is a TEXT-ONLY item but your question referenced an image "
        "(<image ...>). Remove any image reference and describe the new scenario entirely in words; "
        "leave figure_spec as \"\"."),
    "few_options": (
        "\n\nREVISION NEEDED: your item had fewer than 4 options (never use a binary A/B or yes/no "
        "question). Return AT LEAST 4 plausible options — one correct, one equal to naive_general_answer, "
        "and the rest instructive distractors. For comparative items, present 4+ candidate cases or "
        "(choice, reason) pairs."),
}


SOLVE_PROMPT = r"""Answer this multiple-choice question.

{context_block}

Question: {question}

Options:
{options_text}

Reason carefully, then pick the single best option. Return ONLY valid JSON:
{{"reasoning": "brief", "selected_option": "A"}}
"""

CTX_IMAGE_GENERAL = (
    "You can see the following FIGURE (this is exactly what the question's image shows):\n"
    "{figure_spec}\n\n"
    "You are a world-class subject-matter expert, BUT you did NOT watch the lecture the question "
    "refers to. Use ONLY this figure plus your general expert knowledge and reasoning.")

CTX_IMAGE_PLUS_LECTURE = (
    "You can see the following FIGURE (exactly what the question's image shows):\n{figure_spec}\n\n"
    "You ALSO watched the lecture, which established this specific fact:\n"
    "  {fact_what}: {fact_value}\n\n"
    "Use the figure together with this lecture fact.")


CTX_TEXT_GENERAL = (
    "You are a world-class subject-matter expert, BUT you did NOT watch the video the question refers "
    "to. Use ONLY the question and your general expert knowledge and reasoning.")

CTX_TEXT_PLUS_VIDEO = (
    "You did NOT see the video, but it established this specific piece of knowledge:\n"
    "  {fact_what}: {fact_value}\n\n"
    "Use this together with the question.")


FACT_CHECK_PROMPT = r"""You are a meticulous {domain} professor auditing an exam item for FACTUAL CORRECTNESS.
An automated pipeline builds "adaptation" questions whose correct answer depends on a lecture-specific
fact. The pipeline has a known failure mode: it sometimes mistakes an instructor MISSPEAK or an
ASR/caption TRANSCRIPTION ERROR for a legitimate "convention", and then labels a mathematically WRONG
answer as correct. Your job is to catch exactly that.

FIGURE (what the image shows):
{figure_spec}

QUESTION:
{question}

OPTIONS:
{options_text}

The pipeline claims the CORRECT answer is: {gold_letter}) {gold_text}
It claims a general expert (who did NOT watch the lecture) would wrongly pick: {naive}

The "lecture-specific fact" the correct answer relies on:
  what:  {fact_what}
  value: {fact_value}
  source quote (from transcript/captions): {fact_quote}

The pipeline's derivation:
{derivation}

## HOW TO JUDGE
A fact may legitimately make the answer non-general ONLY through the INPUTS to a correct derivation:
  • a SPECIFIC NUMERIC VALUE / parameter the lecture supplies,
  • an EXPLICITLY STATED approximation or simplifying assumption ("approximate Mars' period as 1.5×"),
  • a SCOPE / DEFINITION of what is being asked or measured ("count time from the first rebound"),
  • a NOTATION / NAMING choice.
Under any of these, the derivation still uses correct, universal mathematics/physics, and the answer
is CORRECT (possibly "correct under the stated approximation"). That is FINE — mark it correct.

It is a FACTUAL ERROR (mark it wrong) when the correct answer depends on a claim that CONTRADICTS a
universal, course-independent truth — a formula, derivative, integral, identity, theorem, law, or
sign rule that is simply true or false regardless of any lecture. Example: claiming the gradient of
½‖Ax−b‖² is 2·Aᵀ(Ax−b) (the truth is Aᵀ(Ax−b)); claiming ∫x dx = x²; a wrong sign in a conservation
law. These are misspeaks/mis-transcriptions, not conventions. If the "correct" answer is only correct
under such a false claim — and standard correct math gives a DIFFERENT answer (often the one labelled
"naive") — then the item is factually broken.

Work the problem yourself from the figure using correct {domain}. Then decide.

Return ONLY valid JSON:
{{"independent_answer": "the answer correct math/physics gives (with units if any)",
  "fact_kind": "specific_value | stated_approximation | assumption_or_scope | notation | false_universal_claim | other",
  "claimed_answer_is_correct": true or false,
  "reason": "one or two sentences; if false, name the universal truth that is violated"}}
"""


FACT_CHECK_PROMPT_CASE = r"""You are a meticulous {domain} professor auditing an exam item for FACTUAL CORRECTNESS.
An automated pipeline builds "case study analysis" questions where the correct answer is a LABEL or
CLASSIFICATION obtained by applying a lecture-specific ANALYTICAL FRAMEWORK (a classification scheme /
model / decision or diagnostic criterion) to a NEW labelled figure. The pipeline has a known failure
mode: it sometimes mistakes an instructor MISSPEAK or an ASR/caption TRANSCRIPTION ERROR for a
legitimate "framework", and then labels a WRONG classification as correct. Your job is to catch that.

FIGURE (what the labelled image shows):
{figure_spec}

QUESTION:
{question}

OPTIONS:
{options_text}

The pipeline claims the CORRECT answer is: {gold_letter}) {gold_text}
It claims a general expert (who did NOT watch the lecture) would wrongly pick: {naive}

The "lecture-specific framework" the correct answer relies on:
  what:  {fact_what}
  value: {fact_value}
  source quote (from transcript/captions): {fact_quote}

The pipeline's derivation:
{derivation}

## HOW TO JUDGE
A framework may legitimately make the answer non-general ONLY by supplying a lecture-specific
CRITERION / MAPPING / NAMING / ORDERING that a general expert would not have:
  • a classification scheme or zonal/staging model with specific criteria or boundaries,
  • a decision/diagnostic rule that prioritises features in a non-default way,
  • a non-standard naming or scope for a category/region,
  • a specific ordering / threshold / correspondence the lecture establishes.
Under any of these, applying the framework to the figure is internally correct and the claimed label
is right GIVEN the framework. That is FINE — mark it correct.

It is a FACTUAL ERROR (mark it wrong) when the correct answer depends on a claim that CONTRADICTS a
universal, course-independent truth — e.g. mislabelling a universally-defined structure, reversing a
well-established direction/order that holds regardless of any lecture, or a definition/law stated
backwards. These are misspeaks/mis-transcriptions, not frameworks. ALSO mark it wrong if, even taking
the stated framework at face value, the claimed label is NOT the correct application to this figure
(the derivation is inconsistent), or if the labelled figure does not actually determine a unique
answer.

CRITICAL — THE OVER-GENERALISATION TEST. Decide whether the stated framework, READ AS A GENERAL RULE,
is TRUE or FALSE in {domain}. A legitimate framework is EITHER (a) a pure labelling / naming / scope
choice, OR (b) a TRUE but non-obvious domain fact the lecture emphasises. If instead the "framework"
is a claim that is FALSE as a general rule — typically a context-specific statement (true only for one
example / one hypothesis / one configuration) that has been INFLATED into a universal rule with
"always / regardless / in all cases" — then it is a false_universal_claim, even when phrased as "the
lecture's convention". Example: "the u-test is always left-tailed regardless of the alternative
hypothesis" is FALSE (the tail depends on the alternative hypothesis) → false_universal_claim;
"for a repulsive force the scattering focus is the exterior focus" is TRUE → accept. When unsure
whether a rule is a genuine course-wide definition or an inflated one-off, treat it as
false_universal_claim and reject.

Reason about the figure and the stated framework using correct {domain}. Then decide.

Return ONLY valid JSON:
{{"independent_answer": "the label/classification correct application yields (name the label)",
  "fact_kind": "classification_criterion | model_or_scheme | diagnostic_rule | naming_or_scope | false_universal_claim | other",
  "claimed_answer_is_correct": true or false,
  "reason": "one or two sentences; if false, name the universal truth violated or the inconsistency"}}
"""


GEN_PROMPT_GENERAL = r"""You are an expert exam writer creating exactly ONE high-difficulty
"reasoning-beyond-video" question for a STEM video that serves as a KNOWLEDGE SOURCE.

The video may be ANY genre of STEM content — a lecture, a lab experiment, a hands-on demonstration,
a procedure/how-to, a software/screen recording, a simulation, or field/observational footage.
This one is: {video_type}.

## QUESTION FAMILY FOR THIS ITEM
{form_directive}

## THE CORE IDEA — the answer must require reasoning BEYOND what the video literally shows
Design the question so a person who did NOT watch this video CANNOT answer it, even with world-class
general expertise. The answer must depend on VIDEO-DERIVED KNOWLEDGE — something THIS specific video
establishes (a value, a definition/convention, a demonstrated procedure step, an observed result, a
criterion/threshold) that a non-viewer would not know or would get WRONG — applied to a NEW situation
the question poses. It must NOT be answerable by recalling a single fact ("what number appeared"); it
must require genuine TRANSFER / prediction / counterfactual / diagnosis / design reasoning.

## WHAT COUNTS AS VIDEO-DERIVED KNOWLEDGE (a non-viewer could NOT supply this)
GOOD:
  ✓ a SPECIFIC VALUE / parameter / constant / measurement the video states, shows on screen, or
    measures (not derivable from general knowledge alone).
  ✓ a NON-STANDARD CONVENTION / NAMING / SCOPE the video introduces ("in this setup X means…").
  ✓ a SPECIFIC ASSUMPTION / SIMPLIFICATION / APPROXIMATION the video explicitly imposes that overrides
    the default approach (a stated choice, NOT a wrong formula).
  ✓ a DEMONSTRATED PROCEDURE / TECHNIQUE / SETTING / SEQUENCE the video performs (the specific order of
    steps, a tool setting, the measured outcome of an action) that a non-viewer could not guess.
  ✓ an OBSERVED RESULT / behaviour / threshold the video establishes for this specific system.

## ⚠️ FACTUAL-CORRECTNESS RULE (violating this makes the whole item WRONG — read twice)
The video-derived knowledge makes the answer NON-GENERAL only through the INPUTS to a correct
derivation — a specific value, a stated approximation, an assumption/scope, a convention, a
demonstrated setting. The DERIVATION / REASONING ITSELF must still obey correct, universal
mathematics, physics, and engineering. NEVER build the "correct" answer on a claim that CONTRADICTS a
universal, domain-independent truth (a formula, derivative, identity, theorem, law, sign rule). Such a
claim in a transcript/caption is almost always a MISSPEAK or an ASR/caption TRANSCRIPTION ERROR, not a
real convention; if you enshrine it, the "correct" answer becomes WRONG. Rule of thumb: if a competent
expert would call the claim not merely "unusual" but OUTRIGHT WRONG, do NOT use it — pick another fact
or return "rejected". The `naive_general_answer` must be wrong because the non-viewer LACKS a specific
video input, NEVER because the non-viewer did correct reasoning and you did not.

FORBIDDEN as the video knowledge (these are GENERAL KNOWLEDGE — a non-viewer already knows them):
  ✗ standard theorems / formulas / identities / methods / algorithms taught in every textbook.
  ✗ universal physical/mathematical constants.
  ✗ a "fact" that MISSTATES a universal truth (a wrong formula/law/sign) — that is a factual error.
If your candidate knowledge is something a competent student already knows without THIS video, it is
FORBIDDEN — pick genuinely video-specific knowledge, or return "rejected".

## THE ANTI-SHORTCUT TEST (your item MUST pass this by construction)
Imagine a world-class expert who did NOT watch the video. Using only general knowledge (and the image
if one is present), they apply the STANDARD approach and reach a SPECIFIC answer — the
`naive_general_answer`. Your item must be built so that:
  • `naive_general_answer` is WRONG, and it IS one of the distractor options.
  • the CORRECT answer is reachable ONLY by additionally using the video-derived knowledge.
If a non-viewer would already get the correct answer, the item is INVALID — redesign it or reject.

## REQUIREMENTS (revise your own draft until all are met)
R1. GROUND THE VIDEO KNOWLEDGE in the caption bundle: give `source_timestamp`, and EITHER a verbatim
    `source_quote` (spoken / on-screen text / a stated measurement) OR a concrete `source_observation`
    (demonstrated / observed / a procedure step). Exactly ONE of the two is non-empty, matching
    `source_modality`.
R2. {requires_image_line}
R3. DO NOT LEAK — CRITICAL. (a) The question must NOT restate the video-derived knowledge's value (the
    solver recalls it from the video). (b) IMAGE ITEMS: the question MUST be UNANSWERABLE without actually
    looking at the figure — it must NOT restate, describe, list, or paraphrase ANY concrete content DEPICTED
    in the figure (values, numbers, labels, categories, settings/checkbox states, coordinates, rows, data
    points). REFER to them instead ("the values in the figure", "the two configurations shown", "the
    highlighted region"). ✗ LEAK (forbidden): "…two configs where 'Link to Drive Mode' is enabled in both,
    then the driver switches Eco→Sport…" — this hands the solver the figure's content.
    ✓ OK: "<image 1> For the two dashboard configurations shown, the driver switches Eco→Sport…".
R4. {image_token_rule}
R5. SENSITIVITY. The correct answer must change if the video-derived knowledge were different (and, for
    image items, if the figure's data were different).
R6. UNIQUE ANSWER + INSTRUCTIVE DISTRACTORS. Provide AT LEAST 4 options (4 is the norm; never fewer than
    4, never a binary 2-option / A-vs-B / yes-no question). Exactly one option correct; one distractor
    MUST be `naive_general_answer`; the others are specific slips. Options are PLAIN strings WITHOUT
    "A)"/"B)" prefixes.
R7. MULTI-STEP REASONING — the correct answer must require AT LEAST TWO chained inferential steps that
    combine the video-derived knowledge with the new scenario (e.g. recall a criterion/value, then apply
    it, then compute/compare/decide). FORBIDDEN: single-lookup or one-multiply items answerable by reading
    one number and doing one trivial operation.
R8. KNOWLEDGE-INTENSIVE — `video_derived_knowledge` must be a genuine DOMAIN KNOWLEDGE POINT (a concept,
    law, criterion, method, parameter, or procedure the video establishes), NOT a trivial on-screen
    readout ("the label says X") that carries no transferable reasoning.
R9. GROUNDED DERIVATION — `derivation` is an ORDERED LIST of steps. EVERY step that uses a fact / value /
    criterion / procedure from the video MUST carry the exact `source_timestamp` and a verbatim
    `source_quote` (spoken or on-screen caption line) it is grounded in — use REAL times/lines from the
    caption bundle only, never invented. Steps that only apply those facts to the new scenario's data
    (pure computation) leave `source_timestamp`/`source_quote` as "" and state which scenario data they use.

## OUTPUT — return ONLY valid JSON, no prose:
{{
  "status": "ok",
  "candidate": {{
    "video_id": "{video_id}",
    "adaptation_domain": "{adaptation_domain}",
    "video_type": "{video_type}",
    "qa_type": "{qa_type_label}",
    "question_type": "multiple-choice",
    "requires_image": {requires_image_json},
    "video_derived_knowledge": {{
      "what": "the VIDEO-SPECIFIC knowledge a non-viewer would lack or get wrong",
      "value": "the specific value/rule/criterion/procedure as THIS video establishes it",
      "source_modality": "spoken | on_screen_text | visual_demonstration | observed_event | procedure_step | measured_result",
      "why_not_general": "why someone who did NOT watch this video could not know it / would get it wrong",
      "source_quote": "verbatim spoken/on-screen/measured text — non-empty IFF modality is spoken/on_screen_text/measured_result, else \"\"",
      "source_observation": "concrete description of what is shown/done — non-empty IFF modality is visual_demonstration/observed_event/procedure_step, else \"\"",
      "source_timestamp": "MM:SS or range"
    }},
    "scenario": "1-2 sentences: the NEW situation the question poses (new data / setup / change / fault / goal)",
    "figure_type": "scatter|line|bar|table|coordinate_map|geometry|labeled_diagram|schematic|flow_diagram|none",
    "figure_spec": "image-generation prompt for the figure ONLY (image items): clean minimal textbook/exam style; plain white background; large legible black sans-serif text; include EVERY label and value EXACTLY; no question text, no answer, no hint. Use \"\" for text-only items.",
    "question": "the full question. IMAGE items BEGIN with the token <image 1> and refer to it as \"the figure\" in the body; TEXT-ONLY items contain no image token. Answerable ONLY with the video-derived knowledge (and the figure for image items). Does NOT restate the knowledge value or the figure's data.",
    "options": ["plain option text 1", "plain option text 2", "..."],
    "correct_option": "A|B|... (letter of the correct option; A = first)",
    "answer": "the correct option's body text",
    "naive_general_answer": "the WRONG answer a non-viewer reaches without the video (must equal one of the distractor options)",
    "why_general_fails": "1-2 sentences: why general knowledge alone leads to the wrong answer here",
    "derivation": [{{"step": "what this step establishes/computes", "source_timestamp": "MM:SS or range IF this step uses a fact from the video, else \"\"", "source_quote": "the verbatim caption line (spoken or on-screen) this step is grounded in, else \"\""}}],
    "why_both_needed": "1-2 sentences: what the video supplies and what the scenario/figure supplies",
    "distractor_rationale": {{"B": "the specific mistake this option represents", "...": "..."}}
  }}
}}

If no valid reasoning-beyond-video question can be made from this bundle — e.g. the video is NOT STEM,
has no video-specific knowledge (only standard textbook content), or the captions are too sparse —
return:
{{ "status": "rejected", "reason": "explanation" }}

## CAPTION BUNDLE (audio_captions = what is HEARD: speech/narration/sound; visual_captions = what is SHOWN on screen: demonstration/on-screen text/measurements)
{caption_bundle_json}
"""


TRANSFER_DIRECTIVE = (
    "CROSS-SCENARIO TRANSFER (image): apply the video-derived knowledge (a value/convention/procedure) to "
    "a BRAND-NEW instance rendered as the figure (fresh data/figure the student has never seen). Reading "
    "the figure alone, with general knowledge, must be INSUFFICIENT to get the correct answer.")
DESIGN_DIRECTIVE = (
    "DESIGN / OPTIMIZATION (image): given the principles/criteria/parameters the video established, ask the "
    "solver to CHOOSE parameters or a design for a NEW goal shown in the figure. The optimal/correct choice "
    "must depend on the video's specific criteria, not on generic best practice.")
MULTISTEP_QUANT_DIRECTIVE = (
    "MULTI-STEP QUANTITATIVE REASONING (image): build a question that requires CHAINING at least two "
    "video-derived quantities/relations (e.g. a stated parameter AND a stated approximation/criterion) and "
    "applying them to fresh data in the figure to COMPUTE a result. A single lookup + one multiply must be "
    "insufficient; the derivation must have ≥2 dependent steps.")
COMPARATIVE_DIRECTIVE = (
    "COMPARATIVE REASONING (image): the figure presents SEVERAL (prefer 4+) new cases/configurations/"
    "curves. Ask which one is largest / best / valid / satisfies the criterion, where the answer flips "
    "depending on the video-established criterion or value. The OPTIONS must be the candidate cases "
    "themselves (≥4, e.g. Configuration A/B/C/D) or (choice, reason) pairs — NEVER a binary two-option "
    "A-vs-B question. A non-viewer applying the default criterion must pick the wrong one.")
ERROR_CRITIQUE_DIRECTIVE = (
    "ERROR CRITIQUE (text-only): state a plausible-looking proposed solution, claim, or worked step for a NEW "
    "case, and ask the solver to identify the SPECIFIC flaw (or whether it is correct). The flaw must only be "
    "detectable using the video-derived criterion/method/value; a non-viewer would accept the proposed solution.")
CROSS_DOMAIN_DIRECTIVE = (
    "CROSS-DOMAIN TRANSFER (text-only): apply the principle/method/criterion the video establishes to a "
    "scenario in a DIFFERENT domain or context than the one shown. The mapping must require understanding the "
    "underlying principle; surface-level topic knowledge of the new domain must be insufficient.")
PROCEDURE_ORDERING_DIRECTIVE = (
    "PROCEDURE / PLANNING (text-only, ADAPTATION): the video DEMONSTRATES a procedure/technique/workflow. "
    "Pose a NEW task/goal/object/constraint that is NOT shown in the video, and ask for the correct next "
    "step, the correct ORDERING of steps, or the right setting/tool to achieve IT. The solver must TRANSFER "
    "the demonstrated procedure to this new task — do NOT ask what the video itself does next or to recall "
    "the video's own sequence. The correct plan must depend on the video's specific demonstrated procedure "
    "(a non-viewer would order/choose differently); generic common sense must be insufficient.")


FORMS: dict[str, dict] = {

    "problem_solving": {"qa_type": "Problem-solving Adaptation", "requires_image": True,
                        "prompt_template": GENERATION_PROMPT, "verifier_profile": "numeric",
                        "cot_mode": "image", "form_directive": ""},
    "case_study":      {"qa_type": "Case Study Analysis", "requires_image": True,
                        "prompt_template": CASE_STUDY_PROMPT, "verifier_profile": "framework",
                        "cot_mode": "image", "form_directive": ""},

    "cross_transfer":  {"qa_type": "Cross-Scenario Transfer", "requires_image": True,
                        "prompt_template": GEN_PROMPT_GENERAL, "verifier_profile": "general",
                        "cot_mode": "image", "form_directive": TRANSFER_DIRECTIVE},
    "design_optimize": {"qa_type": "Design / Optimization", "requires_image": True,
                        "prompt_template": GEN_PROMPT_GENERAL, "verifier_profile": "general",
                        "cot_mode": "image", "form_directive": DESIGN_DIRECTIVE},

    "multistep_quant": {"qa_type": "Multi-step Quantitative Reasoning", "requires_image": True,
                        "prompt_template": GEN_PROMPT_GENERAL, "verifier_profile": "numeric",
                        "cot_mode": "image", "form_directive": MULTISTEP_QUANT_DIRECTIVE},
    "comparative":     {"qa_type": "Comparative Reasoning", "requires_image": True,
                        "prompt_template": GEN_PROMPT_GENERAL, "verifier_profile": "general",
                        "cot_mode": "image", "form_directive": COMPARATIVE_DIRECTIVE},
    "error_critique":  {"qa_type": "Error Critique", "requires_image": False,
                        "prompt_template": GEN_PROMPT_GENERAL, "verifier_profile": "general",
                        "cot_mode": "text", "form_directive": ERROR_CRITIQUE_DIRECTIVE},
    "procedure_ordering":{"qa_type": "Procedure / Planning", "requires_image": False,
                        "prompt_template": GEN_PROMPT_GENERAL, "verifier_profile": "general",
                        "cot_mode": "text", "form_directive": PROCEDURE_ORDERING_DIRECTIVE},
    "cross_domain_transfer":{"qa_type": "Cross-Domain Transfer", "requires_image": False,
                        "prompt_template": GEN_PROMPT_GENERAL, "verifier_profile": "general",
                        "cot_mode": "text", "form_directive": CROSS_DOMAIN_DIRECTIVE},
}
DEFAULT_FORMS = ["problem_solving", "case_study"]


def form_spec(form_key: str) -> dict:
    return FORMS.get(form_key) or FORMS["problem_solving"]


def video_form_sequence(video_id: str, forms: list[str]) -> str:
    """Deterministically assign ONE form to a video by hashing its id (resume-stable:
    a re-run always picks the same form, so an already-ok video is never re-formed)."""
    forms = forms or DEFAULT_FORMS
    h = int(hashlib.sha256((video_id or "").encode("utf-8")).hexdigest(), 16)
    return forms[h % len(forms)]


def _video_knowledge(cand: dict) -> dict:
    """Read the video-derived knowledge, tolerating both the new key and the legacy one."""
    return cand.get("video_derived_knowledge") or cand.get("video_knowledge_fact") or {}


def _derivation_text(cand: dict) -> str:
    """Render the derivation (new grounded list-of-steps, or legacy string) as readable text."""
    d = cand.get("derivation")
    if isinstance(d, list):
        lines = []
        for i, s in enumerate(d, 1):
            if not isinstance(s, dict):
                lines.append(f"{i}. {s}"); continue
            ts = s.get("source_timestamp") or ""
            q = s.get("source_quote") or ""
            g = f"  [@{ts} \"{q}\"]" if (ts or q) else ""
            lines.append(f"{i}. {s.get('step','')}{g}")
        return "\n".join(lines)
    return str(d or "")


def _count_by(results: list[dict], keyfn, only_ok: bool = False) -> dict:
    """Tally records by keyfn (optionally only status=='ok'); returns an ordered dict."""
    counts: dict[str, int] = {}
    for r in results:
        if only_ok and r.get("status") != "ok":
            continue
        k = str(keyfn(r))
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def generate_candidate(bundle: dict, feedback: str = "", timeout: int = 300,
                       form_key: str = "problem_solving") -> dict | None:
    spec = form_spec(form_key)
    template = spec["prompt_template"]
    requires_image = spec["requires_image"]
    if template is GEN_PROMPT_GENERAL:
        if requires_image:
            requires_image_line = (
                "THE IMAGE CARRIES THE NEW SCENARIO. `figure_spec` describes ONLY the new figure — "
                "every label/value/axis EXACTLY. It must NOT contain the question, the answer, or the "
                "video-derived knowledge.")
            image_token_rule = (
                'THE QUESTION MUST BEGIN WITH the exact token "<image 1>" (exactly once, at the very '
                'start), followed by the question text. Do NOT place "<image 1>" anywhere else; in the '
                'body, refer to the generated figure as "the figure" (e.g. "the two configurations in '
                'the figure").')
        else:
            requires_image_line = (
                "THIS IS A TEXT-ONLY ITEM. Do NOT invent a figure; leave `figure_spec` as \"\". "
                "Describe the new scenario in words inside the question.")
            image_token_rule = (
                'THE QUESTION MUST NOT CONTAIN "<image 1>" or refer to any image; it refers to the '
                "video only.")
        prompt = template.format(
            caption_bundle_json=build_generation_bundle_text(bundle),
            video_id=bundle.get("video_id", "unknown"),
            adaptation_domain=bundle.get("adaptation_domain", bundle.get("video_type", "general")),
            video_type=bundle.get("video_type", "STEM video"),
            qa_type_label=spec["qa_type"],
            form_directive=spec["form_directive"],
            requires_image_line=requires_image_line,
            image_token_rule=image_token_rule,
            requires_image_json=("true" if requires_image else "false"),
        )
    else:
        prompt = template.format(
            caption_bundle_json=build_generation_bundle_text(bundle),
            video_id=bundle.get("video_id", "unknown"),
            adaptation_domain=bundle.get("adaptation_domain", bundle.get("video_type", "general")),
        )
    if feedback:
        prompt += feedback
    payload = {
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,


        "max_tokens": 16000,
    }
    data = _post_retry(CHAT_URL, payload, timeout=timeout)
    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
    content = msg.get("content") or msg.get("reasoning_content") or ""
    result = extract_json(content)
    if not result:
        return None
    if result.get("status") == "rejected":
        return result
    cand = result.get("candidate") or result
    cand.setdefault("requires_image", requires_image)
    q = cand.get("question", "")
    if not q:
        return None
    if requires_image:


        body = re.sub(r"\s*<image[^>]*>\s*", " ", q, flags=re.IGNORECASE).strip()
        body = re.sub(r"\s{2,}", " ", body)
        cand["question"] = f"<image 1> {body}" if body else "<image 1>"
    else:

        if "<image" in q:
            return {"status": "rejected", "reason": "image_token_in_text_only"}


        scen = (cand.get("scenario") or "").strip()
        if scen:
            probe = re.sub(r"\s+", " ", scen)[:50].lower()
            if probe and probe not in re.sub(r"\s+", " ", q).lower():
                cand["question"] = f"{scen}\n\n{q}".strip()
    return {"status": "ok", "candidate": cand}


def _solve_mcq(context_block: str, question: str, options: list[str],
               temperature: float, timeout: int = 120) -> str:
    """One solve trial. Returns the selected option letter (or '')."""
    opts_text = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
    payload = {
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": SOLVE_PROMPT.format(
            context_block=context_block, question=question, options_text=opts_text)}],
        "temperature": temperature,
        "max_tokens": 12000,
    }
    try:
        data = _post_retry(CHAT_URL, payload, timeout=timeout, tries=2)
    except Exception:
        return ""
    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
    res = extract_json(msg.get("content") or msg.get("reasoning_content") or "")
    return extract_option_letter((res or {}).get("selected_option", ""))


def _check_factual(cand: dict, timeout: int = 150) -> dict:
    """Gate 0 — independent factual-correctness audit.

    Rejects items whose 'correct' answer rests on a MISSTATED universal truth (a formula /
    derivative / identity / law / sign rule contradicting course-independent mathematics or
    physics) — the misspeak / ASR-transcription failure mode. Legitimate non-generality
    (specific value, stated approximation, assumption/scope, notation) passes.

    Returns {ok: bool, fact_kind, independent_answer, reason}. fact_kind is
    'false_universal_claim' for a real factual error, and 'checker_error' if the auditor could
    not be parsed (both reject, but the caller distinguishes them for regen feedback).
    """
    options = cand["options_final"]
    gold = cand["answer_letter"]
    gi = LETTERS.index(gold) if gold in LETTERS else -1
    gold_text = options[gi] if 0 <= gi < len(options) else ""
    vf = _video_knowledge(cand)
    domain = cand.get("adaptation_domain") or "STEM"
    opts_text = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))


    profile = form_spec(cand.get("form_key", "")).get("verifier_profile", "numeric")
    if not cand.get("form_key"):

        profile = "framework" if "case study" in str(cand.get("qa_type", "")).lower() else "numeric"
    template = FACT_CHECK_PROMPT_CASE if profile == "framework" else FACT_CHECK_PROMPT
    prompt = template.format(
        domain=domain,
        figure_spec=cand.get("figure_spec") or "(no figure — this item refers to the video only)",
        question=cand.get("question", ""),
        options_text=opts_text,
        gold_letter=gold, gold_text=gold_text,
        naive=cand.get("naive_general_answer", "(unspecified)"),
        fact_what=vf.get("what", ""), fact_value=vf.get("value", ""),
        fact_quote=vf.get("source_quote") or vf.get("source_observation", ""),
        derivation=_derivation_text(cand),
    )


    payload = {
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 16000,
    }
    res = None
    for _ in range(2):
        try:
            data = _post_retry(CHAT_URL, payload, timeout=timeout, tries=1)
        except Exception:
            continue
        msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
        res = extract_json(msg.get("content") or msg.get("reasoning_content") or "")
        if res:
            break
    if not res:


        return {"ok": False, "fact_kind": "checker_error",
                "independent_answer": "", "reason": "factual checker returned no parseable verdict"}
    kind = str(res.get("fact_kind", "")).strip()
    ok = bool(res.get("claimed_answer_is_correct")) and kind != "false_universal_claim"
    return {"ok": ok, "fact_kind": kind,
            "independent_answer": res.get("independent_answer", ""),
            "reason": res.get("reason", "")}


def verify_candidate(cand: dict, blind_trials: int) -> dict:
    """Three-gate battery enforcing genuine, CORRECT reasoning-beyond-video dependency.
    Works for image forms (figure_spec = image text-proxy) AND text-only forms.

      gate_factual         : the claimed correct answer must be TRUE math/physics — reject
                             misspeaks / mis-transcriptions that make a false claim the "fact"
      gate_no_video_shortcut : (question [+figure] + general knowledge, NO video) -> must FAIL
                             to recover gold (else the item is answerable without the video)
      gate_video_grounded  : (question [+figure] + the video-derived knowledge) -> must recover
                             gold (grounding / solvability sanity check)

    Returns {passed, reason, factual, image_general_correct, image_lecture_correct, gold}.
    (The *_correct keys are kept for backward-compatible gallery/summary code; for text-only
    items they count the text-solver trials.)
    """
    question = cand["question"]
    options = cand["options_final"]
    gold = cand["answer_letter"]
    figure_spec = cand.get("figure_spec", "")
    vf = _video_knowledge(cand)
    requires_image = cand.get("requires_image")
    if requires_image is None:
        requires_image = form_spec(cand.get("form_key", "")).get("requires_image", True)


    factual = _check_factual(cand)
    if not factual["ok"]:

        reason = "factual_error" if factual.get("fact_kind") == "false_universal_claim" \
            else "factual_uncertain"
        return {"passed": False, "reason": reason, "gold": gold,
                "factual": factual, "image_general_correct": None,
                "blind_trials": blind_trials, "image_lecture_correct": None}


    ctx_g = (CTX_IMAGE_GENERAL.format(figure_spec=figure_spec) if requires_image
             else CTX_TEXT_GENERAL)
    g_correct = sum(1 for _ in range(blind_trials)
                    if _solve_mcq(ctx_g, question, options, 0.4) == gold and gold)
    threshold = blind_trials // 2 + 1
    no_video_shortcut = g_correct >= threshold

    if no_video_shortcut:
        return {"passed": False, "reason": "shortcut", "gold": gold, "factual": factual,
                "image_general_correct": g_correct, "blind_trials": blind_trials,
                "image_lecture_correct": None}


    ctx_l = (CTX_IMAGE_PLUS_LECTURE.format(
                 figure_spec=figure_spec, fact_what=vf.get("what", ""), fact_value=vf.get("value", ""))
             if requires_image else
             CTX_TEXT_PLUS_VIDEO.format(fact_what=vf.get("what", ""), fact_value=vf.get("value", "")))
    l_correct = _solve_mcq(ctx_l, question, options, 0.0) == gold and bool(gold)
    if not l_correct:
        return {"passed": False, "reason": "ungrounded", "gold": gold, "factual": factual,
                "image_general_correct": g_correct, "blind_trials": blind_trials,
                "image_lecture_correct": 0}

    return {"passed": True, "reason": "ok", "gold": gold, "factual": factual,
            "image_general_correct": g_correct, "blind_trials": blind_trials,
            "image_lecture_correct": 1}


def render_image(figure_spec: str, size: str, quality: str, timeout: int = 300) -> bytes:
    payload = {
        "model": IMAGE_MODEL,
        "prompt": figure_spec,
        "n": 1,
        "size": size,
        "quality": quality,
        "output_format": "png",
    }
    data = _post_retry(IMAGE_URL, payload, timeout=timeout)
    b64 = (data.get("data") or [{}])[0].get("b64_json") or ""
    if not b64:
        raise RuntimeError(f"no b64 in response: {json.dumps(data)[:200]}")
    return base64.b64decode(b64)


def domain_slug(domain: str) -> str:
    """'biology and neuroscience' -> 'Biology'. Keep the leading meaningful word, TitleCased."""
    words = re.findall(r"[A-Za-z]+", domain or "")
    stop = {"and", "of", "the", "science", "general"}
    for w in words:
        if w.lower() not in stop:
            return w.capitalize()
    return (words[0].capitalize() if words else "STEM")


def bilibili_link(video_id: str) -> str:
    m = re.search(r"av(\d+)", video_id or "")
    return f"https://www.bilibili.com/video/av{m.group(1)}" if m else ""


def source_link(video_id: str) -> str:
    """Resolve Bilibili upload IDs and YouTube video IDs to public watch URLs."""
    vid = video_id or ""
    m = re.search(r"av(\d+)", vid)
    if m:
        return f"https://www.bilibili.com/video/av{m.group(1)}"
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
        return f"https://www.youtube.com/watch?v={vid}"
    return ""


def assemble_row(cand: dict, rid: str, image_rel_path: str) -> dict:
    """Build the MMMU-aligned fields (options/answer already shuffled).
    image_rel_path is "" for text-only items (no generated figure)."""
    return {
        "id": rid,
        "question": cand["question"],
        "options": cand["options_final"],
        "answer": cand["answer_letter"],
        "link_selected": source_link(cand.get("video_id", "")),
        "image_path": image_rel_path,
        "question_type": cand.get("question_type") or "multiple-choice",
        "qa_type": cand.get("qa_type") or "Problem-solving Adaptation",
    }


def _prepare_options(cand: dict) -> None:
    """Shuffle options (MCQ) / normalize (open) in place, setting options_final + answer_letter."""
    qtype = cand.get("question_type") or "multiple-choice"
    if qtype == "open" or not cand.get("options"):
        cand["options_final"] = list(cand.get("options") or [])
        cand["answer_letter"] = extract_option_letter(cand.get("correct_option")) or ""
        cand["question_type"] = "open"
    else:
        correct = extract_option_letter(cand.get("correct_option"))
        new_opts, new_letter = shuffle_plain_options(
            cand["options"], correct, cand.get("question", ""))
        cand["options_final"] = new_opts
        cand["answer_letter"] = new_letter
        cand["answer"] = new_opts[LETTERS.find(new_letter)] if new_letter in LETTERS else cand.get("answer")


def process_generate(bundle_path: Path, verify: bool, blind_trials: int,
                     max_attempts: int, form_key: str = "problem_solving") -> dict:
    """PHASE 1 — generate + verify ONE video. Does NOT render an image.

    An accepted image candidate gets status "verified" (ready for the render phase); an accepted
    text-only candidate gets status "verified_textonly" (finalized after the render phase, no image).
    Rejected / failed videos keep their reject status. No gpt-image-2 call happens here.
    """
    spec = form_spec(form_key)
    requires_image = spec["requires_image"]
    out: dict[str, Any] = {
        "video_id": bundle_path.parent.name,
        "bundle_path": str(bundle_path),
        "status": "pending",
        "error": None,
        "gen_seconds": None,
        "attempts": 0,
        "form_key": form_key,
    }
    try:
        t0 = time.time()
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if not bundle.get("event_captions"):
            out["status"] = "skipped_no_captions"
            return out
        video_id = bundle.get("video_id", bundle_path.parent.name)
        out["video_id"] = video_id
        dom = domain_slug(bundle.get("adaptation_domain", ""))
        rid = f"stem_{dom}_{video_id}"

        cand: dict | None = None
        feedback = ""

        for attempt in range(1, max_attempts + 1):
            out["attempts"] = attempt
            gen = generate_candidate(bundle, feedback=feedback, form_key=form_key)
            if gen is None:
                feedback = REGEN_FEEDBACK["structure"]
                out["status"] = "gen_failed"
                continue
            if gen.get("status") == "rejected":
                if gen.get("reason") == "image_token_in_text_only":

                    feedback = REGEN_FEEDBACK["image_token_in_text_only"]
                    out["status"] = "gen_failed"
                    continue
                out["status"] = "rejected_by_generator"
                out["reject_reason"] = gen.get("reason", "")
                return out

            c = gen["candidate"]
            c["video_id"] = video_id
            c["form_key"] = form_key
            c["qa_type"] = spec["qa_type"]
            c["requires_image"] = requires_image
            if not c.get("question") or not (c.get("options") or c.get("answer")):
                feedback = REGEN_FEEDBACK["structure"]; out["status"] = "gen_failed"; continue
            _prepare_options(c)

            if (c.get("question_type") or "multiple-choice") != "open" and len(c.get("options_final") or []) < 4:
                feedback = REGEN_FEEDBACK["few_options"]; out["status"] = "gen_failed"; continue

            if not verify:
                cand = c
                break

            v = verify_candidate(c, blind_trials)
            c["verify"] = v
            if v["passed"]:
                cand = c
                break

            cand = c
            feedback = REGEN_FEEDBACK.get(v["reason"], "")
            out["status"] = f"rejected_{v['reason']}"

        if cand is None:
            out.setdefault("status", "gen_failed")
            out["gen_seconds"] = round(time.time() - t0, 1)
            return out


        if verify and not (cand.get("verify") or {}).get("passed"):
            out["candidate"] = cand
            out["gen_seconds"] = round(time.time() - t0, 1)
            return out

        out["qa_type"] = spec["qa_type"]
        out["knowledge_point"] = _video_knowledge(cand).get("what", "")
        if requires_image:

            figure_spec = (cand.get("figure_spec") or "").strip()
            if not figure_spec:
                out["status"] = "skipped_no_figure_spec"
                out["candidate"] = cand
                out["gen_seconds"] = round(time.time() - t0, 1)
                return out
            out.update({
                "status": "verified",
                "id": rid,
                "candidate": cand,
                "gen_seconds": round(time.time() - t0, 1),
            })
        else:

            out.update({
                "status": "verified_textonly",
                "id": rid,
                "candidate": cand,
                "gen_seconds": round(time.time() - t0, 1),
            })
    except Exception as e:
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def finalize_textonly(rec: dict) -> dict:
    """Finalize a verified_textonly record to status 'ok' with an empty image (no render)."""
    cand = rec.get("candidate") or {}
    rid = rec.get("id") or f"stem_{domain_slug(cand.get('adaptation_domain',''))}_{rec.get('video_id','')}"
    rec["row"] = assemble_row(cand, rid, "")
    rec["id"] = rid
    rec["image_bytes"] = 0
    rec["image_path"] = ""
    rec["status"] = "ok"
    return rec


def render_record(rec: dict, outdir: Path, size: str, quality: str) -> dict:
    """PHASE 2 — render the figure for a "verified" record and finalize it to status "ok"."""
    try:
        cand = rec.get("candidate") or {}
        rid = rec.get("id") or f"stem_{domain_slug(cand.get('adaptation_domain',''))}_{rec.get('video_id','')}"
        requires_image = cand.get("requires_image")
        if requires_image is None:
            requires_image = form_spec(cand.get("form_key", "")).get("requires_image", True)
        if not requires_image:

            return finalize_textonly(rec)
        figure_spec = (cand.get("figure_spec") or "").strip()
        if not figure_spec:
            rec["status"] = "skipped_no_figure_spec"
            return rec
        img = render_image(figure_spec, size, quality)
        safe = rid.replace(":", "_").replace("/", "_")
        (outdir / "images").mkdir(parents=True, exist_ok=True)
        (outdir / "images" / f"{safe}.png").write_bytes(img)
        image_rel = f"images/{safe}.png"
        rec["row"] = assemble_row(cand, rid, image_rel)
        rec["id"] = rid
        rec["image_bytes"] = len(img)
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "render_failed"
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def write_parquet(rows: list[dict], outdir: Path, path: Path) -> None:
    import pandas as pd
    records = []
    for r in rows:
        rel = r.get("image_path") or ""
        if rel:
            img_abs = outdir / rel
            png = img_abs.read_bytes() if img_abs.is_file() else b""
        else:
            png = b""
        records.append({
            "id": r["id"],
            "question": r["question"],
            "options": r["options"],
            "answer": r["answer"],
            "link_selected": r["link_selected"],
            "image": {"bytes": png, "path": os.path.basename(rel)},
            "question_type": r["question_type"],
            "qa_type": r["qa_type"],
        })
    df = pd.DataFrame.from_records(records,
        columns=["id", "question", "options", "answer", "link_selected",
                 "image", "question_type", "qa_type"])
    df.to_parquet(path, index=False)


def write_gallery(results: list[dict], outdir: Path, path: Path) -> None:
    cards = []
    for r in results:
        cand = r.get("candidate") or {}
        row = r.get("row") or {}
        if r.get("image_path"):
            rel = os.path.relpath(outdir / r["image_path"], path.parent)
            img_html = f'<img src="{rel}" loading="lazy"/>'
        elif r.get("error"):
            img_html = f'<div class="err">ERROR: {r["error"]}</div>'
        else:
            img_html = f'<div class="err">{r.get("status")}</div>'
        q = (row.get("question") or "").replace("<", "&lt;")
        opts = "".join(
            f"<li{' class=ok' if LETTERS.find(row.get('answer','')) == i else ''}>{o}</li>"
            for i, o in enumerate(row.get("options") or []))
        vf = _video_knowledge(cand)
        v = cand.get("verify") or {}
        if v:
            bt = v.get("blind_trials")
            vbadge = (f"<span class='vb {'vgood' if v.get('passed') else 'vbad'}'>"
                      f"img+gen {v.get('image_general_correct')}/{bt}"
                      f"{' ✓dep' if v.get('passed') else ' ✗ '+str(v.get('reason'))}</span>")
        else:
            vbadge = ""
        cards.append(f"""
        <div class="card">
          <div class="hd"><b>{r.get('id') or r.get('video_id')}</b> · {cand.get('figure_type') or '-'}
             · {r.get('qa_type') or cand.get('qa_type') or ''} · {r.get('gen_seconds')}s
             · answer=<b>{row.get('answer')}</b> · {r.get('status')} {vbadge}</div>
          <div class="body">
            <div class="imgwrap">{img_html}</div>
            <div class="txt">
              <div class="q">{q}</div>
              <ol type="A">{opts}</ol>
              <details><summary>video-derived knowledge (recalled from video, not in image)</summary>
                <p>{(vf.get('what') or '')} — <b>{vf.get('value') or ''}</b><br>
                <i>why not general:</i> {(vf.get('why_not_general') or '')}<br>
                <i>source:</i> {(vf.get('source_quote') or vf.get('source_observation') or '')} <code>{vf.get('source_timestamp') or ''}</code></p></details>
              <details><summary>scenario</summary><p>{cand.get('scenario') or cand.get('image_scenario') or ''}</p></details>
              <details><summary>anti-shortcut: naive general-knowledge answer (WRONG)</summary>
                <p><b>{cand.get('naive_general_answer') or '—'}</b><br>{cand.get('why_general_fails') or ''}</p></details>
              <details><summary>why both needed</summary><p>{cand.get('why_both_needed') or ''}</p></details>
              <details><summary>derivation</summary><p>{cand.get('derivation') or ''}</p></details>
              <details><summary>figure_spec (fed to gpt-image-2)</summary><p>{(cand.get('figure_spec') or '')}</p></details>
            </div>
          </div>
        </div>""")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Adaptation-Image (dual-dependency)</title>
<style>
 body{{font:14px system-ui,sans-serif;background:#0f1115;color:#e6e8ec;margin:0;padding:16px}}
 h1{{font-size:18px}}
 .card{{background:#181b22;border:1px solid #2a2f3a;border-radius:10px;margin:14px 0;padding:12px}}
 .hd{{color:#9aa0aa;font-size:12px;margin-bottom:8px}}
 .body{{display:grid;grid-template-columns:460px 1fr;gap:16px}}
 .imgwrap img{{width:460px;border-radius:8px;background:#fff}}
 .err{{width:460px;min-height:160px;display:flex;align-items:center;justify-content:center;
       background:#241a1a;color:#e5534b;border-radius:8px;text-align:center;padding:10px}}
 .q{{font-weight:600;margin-bottom:8px}}
 ol li.ok{{color:#39c07a;font-weight:700}}
 details{{margin:6px 0}} summary{{cursor:pointer;color:#4c8bf5}}
 .vb{{padding:1px 7px;border-radius:6px;font-size:11px;font-weight:700}}
 .vgood{{background:#123d24;color:#39c07a;border:1px solid #39c07a}}
 .vbad{{background:#241a1a;color:#e5534b;border:1px solid #e5534b}}
 code{{background:#0f1115;padding:1px 4px;border-radius:4px}}
 p{{white-space:pre-wrap;word-break:break-word;color:#c8ccd4}}
 @media(max-width:960px){{.body{{grid-template-columns:1fr}} .imgwrap img,.err{{width:100%}}}}
</style></head><body>
<h1>Adaptation-Image (dual-dependency) — {len(results)} samples</h1>
{''.join(cards)}
</body></html>"""
    path.write_text(html, encoding="utf-8")


_INDEX_DIR = ROOT / ".bundle_index"


def _scan_root(root: Path) -> list[Path]:
    """List caption bundles beneath a root using concurrent file checks."""
    try:
        subs = [e.path for e in os.scandir(root) if e.is_dir()]
    except (FileNotFoundError, NotADirectoryError):
        return []
    def _has(p):
        cb = os.path.join(p, "caption_bundle.json")
        return cb if os.path.exists(cb) else None
    out: list[Path] = []
    with ThreadPoolExecutor(max_workers=64) as ex:
        for r in ex.map(_has, subs, chunksize=64):
            if r:
                out.append(Path(r))
    return out


def _bundles_in_dir(d: Path, use_cache: bool = True) -> list[Path]:
    """List caption bundles, caching by root path and modification time."""
    if not d.is_dir():
        return []
    if not use_cache:
        return _scan_root(d)
    try:
        mt = os.stat(d).st_mtime
    except OSError:
        mt = None
    _INDEX_DIR.mkdir(exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(d)).strip("_")
    cache = _INDEX_DIR / f"{key}.txt"
    if mt is not None and cache.exists():
        try:
            lines = cache.read_text(encoding="utf-8").splitlines()
            if lines and lines[0] == f"#mtime={mt}":
                return [Path(x) for x in lines[1:] if x]
        except OSError:
            pass
    found = _scan_root(d)
    try:
        cache.write_text(f"#mtime={mt}\n" + "\n".join(str(p) for p in found), encoding="utf-8")
    except OSError:
        pass
    return found


def find_bundles(caption_dirs, start: int, n: int, sample: int = 0) -> list[Path]:

    if isinstance(caption_dirs, (str, Path)):
        caption_dirs = [caption_dirs]


    expanded: list[str] = []
    for cd in caption_dirs:
        s = str(cd)
        if any(ch in s for ch in "*?["):
            hits = sorted(glob.glob(s))
            if not hits:
                print(f"WARN: glob matched nothing: {s}", file=sys.stderr)
            expanded.extend(hits)
        else:
            expanded.append(s)
    caption_dirs = expanded or caption_dirs

    paths: list[Path] = []
    seen: set[str] = set()
    for cd in caption_dirs:
        d = Path(cd)
        if not d.exists():
            print(f"WARN: caption dir not found, skipping: {d}", file=sys.stderr)
            continue

        found = _bundles_in_dir(d) or _bundles_in_dir(d / "raw")
        if not found:
            print(f"WARN: no caption_bundle.json under {d} (or {d}/raw); skipping",
                  file=sys.stderr)
            continue
        for bp in found:
            vid = bp.parent.name
            if vid in seen:
                continue
            seen.add(vid)
            paths.append(bp)

    if not paths:
        print(f"ERROR: no caption bundles found across: {caption_dirs}", file=sys.stderr)
        return []

    paths = sorted(paths, key=lambda p: p.parent.name)
    if sample and sample > 0 and sample < len(paths):


        step = len(paths) / sample
        idxs = sorted({int(i * step) for i in range(sample)})
        return [paths[i] for i in idxs]
    paths = paths[start:start + n] if n > 0 else paths[start:]
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Adaptation-Image dual-dependency pipeline")
    ap.add_argument("--caption-dir", nargs="+", default=DEFAULT_CAPTION_DIRS,
                    help="one or more caption source dirs (merged). Each holds "
                         "<video_id>/caption_bundle.json directly or under a raw/ subdir.")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--n", type=int, default=8, help="number of videos (0 = all)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="evenly sample N videos across the WHOLE set (representative pilot; overrides start/n)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--quality", default="low", choices=["low", "medium", "high", "auto"])
    ap.add_argument("--verify", dest="verify", action="store_true", default=True,
                    help="enforce video dependency: figure+general-knowledge must FAIL (default on)")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip the blind-gate verification (faster, weaker items)")
    ap.add_argument("--blind-trials", type=int, default=3,
                    help="figure+general-knowledge solve trials; majority decides shortcut")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="regeneration attempts before giving up on a video")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="skip videos already finalized in adaptation_all.jsonl (default on)")
    ap.add_argument("--fresh", dest="resume", action="store_false",
                    help="ignore any existing results and reprocess everything")
    ap.add_argument("--phase", choices=["generate", "render", "all"], default="all",
                    help="generate = make+verify all questions (no images); render = draw figures "
                         "for verified questions; all = phase 1 then phase 2 (default)")
    ap.add_argument("--qa-mode", choices=["problem_solving", "case_study"],
                    default=None,
                    help="DEPRECATED alias for --forms (problem_solving|case_study).")
    ap.add_argument("--forms", default=None,
                    help="comma list of question families to mint (or 'all'). "
                         f"choices: {', '.join(FORMS)}. default: {','.join(DEFAULT_FORMS)}")
    ap.add_argument("--form-mix", default=None,
                    help="optional weights 'key:w,key:w' controlling per-video form assignment "
                         "(default: uniform over --forms).")
    args = ap.parse_args()


    if args.forms:
        forms = list(FORMS) if args.forms.strip() == "all" else \
            [f.strip() for f in args.forms.split(",") if f.strip()]
    elif args.qa_mode:
        forms = [args.qa_mode]
    else:
        forms = list(DEFAULT_FORMS)
    bad = [f for f in forms if f not in FORMS]
    if bad:
        print(f"ERROR: unknown --forms {bad}; choices: {list(FORMS)}", file=sys.stderr)
        return 1

    if args.form_mix:
        weighted: list[str] = []
        for part in args.form_mix.split(","):
            k, _, w = part.partition(":")
            k = k.strip()
            if k in forms:
                weighted.extend([k] * max(1, int(w or 1)))
        form_pool = weighted or forms
    else:
        form_pool = forms

    outdir = Path(args.outdir)
    (outdir / "images").mkdir(parents=True, exist_ok=True)
    all_jsonl = outdir / "adaptation_all.jsonl"

    bundles = find_bundles(args.caption_dir, args.start, args.n, args.sample)
    if not bundles:
        return 1


    RERUN = {"error", "gen_failed", "pending", None}
    prior: dict[str, dict] = {}
    if args.resume and all_jsonl.exists():
        with open(all_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prior[r.get("video_id", "")] = r


    RERUN = {"error", "gen_failed", "pending", None}
    RENDER_RERUN = {"verified", "render_failed"}
    prior: dict[str, dict] = {}
    if args.resume and all_jsonl.exists():
        with open(all_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prior[r.get("video_id", "")] = r

    write_lock = threading.Lock()
    t0 = time.time()

    def checkpoint(rec: dict) -> None:
        prior[rec.get("video_id", "")] = rec
        with write_lock:
            with open(all_jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    span = ('sample=' + str(args.sample)) if args.sample else f"start={args.start} n={args.n}"


    if args.phase in ("generate", "all"):
        done_ids = {vid for vid, r in prior.items() if r.get("status") not in RERUN}
        todo = [bp for bp in bundles if bp.parent.name not in done_ids]
        print(f"[PHASE 1 · generate] {len(bundles)} bundles ({span}); resume skips "
              f"{len(bundles)-len(todo)} -> {len(todo)} to generate. gen={GEN_MODEL} "
              f"forms={form_pool} "
              f"workers={args.workers} verify={args.verify} bt={args.blind_trials} "
              f"max_att={args.max_attempts} chat_rpm={CHAT_LIMITER.rpm}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_generate, bp, args.verify, args.blind_trials,
                              args.max_attempts,
                              video_form_sequence(bp.parent.name, form_pool)): bp for bp in todo}
            done = 0
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"video_id": futs[fut].parent.name, "status": "error",
                           "error": f"{type(e).__name__}: {e}"}
                checkpoint(res)
                done += 1
                vv = (res.get("candidate") or {}).get("verify") or {}
                vnote = (f"[no-vid {vv.get('image_general_correct')}/{vv.get('blind_trials')}]"
                         if vv else "")
                print(f"  gen [{done}/{len(todo)}] {res.get('video_id',''):20s} "
                      f"{res.get('form_key',''):16s} {res.get('status',''):22s} "
                      f"a{res.get('attempts','?')} {res.get('gen_seconds') or ''}s {vnote} "
                      f"{(res.get('error') or res.get('reject_reason') or '')[:60]}", flush=True)
        n_ready = sum(1 for r in prior.values()
                      if r.get("status") in ("verified", "verified_textonly", "ok"))
        print(f"[PHASE 1 done] {n_ready} questions ready to finalize/render.", flush=True)


    if args.phase in ("render", "all"):
        to_render = [r for r in prior.values() if r.get("status") in RENDER_RERUN]
        print(f"[PHASE 2 · render] {len(to_render)} verified questions to render "
              f"(image={IMAGE_MODEL} q={args.quality} workers={args.workers} "
              f"img_rpm={IMAGE_LIMITER.rpm}).", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(render_record, dict(r), outdir, args.size, args.quality): r
                    for r in to_render}
            done = 0
            for fut in as_completed(futs):
                src = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = dict(src); res["status"] = "render_failed"
                    res["error"] = f"{type(e).__name__}: {e}"
                checkpoint(res)
                done += 1
                ib = res.get("image_bytes")
                print(f"  render [{done}/{len(to_render)}] {res.get('video_id',''):20s} "
                      f"{res.get('status',''):14s} {str(ib)+'B' if ib else ''} "
                      f"{(res.get('error') or '')[:70]}", flush=True)


        to_finalize = [r for r in prior.values() if r.get("status") == "verified_textonly"]
        if to_finalize:
            print(f"[PHASE 2 · finalize] {len(to_finalize)} text-only questions (no image).",
                  flush=True)
            for r in to_finalize:
                checkpoint(finalize_textonly(dict(r)))


    results = list(prior.values())
    with open(all_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for r in results:
        if r.get("row"):
            r["image_path"] = r["row"]["image_path"]
            r["id"] = r["row"]["id"]
            r["qa_type"] = r["row"]["qa_type"]
    ok_rows = [r["row"] for r in results if r.get("status") == "ok" and r.get("row")]

    parquet_path = outdir / "adaptation_mmmu.parquet"
    if ok_rows:
        write_parquet(ok_rows, outdir, parquet_path)
    write_gallery(results, outdir, outdir / "gallery.html")

    n_ok = len(ok_rows)
    n_err = sum(1 for r in results if r.get("status") == "error")
    counts: dict[str, int] = {}
    for r in results:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
    accepted_verifies = [v for v in ((r.get("candidate") or {}).get("verify")
                         for r in results if r.get("status") == "ok") if v]
    summary = {
        "timestamp": datetime.now().isoformat(),
        "gen_model": GEN_MODEL, "image_model": IMAGE_MODEL,
        "innovation": "reasoning-beyond-video (video-derived knowledge ⊗ new scenario; image or text-only)",
        "forms": form_pool,
        "qa_type_counts": _count_by(results, lambda r: (r.get("row") or {}).get("qa_type")
                                    or (r.get("candidate") or {}).get("qa_type") or "?",
                                    only_ok=True),
        "phase": args.phase,
        "pipeline": "two-phase: (1) generate+verify all videos, then (2) render image forms + finalize text-only",
        "verified_pending_render": counts.get("verified", 0) + counts.get("render_failed", 0),
        "verified_textonly_pending": counts.get("verified_textonly", 0),
        "verification": {
            "enabled": args.verify, "blind_trials": args.blind_trials,
            "max_attempts": args.max_attempts,
            "gate_factual": "claimed correct answer must be true math/physics (no misstated universal truth)",
            "gate_no_video_shortcut": "question [+figure] + general knowledge (NO video) must FAIL to recover gold",
            "gate_video_grounded": "question [+figure] + video-derived knowledge must recover gold",
            "accepted_no_video_solved": [v.get("image_general_correct") for v in accepted_verifies],
            "rejected_factual_error": counts.get("rejected_factual_error", 0),
            "rejected_factual_uncertain": counts.get("rejected_factual_uncertain", 0),
            "rejected_shortcut": counts.get("rejected_shortcut", 0),
            "rejected_ungrounded": counts.get("rejected_ungrounded", 0),
        },
        "rpm_caps": {"chat": CHAT_LIMITER.rpm, "image": IMAGE_LIMITER.rpm},
        "caption_input_excludes": ["av_sync_pairs"],
        "gen_max_chars": GEN_MAX_CHARS,
        "total": len(results), "ok": n_ok, "errors": n_err,
        "status_counts": counts,
        "outputs": {"parquet": str(parquet_path), "all_jsonl": str(all_jsonl),
                    "images": str(outdir / "images"), "gallery": str(outdir / "gallery.html")},
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone in {round(time.time()-t0,1)}s — corpus={len(results)} ok={n_ok} "
          f"errors={n_err} counts={counts}", flush=True)
    print(f"  parquet: {parquet_path}\n  jsonl:   {all_jsonl}\n"
          f"  gallery: {outdir/'gallery.html'}\n  summary: {outdir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
