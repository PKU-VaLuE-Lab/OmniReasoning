"""Generate evidence-grounded questions from audio-visual captions."""
from __future__ import annotations

import argparse
import asyncio
import glob as globmod
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm


from OmniQA.generation.task_types import TASK_TYPES, TASK_TYPE_BY_ID, render_task_section, render_type_catalog
from OmniQA.generation.video_map import load_video_map


GENERATION_PROMPT = r"""You are an expert exam writer building HARD audio-visual reasoning questions.
From the caption bundle of ONE video, write EXACTLY ONE multiple-choice question of the
ASSIGNED type below.

{task_section}

## THE ONE PROPERTY THAT MATTERS
The question is only valid if BOTH modalities of the actual video are needed to answer
it, and a solver WITHOUT the video cannot answer it:
  • A solver who sees ONLY the visuals            → must be UNABLE to answer.
  • A solver who hears ONLY the audio             → must be UNABLE to answer.
  • A solver who has NO video and reads ONLY the question + options (using world
    knowledge, priors, and wording cues)          → must be UNABLE to answer.
A solver with audio + visual together → answers uniquely.

{structure_rules}

## HARD PROHIBITIONS (question design anti-patterns — any one makes the draft INVALID)
  1. QUESTION LEAKAGE. The question must NOT quote speech verbatim (no "when the man
     says '…'"), must NOT embed the sound description it asks about, and must NOT
     state the decisive audio fact or the decisive visual fact. It may NAME the
     referents, nothing more.
  2. TEXT-ONLY SOLVABILITY. Before finalizing, mentally hide the video: if the
     question wording + options + general knowledge already determine the answer
     → INVALID. Rewrite so the answer is an ARBITRARY, video-internal detail.
  3. WORLD-KNOWLEDGE answers. Do NOT ask facts that live in general knowledge (what
     made actor X famous, which city team Z is from). Priors must not supply the answer.
  4. OPTION PARITY VIOLATIONS. All {num_options} options share the same type,
     granularity, syntax and rough length; the correct option must NOT be the only one
     sharing distinctive words with the question; no option obviously absurd; no
     "all/none of the above"; no bare numbers when other options are phrases.
  5. VISUAL-SYMBOL SUBSTITUTION. A fact claimed to be audio-only must not also be shown
     by an on-screen symbol (flag, logo, banner, scoreboard, jersey text, caption
     overlay), and a fact claimed visual-only must not be uttered in the audio.
  6. UNGROUNDED CONTENT. Every clue MUST be quoted verbatim from the caption bundle
     and carry the EXACT fine-grained timestamp of the caption entry it comes from
     (e.g. '00:14-00:19' — never a rounded, widened, or invented range). Do NOT
     invent events, objects, sounds or numbers that are not in the captions (this
     will be checked against the actual video later).
  7. TEMPORAL INCONSISTENCY. If the question asks about a TRANSIENT state AT THE
     MOMENT of an event ("what is on screen when X is said", "what is heard while Y
     is shown", "此时", "当时", "…的时候"), the two clues MUST have the SAME or
     overlapping timestamps. FAR-APART clues are only valid when they bind a
     TIME-INVARIANT fact (a referent's identity, value, or label) — never a
     momentary state that may change between the two timestamps.
  8. UNRELIABLE AUDIO ANCHOR. The anchor (the fact that singles out the referent)
     must NEVER come from Music / background / ambience audio captions — those
     entries are unreliable and hallucination-prone ("the background music
     continues playing" tells nobody WHICH referent is meant). The anchor must be
     specific SPEECH (a name, number, quoted phrase) or a clearly identifiable
     sound event (whistle, alarm, crash, animal call). Music/BGM may only serve as
     the ANSWER/topic when the assigned type explicitly calls for it, and then
     only its gross character (genre/mood), never fine-grained claims.

## DISTRACTOR PARITY
Provide EXACTLY {num_options} options, all grounded in THIS video or plausible for it:
  • The option a VISUAL-ONLY viewer would pick is PRESENT but WRONG.
  • The option an AUDIO-ONLY listener would pick is PRESENT but WRONG.
  • The option a no-video PRIORS solver would pick is PRESENT but WRONG.

## CLUE DISTANCE — far apart is a PREFERENCE, never a goal
"Long-dependency" means ONE referent (a person / object / label) is tracked across
time: the anchor moment establishes WHICH referent is meant, the answer moment reads
off a property of THAT SAME referent. PREFER anchor and answer moments that are far
apart (tens of seconds to minutes — it makes the question robust), BUT never skip a
caption entry at/near the anchor's own moment that directly provides the answer just
to reach a distant one. If the card name is on screen at the very moment the
narrator mentions its skill, THAT moment is the correct visual clue.
THE SHORTEST NATURAL PATH RULE: the reasoning chain must be the minimal path a
real viewer would take. Every hop to a distant timestamp must be NECESSARY — if the
answer is readable at the anchor's own moment, the chain is exactly those
co-temporal steps. An unnecessary jump to a distant moment is INVALID, even when
both moments show the same content (the referent may have changed in between, and
the chain becomes unverifiable). It also does NOT mean linking two unrelated events
with an inferred relation — see the next section.

## EVIDENCE INTEGRITY — the #1 quality rule (violations are auto-rejected)
Every clause of the question must be DIRECTLY supported by a specific caption entry.
These three patterns are FORBIDDEN:
  1. INVENTED CROSS-EVENT LINKS. Never assert a relation between two events that no
     caption states: a purpose (BAD: "turned on the faucet TO WASH the celery seen
     two minutes later"), a cause, an order ("after X he did Y" when the captions do
     not order them), or an identity ("the man at 01:02 is the chef at 02:50") —
     unless a caption entry explicitly establishes that relation.
  2. STATE PERSISTENCE. Never assume what is true at T1 is still true at T2 (BAD:
     asking for "the slide on screen at 08:08" using a slide caption from 03:52).
     A momentary state may only be claimed for the EXACT timestamp of its own
     caption entry.
  3. REFERENT SPLICING. The referent linking the two clues must be EXPLICITLY
     identifiable in BOTH clue entries (named, or visually/aurally distinctive in
     each) — never "probably the same person".
The ONLY legitimate long-range link is a shared referent's identity; the ONLY
legitimate properties to ask about are time-invariant ones, or momentary states at
their own exact timestamps.

## PRESUPPOSITION CHECK (required in the output — it will be programmatically verified)
For EVERY cross-event connection your question relies on, `presupposition_check`
must contain one item: the connection AND the verbatim caption quote + exact entry
timestamp that DIRECTLY supports it. A connection is only legitimate in ONE of
these three forms:
  (a) SHARED IDENTIFIER: the SAME name / label / distinctive noun appears in BOTH
      clue quotes (e.g. "#6 Hacker" in the visual quote and "Hacker" in the audio
      quote). This is REQUIRED whenever the two clues are NOT co-temporal — "the
      card the narrator meant" without the card's name in the audio is splicing.
  (b) CO-TEMPORAL co-occurrence: the two clue timestamps OVERLAP (same moment) —
      the link is the moment itself; no distant second moment may be substituted.
  (c) a caption entry that EXPLICITLY states the relation (names both sides).
If you cannot supply one of these for a connection, the question is INVALID —
rewrite it until every connection is backed.

## REASONING GRAPH (required — a DAG of hops, not a flat list)
`reasoning_steps` must form a CONNECTED reasoning DAG that ends in the step yielding
the answer:
  • List steps in INFERENCE order (the order a reasoner would walk the chain).
  • Every step carries the CONCRETE timestamp of its evidence — the graph thus
    encodes the temporal order of the events (a chain may trace backwards in time).
  • Every step after the first MUST name ≥1 earlier step in `depends_on` — the
    inference it logically rests on. No forward references, no self references.
  • No dangling steps: the FINAL step yields the answer and must transitively use
    ALL previous steps.
  • At least one `audio` step and one `visual` step; ≥2 steps total.

## CAPTION BUNDLE (events with audio_captions / visual_captions; audio_captions carry
an audio_type tag — Speech / Sound / Music. Use Sound and Music entries too, not only
speech, when the assigned type calls for it.)
{caption_bundle_json}

## OUTPUT — return ONLY valid JSON
{{
  "status": "ok",
  "candidate": {{
    "video_id": "{video_id}",
    "task_type": "{task_type_label}",
    "task_type_id": "the selected task type id",
    "type_selection_reason": "why this type fits this bundle's evidence",
    "fusion_pattern": "exactly one of: P1 | P2 | joint (write ONE word, not the alternatives)",
    "anchor_modality": "audio|visual|n/a (holistic) — which modality the ANCHOR (the fact that singles out the referent) comes from; for P1 this is audio, for P2 this is visual",
    "answer_type": "the single type T shared by all options",
    "binding_key": "the concrete named referent linking the audio and visual evidence (or 'n/a (holistic)')",
    "question": "self-contained; names referents but leaks neither decisive fact",
    "options": ["EXACTLY {num_options} option strings here, all the same type T, no letter prefixes"],
    "correct_option": "A",
    "answer": "correct option text (no letter prefix)",
    "audio_clue": {{"timestamp": "MM:SS-MM:SS copied EXACTLY from the audio caption entry you quote (keep its fine-grained range, do NOT round or widen it)", "quote": "verbatim audio caption text", "fact": "the decisive audio fact"}},
    "visual_clue": {{"timestamp": "MM:SS-MM:SS copied EXACTLY from the visual caption entry you quote (keep its fine-grained range, do NOT round or widen it)", "quote": "verbatim visual caption text", "fact": "the decisive visual fact"}},
    "time_gap_seconds": 0,
    "why_visual_only_fails": "why a visual-only viewer cannot answer",
    "why_audio_only_fails": "why an audio-only listener cannot answer",
    "why_priors_fail": "why a no-video text-only solver cannot answer",
    "single_modality_traps": {{"visual_only_would_pick": "X", "audio_only_would_pick": "Y", "priors_would_pick": "Z"}},
    "presupposition_check": [
      {{"link": "one cross-event connection the question assumes",
        "supported_by": {{"timestamp": "MM:SS-MM:SS of the caption entry that directly establishes it", "quote": "verbatim caption text of that entry"}}}}
    ],
    "reasoning_steps": [
      {{"step_id": "s1", "modality": "audio|visual|audiovisual", "timestamp": "MM:SS-MM:SS",
        "content": "evidence extracted at this step", "depends_on": [],
        "bridge": "how it advances the chain"}},
      {{"step_id": "s2", "modality": "audio|visual|audiovisual", "timestamp": "MM:SS-MM:SS",
        "content": "next inference", "depends_on": ["s1"], "bridge": "..."}}
    ]
  }}
}}

## YOU MUST ALWAYS RETURN A QUESTION — DO NOT GIVE UP
Scan ALL events before deciding. You are NOT allowed to refuse: there is no rejection
option. Always output one "status": "ok" candidate that is your best valid question of
the ASSIGNED type for this bundle.
"""

STRUCTURE_ANCHOR = r"""## STRUCTURE — "anchor + atomic answer" (follow exactly)
  • ATOMIC ANSWER: ONE single value of one type T — a number, a color, a count, a
    price, an on-screen word, an object, an emotion label, a location. It lives in
    modality M_answer at time T2. NOT a composite of two facts.
  • ANCHOR: modality M_anchor at time T1 tells WHICH of several same-type candidates
    is the target. The anchor must pin EXACTLY ONE referent — "a card that has the
    Motivation skill" is INVALID if several cards could match; the audio itself must
    contain enough to single out one referent (its name, an explicit deictic like
    "this card" while it alone is on screen, or a truly unique property).
  • BINDING KEY: name the concrete shared referent that links anchor→answer. It
    MUST be a NAMED, CONCRETE noun — a specific person, object, team, product, or
    labelled thing, ≤6 words, NO temporal/relational words.
      GOOD: "Binning", "car #7", "the white SOCKERART vase", "the red folder".
      BAD (auto-rejected): "the moment when…", "the later scene", "that point".
  • MODALITY-EXCLUSIVE ANCHOR (the golden rule): the anchor fact must be EXCLUSIVE
    to one modality and the answer EXCLUSIVE to the other — no synonym, symbol or
    paraphrase of either may appear in the other modality's captions. Self-tests:
    COVER-THE-PICTURE (audio alone cannot select the referent or read the answer)
    and COVER-THE-SOUND (visuals alone cannot either). Both must pass.
  • OPTIONS: all values of type T, all genuinely appearing in the video for
    DIFFERENT referents, so a single-modality solver cannot tell which is meant.
  • Prefer patterns where the ANSWER is audio (spoken value never shown) and the
    ANCHOR is a pure-visual property (color / on-screen-only position / distinctive
    object / unspoken on-screen text) — they leak least. When the audio names
    per-referent values for SEVERAL referents but never says which is the target,
    that is ideal."""

STRUCTURE_HOLISTIC = r"""## STRUCTURE — "holistic synthesis" (follow exactly)
  • The question synthesizes evidence spread across MULTIPLE events of the video
    (whole-video tone, trajectory, relationship, summary, cross-event plot).
  • UNION NECESSITY: the deciding evidence must be SPLIT across the two modalities —
    removing the audio evidence makes the answer underdetermined, and removing the
    visual evidence also makes it underdetermined. State both halves explicitly in
    audio_clue / visual_clue (pick the most decisive single moment of each).
  • The answer must still be checkable and unambiguous: exactly ONE option is
    supported by the video; every distractor is contradicted by at least one
    modality or fits only one modality.
  • binding_key may be 'n/a (holistic)' but the question must still name its
    referents concretely (no "the scene", "the moment" without content).
  • EVERY reasoning step — even synthesis steps — must carry a CONCRETE timestamp
    of the most representative moment it cites. Writing 'holistic', 'joint',
    'combined', 'entire' or similar words into a timestamp field is FORBIDDEN;
    a synthesis step must point at its single most decisive moment instead."""

STRUCTURE_RULES = {"anchor": STRUCTURE_ANCHOR, "holistic": STRUCTURE_HOLISTIC}

SELF_CRITIQUE_PROMPT = r"""You are a ruthless RED-TEAM reviewer for audio-visual questions. Below is a draft
multiple-choice question with its supporting clues, plus the full caption bundle it
was built from. Your job: try to BREAK the question, then return a FIXED version.

A question is BROKEN if ANY of these can answer it WITHOUT fusing both modalities:
  • VISUAL-ONLY: using only the visual descriptions (incl. any on-screen text,
    flags, logos, jersey numbers, uniforms visible in the captions).
  • AUDIO-ONLY: using only the spoken transcript.
  • PRIORS / TEXT-ONLY: using only the question wording + options + general world
    knowledge, no video at all.
  • WEAK DISTRACTORS: one option is obviously right (or the others obviously wrong)
    to someone who has not seen the video.
  • FUZZY ANCHOR: the binding referent is vague / not a concrete named thing.
  • LEAK: the decisive audio or visual fact is stated in the question wording
    (incl. quoting speech verbatim).
  • UNGROUNDED: a clue or option asserts something NOT present in the captions.
  • INVENTED LINK: the question asserts a cross-event relation (purpose, cause,
    order, identity) that NO caption entry states — e.g. "turned on the faucet to
    wash the celery", "the slide still on screen minutes later", "the same man".
  • STATE PERSISTENCE: a momentary state is claimed at a time other than its own
    caption entry's timestamp.
  • WEAK PRESUPPOSITION: `presupposition_check` is missing, or an item's supporting
    quote does not actually establish the claimed link.

## CAPTION BUNDLE
{caption_bundle_json}

## DRAFT
{candidate_json}

## TASK
1. Attempt each break above. Be concrete about which option each blind solver picks.
2. If the draft survives ALL breaks, return it unchanged with "revised": false.
3. If ANY break succeeds, REPAIR it: re-pick the anchor/answer/options so that the
   audio fact is exclusive to audio, the visual fact exclusive to visual, the answer
   is an arbitrary video-internal value (not world knowledge), all options have
   parity, and every clue is grounded in verbatim caption quotes. Keep the SAME
   task_type. Keep ≥2 alternating-modality reasoning hops and a far-apart time gap.

## OUTPUT — return ONLY valid JSON (same schema as the draft's candidate)
{{
  "revised": true,
  "breaks_found": ["visual_only: picks X because …", "..."],
  "candidate": {{ ...the full repaired candidate, same fields as the draft... }}
}}
If unbroken: {{ "revised": false, "breaks_found": [], "candidate": {{ ...unchanged... }} }}
"""

REGEN_FEEDBACK = {
    "priors_blind": (
        "\n\nThe previous draft FAILED: a TEXT-ONLY solver (question+options only, no "
        "video, no captions) answered correctly — the answer was derivable from world "
        "knowledge or wording cues. Re-pick the answer so it is an ARBITRARY fact "
        "specific to THIS video (this referent's number/price/color/word HERE), "
        "unknowable from general knowledge, and fix distractor parity so no option "
        "stands out from priors."
    ),
    "structure": (
        "\n\nThe previous draft was malformed. Return all required fields: a question, "
        "an options list, a correct_option, and the clues."
    ),
    "reasoning": (
        "\n\nThe previous draft was rejected: its `reasoning_steps` do not form a valid "
        "reasoning graph. Requirements: >=2 steps in inference order, each with a "
        "CONCRETE timestamp (never 'holistic'/'joint'/'combined'); each step after the "
        "first names >=1 EARLIER step in `depends_on`; no dangling steps (the final "
        "step yields the answer and transitively uses all previous steps); at least "
        "one audio AND one visual step."
    ),
    "grounding": (
        "\n\nThe previous draft was rejected by the caption-grounding check: a clue "
        "quote, timestamp, or presupposition_check entry does NOT exist in the caption "
        "bundle. Re-quote VERBATIM from the captions, copy the EXACT fine-grained "
        "timestamp of the entry you use, and back every cross-event link in "
        "presupposition_check with a verbatim supporting quote. Do NOT paraphrase "
        "quotes, do NOT invent timestamps or connections."
    ),
    "simultaneity": (
        "\n\nThe previous draft was rejected: the question asks about a TRANSIENT "
        "state at the moment of an event, but the two clues are far apart in time — "
        "the state may have changed between them. EITHER align the two clues to the "
        "SAME/overlapping timestamps, OR rewrite the question to bind a TIME-INVARIANT "
        "fact (identity / value / label of a referent) instead of a momentary state."
    ),
}


_SIMULTANEITY_RE = re.compile(
    r"\b(when the|when did|when i |when you|when he|when she|when we|when they|"
    r"when it|when a |when an |while the|while i |while he|while she|while we|"
    r"while they|while|at the moment|at that moment|at this moment|"
    r"on[- ]screen when|as the|as soon as|simultaneously|at the same time|right when|"
    r"during the)\b"
    r"|此时|当时|这时|的同时|的时候|当下|同一时刻|当我|当你|当他|当她|当我们",
    re.IGNORECASE)


def needs_simultaneity(question: str) -> bool:
    return bool(_SIMULTANEITY_RE.search(str(question or "")))

TYPE_SELECTION_PROMPT = r"""You are planning ONE exam question for a video. Below is the video's caption bundle
and the FULL catalogue of candidate question types (each with its id, structure and
complete definition). Read the definitions, then pick the SINGLE type that BEST fits
THIS video: the audio+visual evidence that type requires must actually exist in these
captions.

Rules:
- Do NOT pick a music/atmosphere type unless there are Music audio_captions entries.
- Do NOT pick a speaker-emotion / multi-character type unless the audio contains
  real dialogue with identifiable speakers.
- Do NOT pick a counting type unless countable repeated items/actions exist.
- Prefer the type whose REQUIRED audio+visual evidence is most concretely present.
- If no type has sufficient audio AND visual evidence, reject this bundle. Do not
  invent missing evidence or choose a type just to fill a question slot.

## CANDIDATE QUESTION TYPES (full definitions)
{type_list}

## CAPTION BUNDLE
{caption_bundle_json}

Return ONLY valid JSON (no other text):
{{"task_type_id": "<one id from the catalogue above>", "reason": "why this type fits the available evidence"}}
If no type is supported, return {{"reject": true, "reason": "which evidence is missing"}}.
"""


PRIORS_BLIND_PROMPT = r"""Answer this multiple-choice question. You have NO access to the video, NO captions,
NO transcript — ONLY the question and the options below.

Use everything you know about any named entities, plus priors, wording cues, and
elimination. Commit to your single best option.

Question: {question}

Options:
{options_text}

Return ONLY valid JSON (no other text):
{{"reasoning": "your reasoning", "selected_option": "A", "confidence": "high"}}
"""


CLUE_GROUNDED_ANSWER_PROMPT = r"""You are given a multiple-choice question about a video, together with the TWO
decisive facts the question was built from: one heard in the AUDIO and one seen in the
VISUALS. Together these two facts pin down EXACTLY ONE option. Combine them — use one
fact to identify which referent the question is asking about, then read that referent's
value off the other fact. Use ONLY these two facts; do NOT use outside/world knowledge
and do NOT guess from wording.

Decisive fact from AUDIO [{audio_ts}]: heard "{audio_quote}"
  => {audio_fact}

Decisive fact from VISUALS [{visual_ts}]: seen "{visual_quote}"
  => {visual_fact}

Question: {question}

Options:
{options_text}

Pick the single option that these two facts jointly entail. If the two facts genuinely
do not determine a unique option, set selected_option to "" .

Return ONLY valid JSON (no other text):
{{"reasoning": "your reasoning", "selected_option": "A", "confidence": "high"}}
"""

CAPTION_SOLVER_PROMPT = r"""You are given the FULL caption bundle of a video (audio transcripts and visual
descriptions with timestamps) and a multiple-choice question about it. Using ONLY
these captions, derive the single best answer — AUDIT the question's presuppositions
while you do, and finally check the TWO decisive facts the question was built from.

Step 1 — PRESUPPOSITION AUDIT (do this FIRST):
The question may presuppose connections between events (e.g. "the faucet was turned
on TO WASH the celery seen two minutes later", "the slide on screen at 08:08 is the
one shown at 03:52"). List every cross-event link the question assumes, and check
whether the captions EXPLICITLY support it. A link supported only by your own
plausibility reasoning ("cooking usually involves washing vegetables") is NOT
supported. If ANY presupposed link lacks direct caption support, the question is
NOT derivable: set selected_option to "" and name the unsupported presupposition.

Step 2 — DERIVE:
If all presuppositions hold, follow the evidence chain: locate the relevant audio
and visual entries, combine them, and commit to one option. If the captions do not
determine a unique answer, set selected_option to "".

Step 3 — CLUE SUPPORT:
Decisive fact from AUDIO [{audio_ts}]: heard "{audio_quote}"
  => {audio_fact}
Decisive fact from VISUALS [{visual_ts}]: seen "{visual_quote}"
  => {visual_fact}
Do these two facts JOINTLY entail YOUR derived answer? clue_facts_support = true
only if BOTH facts are needed and together sufficient for it.

## CAPTION BUNDLE
{caption_bundle_json}

Question: {question}

Options:
{options_text}

Return ONLY valid JSON (no other text):
{{"presuppositions": [{{"link": "...", "supported": true, "evidence": "..."}}],
  "reasoning": "your step-by-step derivation", "selected_option": "A",
  "clue_facts_support": true, "confidence": "high"}}
"""


class OpenAIClient:
    def __init__(self, api_key: str, api_base: str, model: str,
                 qps: float = 10.0, max_retries: int = 5, max_inflight: int = 0):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        inflight = max_inflight if max_inflight > 0 else max(1, int(qps * 4))
        self._semaphore = asyncio.Semaphore(inflight)
        self._min_interval = 1.0 / qps if qps > 0 else 0.0
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def _rate_limit(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_request_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_time = time.monotonic()

    async def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 8192,
                   disable_thinking: bool = False) -> str | None:
        async with self._semaphore:
            await self._rate_limit()
            url = f"{self.api_base}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}


            is_chat = ("gpt-5.4" in self.model) or ("deepseek" in self.model)


            _m = self.model.lower()
            is_gpt5 = (any(x in _m for x in ("gpt-5", "gpt5", "o1", "o3"))
                       and "gpt-5.4" not in _m)
            if is_gpt5:
                payload = {"model": self.model,
                           "messages": [{"role": "user", "content": prompt}],
                           "max_completion_tokens": max_tokens}
            elif is_chat:
                payload = {"model": self.model,
                           "messages": [{"role": "user", "content": prompt}],
                           "temperature": temperature, "max_tokens": max_tokens}
                if disable_thinking and "deepseek" in self.model:
                    payload["thinking"] = {"type": "disabled"}
            else:
                payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                           "temperature": temperature, "max_tokens": max_tokens}
            for attempt in range(self.max_retries):
                try:
                    async with httpx.AsyncClient(timeout=600) as client:
                        resp = await client.post(url, json=payload, headers=headers)
                        if resp.status_code == 429:
                            self._n429 = getattr(self, "_n429", 0) + 1


                            if attempt >= 11:
                                print(f"  429 quota exhausted after 12 attempts "
                                      f"(total 429s on this client: {self._n429})", flush=True)
                                return None
                            await asyncio.sleep(min(2 ** attempt * 5, 120))
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        content = data["choices"][0]["message"].get("content") or ""
                        if not content.strip():


                            if attempt == self.max_retries - 1:
                                print(f"  empty content after {self.max_retries} attempts", flush=True)
                                return None
                            await asyncio.sleep(min(2 ** attempt * 2, 60))
                            continue
                        return content
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"  API error after {self.max_retries} attempts: {type(e).__name__}: {e}", flush=True)
                        return None
                    await asyncio.sleep(min(2 ** attempt * 2, 60))
            return None


LETTERS = "ABCDEFGH"


def extract_json(text: str | None) -> dict | None:
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
    depth = 0
    start = -1
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
                    pass
                start = -1
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def extract_option_letter(text: str | None) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if len(text) == 1 and text.upper() in LETTERS:
        return text.upper()
    m = re.match(r"^([A-Ha-h])\s*[).:]\s*", text)
    if m:
        return m.group(1).upper()
    for c in text:
        if c.upper() in LETTERS:
            return c.upper()
    return ""


def strip_option_prefix(opt: str) -> str:
    return re.sub(r"^[A-Ha-h]\s*[).:]\s*", "", str(opt).strip()).strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def resolve_correct_letter(candidate: dict) -> str:
    """Map the generator's `correct_option` to an in-range option letter, ROBUST to the
    common failure mode where the model writes the answer TEXT (e.g. "glossy dark red")
    into `correct_option` instead of a letter."""
    opts = candidate.get("options") or []
    valid = set(LETTERS[:len(opts)])
    bodies_norm = {_norm(strip_option_prefix(o)): LETTERS[i] for i, o in enumerate(opts)}

    co = str(candidate.get("correct_option") or "").strip()
    if len(co) == 1 and co.upper() in valid:
        return co.upper()
    m = re.match(r"^([A-Ha-h])\s*[).:]\s*\S", co)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    if _norm(co) in bodies_norm:
        return bodies_norm[_norm(co)]
    ans = _norm(candidate.get("answer"))
    if ans in bodies_norm:
        return bodies_norm[ans]
    legacy = extract_option_letter(co)
    return legacy if legacy in valid else ""


def format_options(options: list[str]) -> str:
    lines = []
    for i, opt in enumerate(options):
        letter = LETTERS[i] if i < len(LETTERS) else str(i)
        body = strip_option_prefix(opt)
        lines.append(f"{letter}. {body}")
    return "\n".join(lines)


def shuffle_options(candidate: dict) -> dict:
    """Deterministic option shuffle (N options) to remove position bias."""
    opts = candidate.get("options") or []
    n = len(opts)
    correct = extract_option_letter(candidate.get("correct_option"))
    if n < 2 or n > len(LETTERS) or not correct or correct not in set(LETTERS[:n]):
        return candidate
    bodies = [strip_option_prefix(o) for o in opts]
    old_idx = {L: i for i, L in enumerate(LETTERS[:n])}
    correct_body = bodies[old_idx[correct]]
    old_traps = candidate.get("single_modality_traps") or {}

    h = int(hashlib.sha256((candidate.get("question") or "").encode("utf-8")).hexdigest(), 16)
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = h % (i + 1)
        h //= (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    new_bodies = [bodies[p] for p in perm]
    new_options = [new_bodies[i] for i in range(n)]
    new_correct = LETTERS[new_bodies.index(correct_body)]

    candidate = dict(candidate)
    candidate["options"] = new_options
    candidate["correct_option"] = new_correct
    candidate["answer"] = correct_body
    candidate["single_modality_traps"] = old_traps
    return candidate


def build_full_caption_text(bundle: dict, max_chars: int = 0) -> str:
    """Lean generation context: whole_video_summary + per-event audio/visual captions
    ONLY (drops av_sync_pairs, per-event summaries, indices). max_chars=0 means NO
    truncation — long-video bundles are sent in full."""
    subset: dict[str, Any] = {"video_id": bundle.get("video_id")}
    if bundle.get("whole_video_summary"):
        subset["whole_video_summary"] = bundle["whole_video_summary"]
    events = []
    for ev in bundle.get("event_captions", []):
        events.append({
            "event_id": ev.get("event_id"),
            "event_range": ev.get("event_range"),
            "audio_captions": [
                {"timestamp": a.get("timestamp"), "audio_type": a.get("audio_type"),
                 "caption": a.get("caption")}
                for a in (ev.get("audio_captions") or [])
            ],
            "visual_captions": [
                {"timestamp": v.get("timestamp"), "caption": v.get("caption")}
                for v in (ev.get("visual_captions") or [])
            ],
        })
    subset["event_captions"] = events
    text = json.dumps(subset, ensure_ascii=False)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n... (truncated)"
    return text


@dataclass
class FilterResult:
    stage: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResult:
    video_id: str
    candidate: dict[str, Any] | None = None
    generation_status: str = ""
    filters: list[FilterResult] = field(default_factory=list)
    final_status: str = "pending"
    attempts: int = 0
    revised: bool = False
    breaks_found: list[str] = field(default_factory=list)
    source: str = ""
    task_type_id: str = ""
    type_selected_by: str = ""


_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿぀-ヿ가-힯]")


def _norm_word_list(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _ts_to_sec(ts: str) -> float | None:
    """'MM:SS' / 'MM:SS-MM:SS' / 'HH:MM:SS' -> seconds of the FIRST point; None if bad."""
    ts = str(ts or "").split("-")[0].strip()
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


def _modality_corpus(bundle: dict, modality: str) -> tuple[str, set[str]]:
    """Normalized (whitespace-joined lowercase word string, word set) of every caption
    text of one modality — the ground truth a clue quote must live in."""
    key = "audio_captions" if modality == "audio" else "visual_captions"
    words: list[str] = []
    for ev in bundle.get("event_captions", []):
        for it in ev.get(key) or []:
            words.extend(_norm_word_list(it.get("caption", "")))
    return " ".join(words), set(words)


def _quote_grounded(quote: str, corpus: str, corpus_words: set[str]) -> bool:
    """Verbatim-containment first; if the model lightly paraphrased, accept when >=60%
    of the quote's distinct content words appear in the modality corpus."""
    q_words = _norm_word_list(quote)
    if not q_words:
        return False
    q = " ".join(q_words)
    if q in corpus:
        return True
    distinct = set(q_words)
    hits = sum(1 for w in distinct if w in corpus_words)
    return len(distinct) >= 4 and hits / len(distinct) >= 0.6


def reasoning_graph_reasons(candidate: dict) -> list[str]:
    """Validate that reasoning_steps form a CONNECTED reasoning DAG:
    >=2 steps, every step has a concrete parseable timestamp (the graph thus ENCODES
    the temporal order of the evidence), every non-first step depends on >=1 earlier
    step, both modalities covered, and the final step transitively uses ALL steps.
    Note: steps are listed in INFERENCE order — a chain may legitimately trace
    backwards in time (late anchor -> earlier evidence), so list order is NOT
    required to match timestamp order."""
    reasons: list[str] = []
    steps = [s for s in (candidate.get("reasoning_steps") or []) if isinstance(s, dict)]
    if len(steps) < 2:
        return [f"need >=2 reasoning steps, got {len(steps)}"]
    ids: list[str] = []
    for i, s in enumerate(steps):
        sid = str(s.get("step_id") or "").strip()
        if not sid:
            reasons.append(f"steps[{i}] missing step_id")
            sid = f"__missing_{i}"
        ids.append(sid)
    if len(set(ids)) != len(ids):
        reasons.append("duplicate step_id values")

    mods = {str(s.get("modality", "")).lower() for s in steps}
    if not any("audio" in m for m in mods):
        reasons.append("no audio reasoning step")
    if not any("visual" in m for m in mods):
        reasons.append("no visual reasoning step")

    id_pos = {sid: i for i, sid in enumerate(ids)}
    children: dict[str, set[str]] = {sid: set() for sid in ids}
    for i, s in enumerate(steps):
        deps = [str(d) for d in (s.get("depends_on") or [])]
        if i > 0 and not deps:
            reasons.append(f"steps[{i}] ({ids[i]}) has no depends_on — chain is broken")
        for d in deps:
            j = id_pos.get(d)
            if j is None:
                reasons.append(f"steps[{i}] depends on unknown step '{d}'")
            elif j >= i:
                reasons.append(f"steps[{i}] depends on non-earlier step '{d}'")
            else:
                children[d].add(ids[i])

    if ids:
        ancestors: set[str] = set()
        stack = [ids[-1]]
        while stack:
            cur = stack.pop()
            if cur in ancestors:
                continue
            ancestors.add(cur)
            stack.extend(str(d) for d in (steps[id_pos[cur]].get("depends_on") or []))
        dangling = [sid for sid in ids if sid not in ancestors]
        if dangling:
            reasons.append(f"dangling steps not used by the final answer step: {dangling}")
    return reasons


def _norm_ts(ts: str) -> str:
    """Canonical form for timestamp comparison: no spaces, single ASCII dash."""
    t = re.sub(r"\s+", "", str(ts or ""))
    return t.replace("—", "-").replace("–", "-").replace("~", "-").replace("〜", "-")


def _modality_entries(bundle: dict, modality: str) -> list[tuple[str, str, set[str]]]:
    """(norm_ts, norm_caption_string, caption_word_set) per caption entry."""
    key = "audio_captions" if modality == "audio" else "visual_captions"
    out: list[tuple[str, str, set[str]]] = []
    for ev in bundle.get("event_captions", []):
        for it in ev.get(key) or []:
            words = _norm_word_list(it.get("caption", ""))
            out.append((_norm_ts(it.get("timestamp", "")), " ".join(words), set(words)))
    return out


def _quote_match_ts(quote: str, entries: list[tuple[str, str, set[str]]]) -> tuple[set[str], set[str]]:
    """Timestamps of the caption entries the quote most likely comes from.
    Returns (matched_ts, all_ts): verbatim containment first, >=60% word-overlap
    fallback; matched is empty when nothing overlaps enough."""
    q_words = _norm_word_list(quote)
    q = " ".join(q_words)
    all_ts = {ts for ts, _, _ in entries if ts}
    if not q_words:
        return set(), all_ts
    matched: set[str] = set()
    for ts, cap_str, _cap_set in entries:
        if q and q in cap_str:
            matched.add(ts)
    if matched:
        return matched, all_ts
    distinct = set(q_words)
    if len(distinct) >= 4:
        for ts, _cap_str, cap_set in entries:
            hits = sum(1 for w in distinct if w in cap_set)
            if hits / len(distinct) >= 0.6:
                matched.add(ts)
    return matched, all_ts


_GENERIC_TOKENS = {
    "the", "and", "for", "with", "that", "this", "these", "those", "there", "here",
    "man", "woman", "person", "people", "men", "women", "girl", "boy", "guy", "lady",
    "video", "screen", "thing", "something", "someone", "anything", "one", "two",
    "card", "cards", "sound", "sounds", "music", "audio", "visual", "text", "name",
    "says", "said", "say", "shows", "show", "shown", "appears", "appear", "seen",
    "first", "second", "next", "then", "when", "what", "which", "while", "after",
    "before", "later", "earlier", "left", "right", "front", "back", "time",
}

_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")


def _shared_identifiers(quote_a: str, quote_b: str) -> set[str]:
    """Explicit shared identifiers between two clue quotes: latin words (>=3 chars,
    non-generic) or CJK bigrams present in BOTH raw quotes."""
    def latin(q: str) -> set[str]:
        return {w for w in _norm_word_list(q)
                if len(w) >= 3 and w.isascii() and w not in _GENERIC_TOKENS}

    def bigrams(q: str) -> set[str]:
        chars = _CJK_RE.findall(str(q or ""))
        return {"".join(chars[i:i + 2]) for i in range(len(chars) - 1)}

    return (latin(quote_a) & latin(quote_b)) | (bigrams(quote_a) & bigrams(quote_b))


def _audio_anchor_type_reasons(bundle: dict, candidate: dict) -> list[str]:
    """When the AUDIO clue is the anchor (anchor_modality == 'audio'), its quote must
    come from a Speech / identifiable-sound caption entry. Music/BGM/ambience entries
    are unreliable and cannot single out a referent — reject."""
    am = str(candidate.get("anchor_modality") or "").strip().lower()
    if not am:

        fp = str(candidate.get("fusion_pattern") or "").strip().upper()
        am = "audio" if fp == "P1" else ("visual" if fp == "P2" else "")
    if am != "audio":
        return []
    quote = str((candidate.get("audio_clue") or {}).get("quote") or "")
    if not quote.strip():
        return []
    for ev in bundle.get("event_captions", []):
        for it in ev.get("audio_captions") or []:
            words = set(_norm_word_list(it.get("caption", "")))
            q_words = set(_norm_word_list(quote))
            if not q_words:
                continue
            hits = len(q_words & words) / len(q_words)
            if hits >= 0.6 or " ".join(_norm_word_list(quote)) in " ".join(_norm_word_list(it.get("caption", ""))):
                atype = str(it.get("audio_type") or "").lower()
                if any(k in atype for k in ("music", "ambient", "background", "bgm", "song")):
                    return [f"audio anchor comes from a {it.get('audio_type')!r} caption "
                            f"entry (music/BGM is unreliable as an anchor) — anchor on "
                            f"specific SPEECH or an identifiable sound event instead"]
                return []
    return []


def grounding_reject_reasons(bundle: dict, candidate: dict) -> list[str]:
    """Programmatic caption-grounding check (ZERO api cost): the clue quotes the
    generator claims are 'verbatim' must actually exist in the caption bundle, and
    each clue timestamp must EXACTLY MATCH the timestamp of the caption entry the
    quote comes from (fine-grained, e.g. '00:14-00:19' — never a coarse or invented
    range). A hallucinated quote or timestamp is caught BEFORE any VLM call."""
    reasons: list[str] = []
    reasons.extend(_audio_anchor_type_reasons(bundle, candidate))
    try:
        duration = float(bundle.get("video_duration_sec") or 0) or None
    except (TypeError, ValueError):
        duration = None
    for modality, key in (("audio", "audio_clue"), ("visual", "visual_clue")):
        clue = candidate.get(key) or {}
        quote = str(clue.get("quote") or "")
        corpus, corpus_words = _modality_corpus(bundle, modality)
        entries = _modality_entries(bundle, modality)
        matched_ts, all_ts = _quote_match_ts(quote, entries)
        if not quote.strip():
            reasons.append(f"{key}.quote is empty")
        elif not _quote_grounded(quote, corpus, corpus_words):
            reasons.append(f"{key}.quote is NOT in the {modality} captions: "
                           f"\"{quote[:100]}\" — re-quote verbatim")
        clue_ts = _norm_ts(clue.get("timestamp", ""))
        sec = _ts_to_sec(clue.get("timestamp", ""))
        if sec is None:
            reasons.append(f"{key}.timestamp unparseable: {clue.get('timestamp')!r}")
        elif duration and sec > duration + 30:
            reasons.append(f"{key}.timestamp {clue.get('timestamp')!r} exceeds video "
                           f"duration {duration:.0f}s")
        else:


            valid_ts = matched_ts or all_ts
            if valid_ts and clue_ts not in valid_ts:
                hint = sorted(matched_ts)[:3] if matched_ts else []
                reasons.append(
                    f"{key}.timestamp {clue.get('timestamp')!r} does not match any "
                    f"{modality} caption entry — copy the EXACT timestamp of the entry "
                    f"you quote" + (f" (that entry is timestamped {hint})" if hint else ""))
    for i, s in enumerate(candidate.get("reasoning_steps") or []):
        if isinstance(s, dict) and s.get("timestamp") and _ts_to_sec(s["timestamp"]) is None:
            reasons.append(f"reasoning_steps[{i}].timestamp unparseable: {s.get('timestamp')!r}")


    ac = candidate.get("audio_clue") or {}
    vc = candidate.get("visual_clue") or {}
    a_sec, v_sec = _ts_to_sec(ac.get("timestamp", "")), _ts_to_sec(vc.get("timestamp", ""))
    if a_sec is not None and v_sec is not None and abs(a_sec - v_sec) > 15.0:
        shared = _shared_identifiers(str(ac.get("quote", "")), str(vc.get("quote", "")))
        if not shared:
            reasons.append(
                "the two clues are >15s apart but share NO explicit identifier "
                "(same name/label/noun in BOTH quotes) — the referent binding is "
                "unverifiable. Rebuild around a referent that is explicitly named "
                "or distinctively identified in BOTH the audio and the visual quote, "
                "or use co-temporal clues.")


    corpus_a, words_a = _modality_corpus(bundle, "audio")
    corpus_v, words_v = _modality_corpus(bundle, "visual")
    all_entry_ts = {ts for ts, _, _ in
                    (_modality_entries(bundle, "audio") + _modality_entries(bundle, "visual")) if ts}
    presups = candidate.get("presupposition_check")
    if not isinstance(presups, list) or not presups:
        reasons.append("presupposition_check is missing/empty — list every cross-event "
                       "link the question assumes, each with its supporting caption quote")
    else:
        for i, item in enumerate(presups):
            if not isinstance(item, dict):
                reasons.append(f"presupposition_check[{i}] is not an object")
                continue
            if not str(item.get("link") or "").strip():
                reasons.append(f"presupposition_check[{i}].link is empty")
            sb = item.get("supported_by")
            if isinstance(sb, list):
                sb = next((x for x in sb if isinstance(x, dict)), {})
            if isinstance(sb, str):
                sb = {"quote": sb}
            if not isinstance(sb, dict):
                sb = {}
            q = str(sb.get("quote") or "")
            if not q.strip():
                reasons.append(f"presupposition_check[{i}].supported_by.quote is empty")
            elif not (_quote_grounded(q, corpus_a, words_a)
                      or _quote_grounded(q, corpus_v, words_v)):
                reasons.append(f"presupposition_check[{i}] supporting quote is NOT in "
                               f"the captions: \"{q[:80]}\"")
            sb_ts = sb.get("timestamp", "")
            if _ts_to_sec(sb_ts) is None:
                reasons.append(f"presupposition_check[{i}].timestamp unparseable: {sb_ts!r}")
            elif all_entry_ts and _norm_ts(sb_ts) not in all_entry_ts:
                reasons.append(f"presupposition_check[{i}].timestamp {sb_ts!r} is not a "
                               f"caption entry timestamp")
    return reasons


async def _blind_correct(client: OpenAIClient, prompt: str, correct: str,
                         trials: int) -> tuple[int, int, list[str], str]:
    """Return (correct_count, valid_trials, picked_letters, reasoning_excerpt)."""
    outs = await asyncio.gather(*[client.chat(prompt, temperature=0.3, max_tokens=4096,
                                              disable_thinking=True)
                                  for _ in range(trials)])
    correct_count = 0
    valid = 0
    picks: list[str] = []
    reasoning = ""
    for raw in outs:
        res = extract_json(raw)
        if res is None:
            continue
        valid += 1
        pick = extract_option_letter(res.get("selected_option", ""))
        picks.append(pick)
        if pick == correct:
            correct_count += 1
            if not reasoning:
                reasoning = str(res.get("reasoning", ""))[:280]
    return correct_count, valid, picks, reasoning


async def stage_caption_consistency(client: OpenAIClient, bundle: dict, candidate: dict,
                                    caption_max_chars: int = 0) -> tuple[FilterResult, str]:
    """MERGED gate (gold verify + logical self-consistency in ONE call): a strong text
    solver given the FULL captions + question (no gold revealed) must derive the gold
    answer, pass the presupposition audit, and confirm the two clue facts support it.
    Returns (FilterResult, reassigned_gold): reassigned_gold is a new letter when the
    solver consistently derives a DIFFERENT answer that the clue facts support
    (generator mislabel — gold is corrected), else ""."""
    correct = extract_option_letter(candidate["correct_option"])
    ac = candidate.get("audio_clue") or {}
    vc = candidate.get("visual_clue") or {}
    prompt = CAPTION_SOLVER_PROMPT.format(
        caption_bundle_json=build_full_caption_text(bundle, max_chars=caption_max_chars),
        audio_ts=ac.get("timestamp", ""), audio_quote=ac.get("quote", ""),
        audio_fact=ac.get("fact", ""),
        visual_ts=vc.get("timestamp", ""), visual_quote=vc.get("quote", ""),
        visual_fact=vc.get("fact", ""),
        question=candidate["question"], options_text=format_options(candidate["options"]))
    raw = await client.chat(prompt, temperature=0.0, max_tokens=8192)
    res = extract_json(raw)
    if res is None:

        return (FilterResult("caption_consistency", True,
                             {"note": "solver unparseable, skipped", "picks": []}), "")
    pick = extract_option_letter(res.get("selected_option", ""))
    presups = [p for p in (res.get("presuppositions") or []) if isinstance(p, dict)]
    unsupported = [p for p in presups if p.get("supported") is False]
    clue_support = res.get("clue_facts_support")
    details = {"gold": correct, "solver_pick": pick,
               "solver_reasoning": str(res.get("reasoning", ""))[:400],
               "clue_facts_support": clue_support,
               "unsupported_presuppositions": [
                   str(p.get("link", ""))[:150] for p in unsupported][:5],
               "confidence": res.get("confidence", "")}
    if unsupported:
        return FilterResult("caption_consistency", False, details), ""
    if pick == correct:
        details["answer_verified"] = True
        return FilterResult("caption_consistency", True, details), ""
    valid_letters = set(LETTERS[:len(candidate.get("options") or [])])
    if pick in valid_letters and clue_support is not False:


        details["answer_verified"] = True
        details["gold_reassigned"] = f"{correct}->{pick}"
        return FilterResult("caption_consistency", True, details), pick
    return FilterResult("caption_consistency", False, details), ""


async def stage_priors_blind(client: OpenAIClient, candidate: dict, trials: int) -> FilterResult:
    """TEXT-ONLY filter: PASS = a no-video solver (question+options only) FAILS every
    trial. A single correct pick (cc>=1) rejects the question."""
    correct = extract_option_letter(candidate["correct_option"])
    prompt = PRIORS_BLIND_PROMPT.format(
        question=candidate["question"], options_text=format_options(candidate["options"]))
    cc, valid, picks, reasoning = await _blind_correct(client, prompt, correct, trials)
    solvable = cc >= 1
    return FilterResult("priors_blind", not solvable, {"correct_count": cc, "valid_trials": valid,
                                                       "trials": trials, "picks": picks,
                                                       "leak_reasoning": reasoning})


async def stage_verify_answer(client: OpenAIClient, candidate: dict,
                              trials: int = 1) -> tuple[str, dict]:
    """Re-derive the correct option from ONLY the two committed clue facts, ignoring the
    generator's self-labeled `correct_option` (which is systematically mis-assigned).
    Returns (letter, details); "" means abstain (keep original label, flag unverified)."""
    ac = candidate.get("audio_clue") or {}
    vc = candidate.get("visual_clue") or {}
    opts = candidate.get("options") or []
    valid_letters = set(LETTERS[:len(opts)])
    if not (ac.get("fact") and vc.get("fact")) or not opts:
        return "", {"reason": "missing_clue_or_options"}
    prompt = CLUE_GROUNDED_ANSWER_PROMPT.format(
        audio_ts=ac.get("timestamp", ""), audio_quote=ac.get("quote", ""),
        audio_fact=ac.get("fact", ""),
        visual_ts=vc.get("timestamp", ""), visual_quote=vc.get("quote", ""),
        visual_fact=vc.get("fact", ""),
        question=candidate.get("question", ""), options_text=format_options(opts))
    outs = await asyncio.gather(*[client.chat(prompt, temperature=0.0, max_tokens=4096,
                                              disable_thinking=True)
                                  for _ in range(max(1, trials))])
    picks: list[str] = []
    reasoning = ""
    for raw in outs:
        res = extract_json(raw)
        if res is None:
            continue
        pick = extract_option_letter(res.get("selected_option", ""))
        if pick in valid_letters:
            picks.append(pick)
            if not reasoning:
                reasoning = str(res.get("reasoning", ""))[:280]
    if not picks:
        return "", {"reason": "no_valid_pick", "picks": picks}
    winner = max(set(picks), key=picks.count)
    if picks.count(winner) * 2 <= len(picks):
        return "", {"reason": "no_majority", "picks": picks}
    return winner, {"picks": picks, "reasoning": reasoning}


async def stage_select_type(client: OpenAIClient, bundle: dict,
                            caption_max_chars: int = 0) -> dict | None:
    """Select a task from the bundle's evidence; reject unsupported or invalid picks."""
    prompt = TYPE_SELECTION_PROMPT.format(
        type_list=render_type_catalog(),
        caption_bundle_json=build_full_caption_text(bundle, max_chars=caption_max_chars))


    res = extract_json(await client.chat(prompt, temperature=0.3, max_tokens=16384))
    if not isinstance(res, dict) or res.get("reject"):
        return None
    tid = str(res.get("task_type_id", "")).strip()
    t = TASK_TYPE_BY_ID.get(tid)
    if t:
        return dict(t, selection_reason=str(res.get("reason", ""))[:300])
    return None


async def stage_generate(client: OpenAIClient, bundle: dict, task_type: dict,
                         attempt: int,
                         previous_questions: list[str] | None, last_fail: str | None,
                         extra_feedback: str = "", num_options: int = 8,
                         caption_max_chars: int = 0) -> dict | None:


    task_section = render_task_section(task_type)
    structure_rules = STRUCTURE_RULES[task_type["structure"]]
    task_type_label = task_type["name"]
    prompt = GENERATION_PROMPT.format(
        task_section=task_section,
        structure_rules=structure_rules,
        caption_bundle_json=build_full_caption_text(bundle, max_chars=caption_max_chars),
        video_id=bundle.get("video_id", "unknown"),
        num_options=num_options,
        task_type_label=task_type_label)
    if last_fail and last_fail in REGEN_FEEDBACK:
        prompt += REGEN_FEEDBACK[last_fail]
    if extra_feedback:
        prompt += extra_feedback
    if previous_questions:
        prev = "\n".join(f"- {q}" for q in previous_questions[-3:])
        prompt += f"\n\nDo NOT repeat the focus of these already-accepted questions:\n{prev}"
    raw = await client.chat(prompt, temperature=0.8 if attempt > 1 else 0.7, max_tokens=32768)
    return extract_json(raw)


async def stage_self_critique(client: OpenAIClient, bundle: dict, candidate: dict,
                              num_options: int = 8,
                              caption_max_chars: int = 0) -> tuple[dict, bool, list[str]]:
    """Red-team the draft and return a (possibly repaired) candidate. On any error
    the original candidate is returned unchanged."""
    prompt = SELF_CRITIQUE_PROMPT.format(
        caption_bundle_json=build_full_caption_text(bundle, max_chars=caption_max_chars),
        candidate_json=json.dumps(candidate, ensure_ascii=False, indent=1))
    res = extract_json(await client.chat(prompt, temperature=0.4, max_tokens=32768))
    if not res or not isinstance(res.get("candidate"), dict):
        return candidate, False, []
    new_cand = res["candidate"]
    if not new_cand.get("question") or len(new_cand.get("options") or []) != num_options or not new_cand.get("correct_option"):
        return candidate, False, list(res.get("breaks_found") or [])
    new_cand.setdefault("video_id", candidate.get("video_id"))
    new_cand.setdefault("task_type", candidate.get("task_type", ""))
    return new_cand, bool(res.get("revised")), list(res.get("breaks_found") or [])


def parse_time_gap(candidate: dict) -> float:
    a = _ts_to_sec((candidate.get("audio_clue") or {}).get("timestamp", ""))
    v = _ts_to_sec((candidate.get("visual_clue") or {}).get("timestamp", ""))
    if a is None or v is None:
        return -1.0
    return abs(a - v)


_GATE_FEEDBACK = {
    "priors_blind": (" The text-only solver answered correctly WITHOUT any video; its "
                     "reasoning was: \"{leak}\". Remove exactly this leak — make the "
                     "answer an arbitrary video-internal value, not derivable from "
                     "general knowledge or wording cues."),
    "caption_consistency": (" A strong solver GIVEN the full captions could NOT derive "
                            "your gold answer (it picked a different option); its "
                            "reasoning was: \"{leak}\". The question is not logically "
                            "entailed by the caption evidence — fix the question, "
                            "options, or clues so the gold follows necessarily from "
                            "the captions."),
}


async def process_one_question(client: OpenAIClient, bundle: dict, video_id: str,
                               max_attempts: int, blind_trials: int, do_critique: bool,
                               num_options: int,
                               variant_idx: int = 0, num_variants: int = 1,
                               gate_client: OpenAIClient | None = None,
                               verify_answer: bool = True, gold_trials: int = 1,
                               caption_max_chars: int = 0) -> CandidateResult:
    """Run ONE question's full pipeline:
    content-based task selection -> gen -> [critique on retry attempts]
    -> programmatic guards -> merged solver gate (gold + consistency) -> priors gate.
    Skip the question slot if the selector cannot identify a supported task."""
    gate_client = gate_client or client
    result = CandidateResult(video_id=video_id)
    last_fail: str | None = None
    extra_feedback: str = ""
    if num_variants > 1:
        extra_feedback = (f" You are writing question variant {variant_idx + 1} of {num_variants} "
                          f"for this video; anchor it on a DIFFERENT event / entity / moment than "
                          f"the other variants would naturally pick, to maximize diversity.")


    task_type = await stage_select_type(client, bundle, caption_max_chars)
    if task_type is None:
        result.generation_status = "type_selection_failed"
        result.final_status = "rejected"
        result.filters.append(FilterResult(
            "type_selection", False, {"reason": "No supported task type selected"}))
        return result
    result.task_type_id = task_type["id"]
    result.type_selected_by = "llm"
    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        gen = await stage_generate(client, bundle, task_type, attempt,
                                   None, last_fail, extra_feedback, num_options,
                                   caption_max_chars)
        extra_feedback = ""
        if gen is None:
            last_fail = "structure"
            result.generation_status = f"failed_attempt_{attempt}"
            continue
        candidate = gen.get("candidate") or gen


        if do_critique and attempt > 1:
            candidate, revised, breaks = await stage_self_critique(gate_client, bundle, candidate,
                                                                   num_options, caption_max_chars)
            result.revised = revised
            result.breaks_found = breaks


        opts = candidate.get("options") or []
        correct_letter = resolve_correct_letter(candidate)
        if (not candidate.get("question") or not opts
                or correct_letter not in set(LETTERS[:len(opts)])):
            result.candidate = candidate
            result.generation_status = f"malformed_attempt_{attempt}"
            last_fail = "structure"
            continue
        candidate["correct_option"] = correct_letter


        fp_raw = str(candidate.get("fusion_pattern") or "").strip().upper()
        if fp_raw in ("P1", "P2"):
            candidate["fusion_pattern"] = fp_raw
        elif "JOINT" in fp_raw:
            candidate["fusion_pattern"] = "joint"
        am = str(candidate.get("anchor_modality") or "").strip().lower()
        if am not in ("audio", "visual", "n/a"):
            candidate["anchor_modality"] = {"P1": "audio", "P2": "visual"}.get(
                candidate["fusion_pattern"], "")


        rg_reasons = reasoning_graph_reasons(candidate)
        if rg_reasons and all("has no depends_on" in r or "missing step_id" in r
                              for r in rg_reasons):
            steps = [s for s in (candidate.get("reasoning_steps") or []) if isinstance(s, dict)]
            for i, s in enumerate(steps):
                if not str(s.get("step_id") or "").strip():
                    s["step_id"] = f"s{i + 1}"
                if i > 0 and not (s.get("depends_on") or []):
                    s["depends_on"] = [steps[i - 1].get("step_id", f"s{i}")]
            candidate["reasoning_graph_repaired"] = True
            rg_reasons = reasoning_graph_reasons(candidate)
        if rg_reasons:
            result.candidate = candidate
            result.filters.append(FilterResult("reasoning_graph", False, {"reasons": rg_reasons}))
            result.generation_status = f"no_reasoning_steps_attempt_{attempt}"
            last_fail = "reasoning"
            extra_feedback = ("\n\nThe reasoning-graph check failed on your previous "
                              "draft:\n" + "\n".join(f"- {r}" for r in rg_reasons))
            continue
        result.filters.append(FilterResult(
            "reasoning_graph", True,
            {"repaired": bool(candidate.get("reasoning_graph_repaired"))}))


        g_reasons = grounding_reject_reasons(bundle, candidate)
        if g_reasons:
            result.candidate = candidate
            result.filters.append(FilterResult("grounding", False, {"reasons": g_reasons}))
            result.generation_status = f"ungrounded_attempt_{attempt}"
            last_fail = "grounding"
            extra_feedback = ("\n\nThe caption-grounding check failed on your previous "
                              "draft:\n" + "\n".join(f"- {r}" for r in g_reasons))
            continue
        result.filters.append(FilterResult("grounding", True, {}))


        gap = parse_time_gap(candidate)
        if needs_simultaneity(candidate.get("question", "")) and gap > 15.0:
            result.candidate = candidate
            result.filters.append(FilterResult(
                "simultaneity", False,
                {"time_gap_seconds": gap,
                 "reason": f"question asks about a momentary state but clues are {gap:.0f}s apart"}))
            result.generation_status = f"simultaneity_mismatch_attempt_{attempt}"
            last_fail = "simultaneity"
            continue
        result.filters.append(FilterResult("simultaneity", True, {"time_gap_seconds": gap}))

        candidate["task_type"] = task_type["name"]
        candidate["task_type_id"] = task_type["id"]
        if task_type.get("selection_reason"):
            candidate["type_selection_reason"] = task_type["selection_reason"]
        candidate["time_gap_seconds"] = gap


        solver_result, new_gold = (FilterResult("caption_consistency", True, {"note": "disabled"}), "")
        if verify_answer:
            solver_result, new_gold = await stage_caption_consistency(
                gate_client, bundle, candidate, caption_max_chars)
            if new_gold:
                candidate["answer_verified"] = True
                candidate["correct_option_original"] = candidate.get("correct_option")
                candidate["correct_option"] = new_gold
                candidate["answer_reassigned"] = True
            elif solver_result.details.get("answer_verified"):
                candidate["answer_verified"] = True
            candidate["gold_check"] = solver_result.details
        result.filters.append(solver_result)

        candidate = shuffle_options(candidate)
        result.candidate = candidate
        result.generation_status = "generated"

        gate_results = [solver_result]
        if solver_result.passed:

            priors_result = await stage_priors_blind(gate_client, candidate, blind_trials)
            result.filters.append(priors_result)
            gate_results.append(priors_result)

        first_fail = next((res for res in gate_results if not res.passed), None)
        if first_fail is not None:
            last_fail = first_fail.stage
            leak = (first_fail.details.get("leak_reasoning")
                    or first_fail.details.get("solver_reasoning") or "")
            extra_feedback = _GATE_FEEDBACK.get(first_fail.stage, "").format(leak=leak) if leak else ""
            continue

        result.final_status = "accepted"
        break

    if result.final_status == "pending":
        result.final_status = "exhausted_attempts"
    return result


async def process_video(client: OpenAIClient, bundle: dict, max_attempts: int,
                        questions_per_video: int, blind_trials: int,
                        do_critique: bool, num_options: int,
                        gate_client: OpenAIClient | None,
                        verify_answer: bool, gold_trials: int,
                        caption_max_chars: int = 0) -> list[CandidateResult]:
    video_id = bundle.get("video_id", "unknown")
    questions = [
        process_one_question(
            client, bundle, video_id, max_attempts, blind_trials, do_critique,
            num_options, variant_idx=i, num_variants=questions_per_video,
            gate_client=gate_client, verify_answer=verify_answer,
            gold_trials=gold_trials, caption_max_chars=caption_max_chars)
        for i in range(questions_per_video)
    ]
    return list(await asyncio.gather(*questions))



def expand_raw_dirs(spec: str) -> list[Path]:
    """--caption-raw-dir accepts comma-separated dirs AND glob patterns."""
    out: list[Path] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        matches = sorted(globmod.glob(part)) if globmod.has_magic(part) else [part]
        out.extend(Path(m) for m in matches)

    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        k = str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def load_caption_bundles(caption_raw_dirs: list[Path], limit_per_dir: int,
                         only_ids: set[str] | None = None) -> list[tuple[Path, str]]:
    """Load caption bundles with a per-directory limit.

    Add a path-hash suffix when dataset names occur beneath different roots."""
    dir_sources: list[tuple[Path, str]] = []
    for d in caption_raw_dirs:
        if not d.exists():
            print(f"WARNING: caption raw dir not found, skipping: {d}", file=sys.stderr)
            continue
        dir_sources.append((d, d.parent.name if d.name == "raw" else d.name))
    name_counts: dict[str, int] = {}
    for _, s in dir_sources:
        name_counts[s] = name_counts.get(s, 0) + 1

    out: list[tuple[Path, str]] = []
    for d, source in dir_sources:
        if name_counts[source] > 1:
            suffix = hashlib.sha1(str(d).encode()).hexdigest()[:6]
            source = f"{source}::{suffix}"


        import os as _os
        from concurrent.futures import ThreadPoolExecutor as _TPE
        try:
            names = sorted(e.name for e in _os.scandir(d))
        except OSError:
            names = sorted(p.name for p in d.iterdir())
        if only_ids is not None:
            names = [n for n in names if n in only_ids]


        def _has(name):
            cb = d / name / "caption_bundle.json"
            try:
                return cb if cb.exists() else None
            except OSError:
                return None
        paths = []
        _CHUNK = 4096
        with _TPE(max_workers=64) as _ex:
            for _i in range(0, len(names), _CHUNK):
                for r in _ex.map(_has, names[_i:_i + _CHUNK], chunksize=32):
                    if r is not None:
                        paths.append(r)
                if 0 < limit_per_dir <= len(paths):
                    paths = paths[:limit_per_dir]
                    break
        print(f"  {source}: {len(paths)} bundles", flush=True)
        out.extend((p, source) for p in paths)
    print(f"Found {len(out)} caption bundles across {len(dir_sources)} source(s)", flush=True)
    return out


def load_bundle(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("event_captions"):
            return data
    except Exception:
        pass
    return None


BUNDLE_META_KEYS = ("raw_video_local_path", "time_video_local_path",
                    "time_video_oss_path", "video_duration_sec")


def resolve_video_path_from_bundle(bp: Path, bundle: dict,
                                   map_by_pair: dict | None = None,
                                   map_by_id: dict | None = None,
                                   source: str = "") -> str:
    """Best-effort absolute path to a LOCAL video file for this bundle.
    The prebuilt caption<->video map (build_video_map.py) wins; then
    raw_video_local_path (usually absolute); then time_video_local_path (absolute,
    next to the bundle, or repo-root-relative)."""
    vid = str(bundle.get("video_id") or bp.parent.name)
    for p in ((map_by_pair or {}).get((source, vid)), (map_by_id or {}).get(vid)):
        if p:
            try:
                if Path(p).exists():
                    return p
            except OSError:
                pass
    candidates: list[Path] = []
    rp = str(bundle.get("raw_video_local_path") or "")
    if rp:
        candidates.append(Path(rp))
    tp = str(bundle.get("time_video_local_path") or "")
    if tp:
        p = Path(tp)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(bp.parent / p.name)
            if len(bp.parents) > 3:
                candidates.append(bp.parents[3] / p)
    for c in candidates:
        try:
            if c.exists():
                return str(c)
        except OSError:
            continue
    return ""


async def run_pipeline(args: argparse.Namespace) -> int:
    client = OpenAIClient(api_key=args.openai_api_key, api_base=args.openai_api_base,
                          model=args.openai_model, qps=args.api_qps, max_inflight=args.max_inflight)
    gate_model = (args.gate_model or "").strip()
    if gate_model and gate_model != args.openai_model:
        gate_client = OpenAIClient(
            api_key=args.gate_api_key or args.openai_api_key,
            api_base=args.gate_api_base or args.openai_api_base,
            model=gate_model, qps=args.gate_api_qps or args.api_qps,
            max_inflight=args.gate_max_inflight or args.max_inflight)
        print(f"Gate judge model: {gate_model} (gen model: {args.openai_model})", flush=True)
    else:
        gate_client = client
    print(f"Content-based task selection from {len(TASK_TYPES)} reasoning categories", flush=True)
    args._map_by_pair, args._map_by_id = load_video_map(args.video_map)
    if args._map_by_id:
        print(f"Video map: {len(args._map_by_pair)} mapped entries from {args.video_map}", flush=True)

    raw_dirs = expand_raw_dirs(args.caption_raw_dir)
    only_ids: set[str] | None = None
    if args.only_video_ids:
        p = Path(args.only_video_ids)
        ids: set[str] = set()
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("{"):
                try:
                    ids.add(str(json.loads(ln).get("video_id", "")).strip())
                except Exception:
                    pass
            else:
                ids.add(ln)
        ids.discard("")
        only_ids = ids
        print(f"only-video-ids filter: {len(only_ids)} ids from {p}", flush=True)
    bundle_paths = load_caption_bundles(raw_dirs, args.num_videos, only_ids)
    if not bundle_paths:
        return 1
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = out_dir / "accepted.jsonl"
    all_path = out_dir / "all.jsonl"

    done: set[str] = set()
    if args.resume and all_path.exists():
        for line in all_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done.add(f"{row.get('source', '')}::{row['video_id']}")
            except Exception:
                pass
        bundle_paths = [(bp, src) for (bp, src) in bundle_paths
                        if f"{src}::{bp.parent.name}" not in done]
        print(f"Resume: {len(done)} videos already processed, {len(bundle_paths)} remaining", flush=True)

    sem = asyncio.Semaphore(args.concurrency)

    async def process_with_sem(bp: Path, source: str):
        async with sem:
            bundle = load_bundle(bp)
            if bundle is None:
                return [], None

            bundle["_resolved_video_path"] = resolve_video_path_from_bundle(
                bp, bundle, args._map_by_pair, args._map_by_id, source.split("::")[0])
            try:
                res = await process_video(client, bundle, args.max_attempts,
                                          args.questions_per_video, args.blind_trials,
                                          not args.no_critique, args.num_options,
                                          gate_client, not args.no_verify_answer,
                                          args.gold_trials, args.caption_max_chars)
            except Exception as e:
                vid = bundle.get("video_id", bp.parent.name)
                print(f"  [warn] video {vid} failed: {type(e).__name__}: {e}", flush=True)
                r = CandidateResult(video_id=vid, source=source,
                                    final_status="error", generation_status=f"error: {type(e).__name__}")
                return [r], bundle
            for r in res:
                r.source = source
            return res, bundle

    tasks = [process_with_sem(bp, src) for (bp, src) in bundle_paths]
    pbar = tqdm(total=len(tasks), desc="Videos", unit="vid")
    accepted_f = open(accepted_path, "a", encoding="utf-8")
    all_f = open(all_path, "a", encoding="utf-8")

    all_results: list[CandidateResult] = []
    gate_fail = {"structure": 0, "reasoning": 0, "grounding": 0, "simultaneity": 0,
                 "caption_consistency": 0, "priors_blind": 0}
    n_accepted = 0
    n_revised = 0

    for coro in asyncio.as_completed(tasks):
        vresults, bundle = await coro
        all_results.extend(vresults)
        meta = {k: bundle.get(k) for k in BUNDLE_META_KEYS} if bundle else {}
        if bundle:
            meta["video_path"] = bundle.get("_resolved_video_path", "")
        for r in vresults:
            rec = json.dumps({
                "video_id": r.video_id, "source": r.source, "final_status": r.final_status,
                "generation_status": r.generation_status, "attempts": r.attempts,
                "revised": r.revised, "breaks_found": r.breaks_found,
                "task_type_id": r.task_type_id,
                "task_type": (r.candidate or {}).get("task_type", ""),
                "type_selected_by": r.type_selected_by,
                **meta,
                "candidate": r.candidate,
                "filters": [{"stage": f.stage, "passed": f.passed, "details": f.details} for f in r.filters],
            }, ensure_ascii=False)
            all_f.write(rec + "\n")
            if r.revised:
                n_revised += 1
            if r.final_status == "accepted":
                n_accepted += 1
                accepted_f.write(rec + "\n")
            else:
                failed = [f.stage for f in r.filters if not f.passed]
                if failed:

                    gate_fail[failed[-1]] = gate_fail.get(failed[-1], 0) + 1
                elif r.generation_status.startswith(("failed", "malformed")):
                    gate_fail["structure"] += 1
                elif "no_reasoning_steps" in r.generation_status:
                    gate_fail["reasoning"] = gate_fail.get("reasoning", 0) + 1
                elif "ungrounded" in r.generation_status:
                    gate_fail["grounding"] = gate_fail.get("grounding", 0) + 1
                elif "simultaneity_mismatch" in r.generation_status:
                    gate_fail["simultaneity"] = gate_fail.get("simultaneity", 0) + 1
        accepted_f.flush(); all_f.flush()
        pbar.update(1)
        pbar.set_postfix(ok=n_accepted, refresh=True)
    pbar.close(); accepted_f.close(); all_f.close()

    total = max(1, len(all_results))
    pos: dict[str, int] = {}
    fusion: dict[str, int] = {}
    per_source: dict[str, dict[str, int]] = {}
    type_actual: dict[str, int] = {}
    clue_counts: list[int] = []
    gaps: list[float] = []
    for r in all_results:
        src = r.source or "?"
        ps = per_source.setdefault(src, {"slots": 0, "accepted": 0})
        ps["slots"] += 1
        if r.final_status == "accepted" and r.candidate:
            ps["accepted"] += 1
            c = extract_option_letter(r.candidate.get("correct_option", ""))
            pos[c] = pos.get(c, 0) + 1
            fp = str(r.candidate.get("fusion_pattern", "?"))
            fusion[fp] = fusion.get(fp, 0) + 1
            tid = str(r.candidate.get("task_type_id", r.task_type_id or "?"))
            type_actual[tid] = type_actual.get(tid, 0) + 1
            clue_counts.append(len(r.candidate.get("reasoning_steps") or []))
            g = r.candidate.get("time_gap_seconds")
            if isinstance(g, (int, float)) and g >= 0:
                gaps.append(float(g))
    for src, ps in per_source.items():
        s = max(1, ps["slots"])
        ps["acceptance_rate"] = f"{ps['accepted']}/{ps['slots']} ({100*ps['accepted']/s:.1f}%)"

    def _avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else None
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.openai_model,
        "gate_model": (args.gate_model or args.openai_model),
        "type_selection": "content",
        "total_question_slots": len(all_results),
        "accepted": n_accepted,
        "num_options": args.num_options,
        "acceptance_rate": f"{n_accepted}/{len(all_results)} ({100*n_accepted/total:.1f}%)",
        "per_source": per_source,
        "accepted_avg_clue_count": _avg(clue_counts),
        "accepted_avg_time_gap_seconds": _avg(gaps),
        "revised_by_critique": n_revised,
        "blind_trials": args.blind_trials,
        "enabled_blind_gates": ["caption_consistency", "priors"],
        "max_attempts": args.max_attempts,
        "self_critique": not args.no_critique,
        "gate_fail_attribution": gate_fail,
        "accepted_answer_position": pos,
        "accepted_fusion_pattern": fusion,
        "accepted_task_type_distribution": type_actual,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*60}\nSUMMARY", flush=True)
    print(f"  accepted: {n_accepted}/{len(all_results)} ({100*n_accepted/total:.1f}%)", flush=True)
    print(f"  revised_by_critique: {n_revised}", flush=True)
    print(f"  gate_fail_attribution: {gate_fail}", flush=True)
    print(f"  accepted_answer_position: {pos}", flush=True)
    print(f"  accepted_task_type_distribution: {type_actual}", flush=True)
    print(f"  accepted -> {accepted_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Content-driven audio-visual QA generator")
    p.add_argument("--caption-raw-dir", required=True,
                   help="comma-separated caption raw dirs or glob patterns; --num-videos applies PER dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--openai-api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--openai-model", default="deepseek-v4-flash")
    p.add_argument("--gate-model", default="",
                   help="optional separate judge model for the priors gate + gold verify. Empty = use --openai-model.")
    p.add_argument("--gate-api-base", default="")
    p.add_argument("--gate-api-key", default="")
    p.add_argument("--gate-api-qps", type=float, default=0.0)
    p.add_argument("--gate-max-inflight", type=int, default=0)
    p.add_argument("--caption-max-chars", type=int, default=0,
                   help="truncate the caption context to this many chars; 0 = NO truncation "
                        "(default: full summary + audio/visual captions are sent)")
    p.add_argument("--num-videos", type=int, default=0)
    p.add_argument("--questions-per-video", type=int, default=1)
    p.add_argument("--num-options", type=int, default=8, help="number of answer options per question (max 8)")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="generation attempts per question; on a text-only leak the leak reasoning is fed back")
    p.add_argument("--blind-trials", type=int, default=2,
                   help="trials for the text-only (priors) gate; leak iff correct in ANY trial (cc>=1)")
    p.add_argument("--no-critique", action="store_true", help="disable the self-critique / red-team repair pass")
    p.add_argument("--no-verify-answer", action="store_true",
                   help="disable the clue-grounded gold verification; ON by default")
    p.add_argument("--gold-trials", type=int, default=1)
    p.add_argument("--only-video-ids", default="",
                   help="optional file of video ids (one per line, or jsonl rows with a video_id field)")
    p.add_argument("--video-map", default=str(Path("data/video_map.jsonl")),
                   help="caption<->video map built by build_video_map.py (used for the video_path meta if present)")
    p.add_argument("--api-qps", type=float, default=15.0)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--max-inflight", type=int, default=128)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.openai_api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 1
    if not (2 <= args.num_options <= len(LETTERS)):
        print(f"ERROR: --num-options must be in [2, {len(LETTERS)}]", file=sys.stderr)
        return 1
    return asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    raise SystemExit(main())
