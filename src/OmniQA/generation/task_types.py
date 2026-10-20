"""Reasoning categories and evidence requirements for content-based question planning."""
from __future__ import annotations


TASK_TYPES: list[dict] = [
    {
        "id": "counting",
        "name": "Cross-modal Counting",
        "structure": "anchor",
        "guidance": (
            "CROSS-MODAL COUNTING. The answer is an integer count of occurrences — "
            "spoken mentions of an item, visible repetitions of one action/event, "
            "on-screen objects of a category, or occurrences of a sound. BOTH "
            "modalities are required to pin down WHICH thing is being counted: one "
            "modality names/identifies the target (or the scene interval), the other "
            "holds the actual occurrences, and the number itself is NEVER spoken. "
            "Covers object-, action-, and audio-counting. Options are distinct "
            "integers / count phrases of identical granularity, all plausible for "
            "this video."),
    },
    {
        "id": "spatial",
        "name": "Spatial Grounding & Localization",
        "structure": "anchor",
        "guidance": (
            "SPATIAL GROUNDING & LOCALIZATION. Ask WHERE a referent is on screen "
            "(left/right, in hand, background …), or the spatial relation (left of / "
            "behind / above / holding …) between two referents. The referent is "
            "identified ONLY through audio — a spoken mention, the speaker of a line, "
            "or a distinctive non-speech sound whose source must be found — while the "
            "location / relation is visual-only. Covers speaker localization, "
            "sounding-object grounding, and object-object spatial relations; needs "
            ">=2 visually distinct candidates. Options are distinct locations / "
            "relations of identical granularity."),
    },
    {
        "id": "attribute",
        "name": "Attribute & Fine-grained Perception",
        "structure": "anchor",
        "guidance": (
            "ATTRIBUTE & FINE-GRAINED PERCEPTION. Ask a visual attribute or fine "
            "detail — colour, size, material, shape, position, gesture, on-screen "
            "text/number, a chart/diagram value, a state, or how a state CHANGES — of "
            "the referent that the AUDIO singles out. The attribute/detail is "
            "visual-only and never spoken; the referent selection is audio-only. May "
            "compare two referents on one attribute (one spoken, one shown) or infer "
            "an unobservable attribute (material / purpose / quality) from a visible "
            "property plus a spoken statement. Neither modality alone suffices."),
    },
    {
        "id": "sound_event",
        "name": "Sound-Source & Event Recognition",
        "structure": "anchor",
        "guidance": (
            "SOUND-SOURCE & EVENT RECOGNITION. Identify WHAT is happening — the "
            "visible event, the sound / music / instrument heard, or the kind of "
            "place / scene — where one modality marks or disambiguates the other. "
            "Typical forms: what makes a described sound (sound audio-only, source "
            "event visible); what event occurs at a moment marked by a distinctive "
            "sound or spoken cue; what is heard during a visually-identified scene; "
            "what / where the scene is when the visuals are ambiguous and the audio "
            "(ambient sound, announcement, dialogue) disambiguates, or vice versa. "
            "The identifying cue lives in one modality, the answer in the other."),
    },
    {
        "id": "temporal",
        "name": "Temporal Reasoning (order & when)",
        "structure": "anchor",
        "guidance": (
            "TEMPORAL REASONING. Reason over the mixed-modality timeline: either "
            "ORDER 3-4 events / items (some seen, some heard, some spoken) into the "
            "correct chronological sequence, or LOCALISE when a described "
            "audio+visual plot/event happens (which event / timestamp). No single "
            "modality contains all the endpoints, and the events must be clearly "
            "ordered or far apart in the captions. For ordering, options are order "
            "permutations of the same events; for localisation, options are "
            "distinct timestamps / event ids from this video."),
    },
    {
        "id": "prediction",
        "name": "Prediction & Continuation",
        "structure": "anchor",
        "guidance": (
            "PREDICTION & CONTINUATION. Given an audio+visual setup, ask what happens "
            "NEXT / right after an AV event, or how a developing cross-event "
            "situation resolves — including first-person (egocentric) intent or next "
            "action when the bundle is egocentric. The setup spans both modalities; "
            "the continuation is observable only in one modality at a later timestamp "
            "(or, for cross-event plots, in a later event). The answer must follow "
            "from video-internal evidence, not general world knowledge."),
    },
    {
        "id": "causal",
        "name": "Causal & Counterfactual Reasoning",
        "structure": "anchor",
        "guidance": (
            "CAUSAL & COUNTERFACTUAL REASONING. Ask WHY an observable event / "
            "reaction happened, or what would most likely happen if a video event "
            "had NOT occurred. The cause lives in ONE modality and the effect in the "
            "OTHER (e.g. a sound / spoken line explains a later visible reaction). "
            "The premise must be an AV event from THIS video; the answer is a single "
            "video-internal cause / consequence — not general world knowledge — and "
            "distractors are other plausible video-internal causes."),
    },
    {
        "id": "emotion",
        "name": "Emotion & Sentiment",
        "structure": "anchor",
        "guidance": (
            "EMOTION & SENTIMENT. Ask a person's emotion / attitude, how it CHANGES "
            "between two far-apart moments, or the overall emotional tone / "
            "trajectory of the video. The target person is singled out by a "
            "VISUAL-only property (appearance / position / an appearance trait — "
            "NEVER a spoken name), and the emotion is carried by vocal tone / word "
            "choice (audio); overall-tone variants fuse music + visuals + narration. "
            "Both modalities are required. Options are distinct emotion labels, "
            "'X to Y' transitions, or trajectories."),
    },
    {
        "id": "music",
        "name": "Music & Ambient Audio Function",
        "structure": "holistic",
        "guidance": (
            "MUSIC & AMBIENT AUDIO FUNCTION. Built on NON-SPEECH audio (a Music / "
            "Ambient audio_caption entry): ask how the music / background sound "
            "relates to what is shown — mood / atmosphere match, its narrative or "
            "emotional FUNCTION across the video (e.g. a recurring sound effect), or "
            "how the soundscape CHANGES at a visually-marked transition. BOTH the "
            "audio character and the visuals are required; the question must not be "
            "answerable from the music alone or the visuals alone, and must not hint "
            "the answer. Do NOT build this on speech."),
    },
    {
        "id": "relationship",
        "name": "Relationship & Interaction",
        "structure": "holistic",
        "guidance": (
            "RELATIONSHIP & INTERACTION. Infer the social / functional relationship "
            "between two people from how they address each other (dialogue / audio) "
            "AND how they interact on screen (visual), or identify what one person "
            "does to / with another person or an object — where the person is "
            "distinguished by voice / dialogue and the action / target is visible. "
            "Characters must recur or be clearly distinguishable; the relationship "
            "must not be stated outright in either modality alone."),
    },
    {
        "id": "consistency",
        "name": "Cross-modal Consistency & Existence",
        "structure": "anchor",
        "guidance": (
            "CROSS-MODAL CONSISTENCY & EXISTENCE. Cross-check the spoken track "
            "against the visual track: ask which MENTIONED object / sound actually "
            "appears (or which is mentioned-but-never-shown), which listed statement "
            "is FALSE for this video, or what is anomalous — an audio-visual mismatch "
            "(a sound that doesn't fit the picture, or a claim that contradicts what "
            "is shown). Requires genuinely comparing the spoken mentions with the "
            "visual captions. Distractors are things the narration / context suggests "
            "but the video never shows / never says. Physical objects only for "
            "existence probes (not names / professions)."),
    },
    {
        "id": "reference",
        "name": "Cross-modal Reference & Detail Binding",
        "structure": "anchor",
        "guidance": (
            "CROSS-MODAL REFERENCE & DETAIL BINDING. Resolve which referent a spoken "
            "mention points to among several visually-similar candidates (or vice "
            "versa), or recall a precise detail by BINDING a spoken mention to a "
            "visible circumstance, possibly in a different, far-apart event (e.g. "
            "'where was the boy when someone mentioned watermelon?'; 'which earlier "
            "line explains what X does here?'). Do NOT ask the element directly — the "
            "binding key is a concrete named entity in one modality, and the "
            "disambiguating / answer property is exclusive to the other."),
    },
    {
        "id": "summary",
        "name": "Global Understanding & Summarization",
        "structure": "holistic",
        "guidance": (
            "GLOBAL UNDERSTANDING & SUMMARIZATION. Ask for the best summary / main "
            "topic / purpose of the WHOLE video, where BOTH the narration and the "
            "visuals are needed to pick the right one. Distractors are summaries that "
            "fit only the audio, only the visuals, or a different video of the same "
            "genre."),
    },
]


TASK_TYPE_BY_ID: dict[str, dict] = {t["id"]: t for t in TASK_TYPES}


def render_type_catalog() -> str:
    """Describe the available reasoning operations for the video-content selector."""
    return "\n\n".join(
        f"### id: {t['id']}  —  {t['name']}  [{t['structure']}]\n{t['guidance']}"
        for t in TASK_TYPES
    )


def render_task_section(task_type: dict) -> str:
    """Render the selected task's evidence requirements without external examples."""
    return "\n".join([
        "## SELECTED QUESTION TYPE (must follow)",
        f"Type: {task_type['name']}",
        task_type["guidance"],
    ])
