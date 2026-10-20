"""Read scoring outcomes consistently across benchmark reports."""
from OmniReasoningBench.scoring.judge import JUDGE_MODEL


FAILURE_METHODS = {"error", "judge_failed", "no_answer_truncated"}


def outcome(row: dict) -> tuple[float, int]:
    """Return correctness and failure status from a scored row.

    Open-ended scores must come from the LLM-first scorer, including its recorded
    numeric fallback. Legacy MCQ scores may still use the `acc` field.
    """
    method = row.get("method")
    if row.get("answer_format") != "mcq":
        allowed = {"llm_judge", *FAILURE_METHODS}
        if row.get("answer_format") == "open_numeric":
            allowed.add("rule_numeric_fallback")
        if method not in allowed:
            raise ValueError("Rescore open-ended predictions with omnireasoning score-judge")
        if row.get("judge_model", JUDGE_MODEL) != JUDGE_MODEL:
            raise ValueError(f"Open-ended reports require {JUDGE_MODEL}")
    failed = method in FAILURE_METHODS or bool(row.get("error")) or bool(row.get("failed"))
    if failed:
        return 0.0, 1
    if "correct" in row:
        if not isinstance(row["correct"], bool):
            raise ValueError("The scored 'correct' field must be a boolean")
        return float(row["correct"]), 0
    if row.get("answer_format") != "mcq":
        raise ValueError("Open-ended score-judge output must include 'correct'")
    return float(float(row.get("acc") or 0.0) >= 1.0), 0
