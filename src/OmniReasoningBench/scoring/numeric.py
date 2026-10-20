"""Numeric fallback used only when the LLM judge cannot return a verdict."""
import re

_FINAL_RE = re.compile(r"final\s*answer\s*[:：]\s*(.+)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _parse_number(text: str):
    if not text:
        return None
    m = _FINAL_RE.findall(text)
    seg = m[-1] if m else text
    nums = _NUM_RE.findall(seg.replace(",", ""))
    if not nums and not m:
        nums = _NUM_RE.findall(text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None

def _numeric_correct(gold: str, pred_text: str, rel_tol: float = 0.01) -> bool:
    g = _parse_number(gold)
    p = _parse_number(pred_text)
    if g is None or p is None:
        return False
    if g == 0:
        return abs(p) < 1e-6
    return abs(p - g) <= max(rel_tol * abs(g), 1e-6)
