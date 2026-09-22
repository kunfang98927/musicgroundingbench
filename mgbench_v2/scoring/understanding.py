"""Rule-based scoring of v2 understanding answers (objective answers -> no LLM judge needed).

score(pred_text, rec, vocab) -> (correct: bool, parsed: object|None); parsed is None when the prediction could not be parsed at all.
"""
import re
from typing import Dict, Optional, Sequence

WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
NOTE_RE = re.compile(r"(?<![A-Za-z#])([A-G]#?)(?![A-Za-z#])")


def norm_text(s: str) -> str:
    s = s.lower().replace("\n", " ")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w#\s.\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_number(text: str) -> Optional[float]:
    m = NUM_RE.search(text)
    if m:
        return float(m.group(0))
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in WORDS:
            return float(WORDS[w])
    return None


def parse_bool(text: str) -> Optional[bool]:
    t = norm_text(text)
    if t.startswith("yes") or t.startswith("true"):
        return True
    if t.startswith("no") or t.startswith("false"):
        return False
    return None


def parse_notes(text: str):
    t = text.strip()
    if re.search(r"\b(none|nothing|no note|no notes|all of (them|these))\b", t, re.I) and not NOTE_RE.search(re.sub(r"\bNone\b", "", t)):
        return []
    return sorted(set(NOTE_RE.findall(t)))


def find_label(text: str, labels: Sequence[str]) -> Optional[str]:
    """the longest label of the family that occurs as a whole phrase in the prediction"""
    t = text
    best = None
    for lab in sorted(labels, key=len, reverse=True):
        pat = r"(?<![A-Za-z#])" + re.escape(lab) + r"(?![A-Za-z#])"
        if re.search(pat, t, re.I):
            best = lab
            break
    return best


def score(pred: str, rec: Dict, vocab: Optional[Dict[str, Sequence[str]]] = None):
    gt, g = rec["gold_answer_type"], rec["gold_answer"]
    if gt in ("integer", "number"):
        p = parse_number(pred)
        if p is None:
            return False, None
        tol = float(rec.get("gold_tolerance", 0) or 0)
        return abs(p - float(g)) <= tol + 1e-9, p
    if gt == "boolean":
        p = parse_bool(pred)
        return (p is not None and p == bool(g)), p
    if gt == "sequence":
        p = parse_notes(pred)
        return sorted(p) == sorted(g), p
    if gt == "class_label":
        labels = (vocab or {}).get(rec["v2"]["family"]) or [g]
        p = find_label(pred, labels)
        return (p is not None and p.lower() == str(g).lower()), p
    return norm_text(pred) == norm_text(str(g)), pred
