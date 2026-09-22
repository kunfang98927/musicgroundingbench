"""Span metrics for v2 grounding (text answers 'From X second(s) to Y second(s)')."""
import re
from typing import List, Sequence, Tuple

SPAN_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*seconds?\s*to\s*([0-9]*\.?[0-9]+)\s*seconds?", re.I)
Span = Tuple[float, float]


def parse_spans(text: str) -> List[Span]:
    return [(float(a), max(float(b), float(a) + 1e-8)) for a, b in SPAN_RE.findall(text or "")]


def iou(a: Span, b: Span) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / u if u > 0 else 0.0


def match(pred: Sequence[Span], gt: Sequence[Span], thr: float) -> int:
    """greedy one-to-one matching by IoU; returns the number of matched pairs"""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); tp += 1
    return tp


class Acc:
    def __init__(self):
        self.n = self.tp = self.fp = self.fn = self.exact = 0
        self.n_empty = self.empty_ok = 0

    def add(self, pred, gt, thr):
        tp = match(pred, gt, thr)
        self.n += 1; self.tp += tp; self.fp += len(pred) - tp; self.fn += len(gt) - tp
        ok = tp == len(pred) == len(gt)
        self.exact += ok
        if not gt:
            self.n_empty += 1; self.empty_ok += (len(pred) == 0)

    def out(self):
        p = self.tp / max(self.tp + self.fp, 1); r = self.tp / max(self.tp + self.fn, 1)
        return {"n": self.n, "P": round(p, 3), "R": round(r, 3), "F1": round(2 * p * r / max(p + r, 1e-9), 3), "exact": round(self.exact / max(self.n, 1), 3),
                "empty_acc": round(self.empty_ok / self.n_empty, 3) if self.n_empty else None}
