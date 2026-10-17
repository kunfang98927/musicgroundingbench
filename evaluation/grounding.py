"""Grounding metrics for MusicGroundingBench (Section 4.1 of the paper).

Predictions and references are sets of time intervals. Reported:

* IFR - instruction-following rate: the answer contains at least one parseable time span
  ("X second(s) to Y second(s)"), or it is an explicit "no matching note" answer (the
  dataset's own phrasings, or "None").
* MergedIoU - IoU of the time regions covered by the predicted and the reference spans after
  merging overlapping spans within each set (averaged over questions; 0 when both are empty).
* Precision / recall / F1 of spans under greedy one-to-one matching at an IoU threshold
  (pooled over all questions), and the onset / offset mean absolute error (ms) of the matched
  spans.

Usage:
    python -m evaluation.grounding --qa MGBench-2B/qa/grounding/test.json --pred preds.json
"""
import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from .io import load_predictions, align

Span = Tuple[float, float]

_TEXT_TIME_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*second[s]?\s*to\s*([0-9]*\.?[0-9]+)\s*second[s]?", re.IGNORECASE)

# Answers that state that no note satisfies the query: the reference phrasings used in the
# dataset (MusicGroundingBench-2B and -3N), plus "None".
ABSTENTIONS = {"none", "there is no such note in this clip", "no matching note was found in this clip", "no time span"}


def parse_spans(text: str) -> List[Span]:
    """Every "X second(s) to Y second(s)" in the answer, as (start, end) with end >= start."""
    spans = []
    for m in _TEXT_TIME_RE.finditer(text or ""):
        t0, t1 = float(m.group(1)), float(m.group(2))
        spans.append((t0, max(t0, t1)))
    return spans


def is_abstention(text: str) -> bool:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".").strip() in ABSTENTIONS


def iou_1d(a: Span, b: Span) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(0.0, (a[1] - a[0]) + (b[1] - b[0]) - inter)
    return inter / union if union > 0 else 0.0


def greedy_match(preds: List[Span], gts: List[Span], iou_thr: float) -> List[Tuple[Span, Span, float]]:
    """Repeatedly pair the remaining prediction and reference with the highest IoU, while it is >= iou_thr."""
    matches = []
    rest_p, rest_g = set(range(len(preds))), set(range(len(gts)))
    while rest_p and rest_g:
        best, best_iou = None, -1.0
        for pi in rest_p:
            for gi in rest_g:
                v = iou_1d(preds[pi], gts[gi])
                if v > best_iou:
                    best, best_iou = (pi, gi), v
        if best is None or best_iou < iou_thr:
            break
        pi, gi = best
        matches.append((preds[pi], gts[gi], best_iou))
        rest_p.remove(pi)
        rest_g.remove(gi)
    return matches


def merge_intervals(spans: List[Span]) -> List[Span]:
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def merged_iou(preds: List[Span], gts: List[Span]) -> float:
    p, g = merge_intervals(preds), merge_intervals(gts)
    inter, i, j = 0.0, 0, 0
    while i < len(p) and j < len(g):
        inter += max(0.0, min(p[i][1], g[j][1]) - max(p[i][0], g[j][0]))
        if p[i][1] <= g[j][1]:
            i += 1
        else:
            j += 1
    union = sum(e - s for s, e in p) + sum(e - s for s, e in g) - inter
    return inter / union if union > 0 else 0.0


class GroundingScore:
    """Accumulates one group of questions (all of them, or one concept / category)."""

    def __init__(self, thresholds=(0.5, 0.7)):
        self.thresholds = thresholds
        self.n = self.followed = 0
        self.merged_iou_sum = 0.0
        self.pred_spans = self.gt_spans = 0
        self.matched = {t: 0 for t in thresholds}
        self.onset_err = {t: 0.0 for t in thresholds}
        self.offset_err = {t: 0.0 for t in thresholds}

    def update(self, prediction: str, gts: List[Span]):
        preds = parse_spans(prediction)
        self.n += 1
        self.followed += bool(preds) or is_abstention(prediction)
        self.merged_iou_sum += merged_iou(preds, gts)
        self.pred_spans += len(preds)
        self.gt_spans += len(gts)
        for t in self.thresholds:
            for (ps, pe), (gs, ge), _ in greedy_match(preds, gts, t):
                self.matched[t] += 1
                self.onset_err[t] += abs(ps - gs)
                self.offset_err[t] += abs(pe - ge)

    def result(self) -> Dict:
        div = lambda a, b: a / b if b else 0.0
        out = {"n": self.n, "IFR": div(self.followed, self.n), "MergedIoU": div(self.merged_iou_sum, self.n)}
        for t in self.thresholds:
            p, r = div(self.matched[t], self.pred_spans), div(self.matched[t], self.gt_spans)
            out[f"IoU>={t}"] = {
                "P": p, "R": r, "F1": div(2 * p * r, p + r),
                "onset_MAE_ms": div(self.onset_err[t], self.matched[t]) * 1000.0,
                "offset_MAE_ms": div(self.offset_err[t], self.matched[t]) * 1000.0,
                "matched_spans": self.matched[t],
            }
        return out


def group_of(item: Dict) -> str:
    """MusicGroundingBench-2B: the music concept; MusicGroundingBench-3N: the query category (ABS / REL / ORD / PAT)."""
    if item.get("concept"):
        return item["concept"]
    parts = (item.get("query_key") or "").split("/")
    return parts[1] if len(parts) > 1 else "unknown"


def evaluate(qa: List[Dict], predictions: List[str], thresholds=(0.5, 0.7)) -> Dict:
    overall, groups = GroundingScore(thresholds), defaultdict(lambda: GroundingScore(thresholds))
    for item, pred in zip(qa, predictions):
        gts = [tuple(s) for s in item["answer_spans_sec"]]
        overall.update(pred, gts)
        groups[group_of(item)].update(pred, gts)
    return {"overall": overall.result(), "per_group": {g: s.result() for g, s in sorted(groups.items())}}


def _fmt(res: Dict, thresholds) -> str:
    cells = [f"{res['n']:>5d}", f"{res['IFR']:>6.4f}", f"{res['MergedIoU']:>9.4f}"]
    for t in thresholds:
        m = res[f"IoU>={t}"]
        mae = f"{m['onset_MAE_ms']:.2f} / {m['offset_MAE_ms']:.2f}" if m["matched_spans"] else "- / -"
        cells += [f"{m['P']:>7.4f}", f"{m['R']:>7.4f}", f"{m['F1']:>7.4f}", f"{mae:>17s}"]
    return "  ".join(cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", required=True, help="reference QA file, e.g. MGBench-2B/qa/grounding/test.json")
    ap.add_argument("--pred", required=True, help="predictions: JSON list or JSON lines with sample_id, question, prediction")
    ap.add_argument("--iou", type=float, nargs="+", default=[0.5, 0.7], help="IoU thresholds (default: 0.5 0.7)")
    ap.add_argument("--allow-missing", action="store_true", help="score questions without a prediction as empty answers")
    ap.add_argument("--out", help="write the full results as JSON")
    a = ap.parse_args()
    qa = json.load(open(a.qa, encoding="utf-8"))
    preds = align(qa, load_predictions(a.pred), a.allow_missing)
    res = evaluate(qa, preds, tuple(a.iou))
    head = [f"{'n':>5s}", f"{'IFR':>6s}", f"{'MergedIoU':>9s}"] + [x for t in a.iou for x in (f"{'P@' + str(t):>7s}", f"{'R@' + str(t):>7s}", f"{'F1@' + str(t):>7s}", f"{'MAE@' + str(t) + ' (ms)':>17s}")]
    print(f"{'group':18s}  " + "  ".join(head))
    print(f"{'overall':18s}  " + _fmt(res["overall"], a.iou))
    for g, r in res["per_group"].items():
        print(f"{g:18s}  " + _fmt(r, a.iou))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
