"""Understanding metrics for MusicGroundingBench (Section 4.2 of the paper), from the output of
evaluation.judge:

* IFR - share of questions whose answer the judge marks as following the instruction.
* Accuracy - share of ALL questions answered correctly (an answer that does not follow the
  instruction counts as wrong), overall and per music concept.

Usage:
    python -m evaluation.understanding --qa MGBench-2B/qa/understanding/test.json --judged judged.json
"""
import argparse
import json
from collections import defaultdict
from typing import Dict, List

CONCEPTS = ["rhythm", "tonality", "interval", "bar_comparison", "harmony", "repetition", "harmonic_pattern"]


def followed(v: Dict) -> bool:
    return v.get("instruction_following_score", 0) == 1


def correct(v: Dict) -> bool:
    return followed(v) and v.get("accuracy_score", 0.0) == 1.0


def evaluate(qa: List[Dict], judged: List[Dict]) -> Dict:
    if [(j["sample_id"], j["question"]) for j in judged] != [(q["sample_id"], q["question"]) for q in qa]:
        raise SystemExit("the judged file does not cover the QA file question by question (run evaluation.judge to completion)")
    groups = defaultdict(list)
    for q, v in zip(qa, judged):
        groups[q["concept"]].append(v)
    score = lambda vs: {"n": len(vs), "IFR": sum(map(followed, vs)) / len(vs), "Accuracy": sum(map(correct, vs)) / len(vs)}
    order = [c for c in CONCEPTS if c in groups] + sorted(set(groups) - set(CONCEPTS))
    return {"overall": score(judged), "per_concept": {c: score(groups[c]) for c in order}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", required=True, help="reference QA file, e.g. MGBench-2B/qa/understanding/test.json")
    ap.add_argument("--judged", required=True, help="output of evaluation.judge")
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args()
    res = evaluate(json.load(open(a.qa, encoding="utf-8")), json.load(open(a.judged, encoding="utf-8")))
    print(f"{'group':18s}      n  IFR     Accuracy")
    for name, r in [("overall", res["overall"])] + list(res["per_concept"].items()):
        print(f"{name:18s}  {r['n']:5d}  {r['IFR']:.4f}  {r['Accuracy']:.4f}")
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
