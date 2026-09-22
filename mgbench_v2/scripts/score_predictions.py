"""Score inference outputs of inf_v2.py against the v2 QA files.

python -m mgbench_v2.scripts.score_predictions --task understanding --preds P.json --meta test-small.json --train train.json --out report.json
python -m mgbench_v2.scripts.score_predictions --task grounding     --preds P.json --meta test-small.json --out report.json [--thr 0.5]
"""
import argparse, json
from collections import Counter, defaultdict

from mgbench_v2.scoring import grounding as G
from mgbench_v2.scoring import understanding as U


def load_meta(path):
    return json.load(open(path))


def find_meta(meta_by_key, p):
    """join a prediction to its QA record by (sample_id, question); `index` is only the position in the *sampled* inference set"""
    m = meta_by_key.get((p["sample_id"], p["question"]))
    if m is None:
        raise KeyError(f"no QA record for {p['sample_id']!r} / {p['question']!r}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["understanding", "grounding"], required=True)
    ap.add_argument("--preds", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--train", default=None, help="understanding: train json (label vocabulary of class-label families)")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds, meta = json.load(open(a.preds)), load_meta(a.meta)
    meta_by_key = {(m["sample_id"], m["question"]): m for m in meta}
    rep = {"n_pred": len(preds)}
    fam_of = lambda r: r["v2"]["family"] if "v2" in r else r["query_key"]
    if a.task == "understanding":
        vocab = defaultdict(set)
        for r in json.load(open(a.train)) if a.train else meta:
            if r["gold_answer_type"] == "class_label":
                vocab[r["v2"]["family"]].add(str(r["gold_answer"]))
        vocab = {k: sorted(v) for k, v in vocab.items()}
        fam, con, unparsed = defaultdict(list), defaultdict(list), Counter()
        gold_by_fam = defaultdict(list)
        for p in preds:
            m = find_meta(meta_by_key, p)
            ok, parsed = U.score(p["prediction"], m, vocab)
            fam[fam_of(m)].append(ok); con[m["concept"]].append(ok)
            unparsed[fam_of(m)] += parsed is None
            gold_by_fam[fam_of(m)].append(str(m["gold_answer"]))
        rep["overall_acc"] = round(sum(sum(v) for v in fam.values()) / max(len(preds), 1), 4)
        rep["per_family"] = {f: {"n": len(v), "acc": round(sum(v) / len(v), 3), "majority": round(Counter(gold_by_fam[f]).most_common(1)[0][1] / len(v), 3), "unparsed": unparsed[f]}
                             for f, v in sorted(fam.items())}
        rep["per_concept"] = {c: {"n": len(v), "acc": round(sum(v) / len(v), 3)} for c, v in sorted(con.items())}
        rep["concept_majority_weighted"] = {}
        maj = defaultdict(list)
        for f, r in rep["per_family"].items():
            maj[next(m["concept"] for m in meta if fam_of(m) == f)].append((r["n"], r["majority"]))
        rep["concept_majority_weighted"] = {c: round(sum(n * x for n, x in v) / sum(n for n, _ in v), 3) for c, v in maj.items()}
    else:
        fam, con, glob = defaultdict(G.Acc), defaultdict(G.Acc), G.Acc()
        base_all = defaultdict(G.Acc)
        for p in preds:
            m = find_meta(meta_by_key, p)
            gt = [tuple(x) for x in m["answer_spans_sec"]]
            pr = G.parse_spans(p["prediction"])
            for acc in (fam[fam_of(m)], con[m.get("concept", m["query_key"].split("/")[1])], glob):
                acc.add(pr, gt, a.thr)
            base_all[fam_of(m)].add([(n["start"], n["end"]) for n in m["notes"]], gt, a.thr)      # baseline: predict every note
        rep["global"] = glob.out()
        rep["per_family"] = {f: dict(v.out(), baseline_all_notes_F1=base_all[f].out()["F1"]) for f, v in sorted(fam.items())}
        rep["per_concept"] = {c: v.out() for c, v in sorted(con.items())}
        rep["iou_thr"] = a.thr
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k in ("overall_acc", "global", "n_pred")}))
    for f, v in rep["per_family"].items():
        print(f, v)


if __name__ == "__main__":
    main()
