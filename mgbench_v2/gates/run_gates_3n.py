"""Release gates for the 3N QA layer: structure, independent re-derivation from the stored notes, text contract, empty-share, shortcut baselines."""
import argparse, json, os, re
from collections import Counter, defaultdict

import numpy as np

from mgbench_v2.gates.run_gates import _iou

SPLITS = ("train", "val", "test", "test_small")
VEL_MARGIN, INT_MARGIN = 24, 2


def expected_idx(r):
    n = sorted(r["notes"], key=lambda x: (x["start"], x["end"]))
    p = [x["pitch"] for x in n]; v = [x["velocity"] for x in n]; N = len(n)
    fam, sub, prm = r["v2"]["family"], r["v2"]["sub"], r["v2"]["params"]
    if fam == "A-PITCH":
        return [i for i in range(N) if p[i] in prm["vals"]]
    if fam == "O-NOTE":
        k = sub.split("=")[1]
        return {"first": [0], "second": [1] if N >= 2 else [], "third": [2] if N >= 3 else [], "last": [N - 1]}[k]
    if fam == "O-RANGE":
        return (list(range(2)) if sub == "prefix" else list(range(N - 2, N))) if N >= 2 else []
    if fam == "R-PITCH":
        name = sub.split("=")[1]
        dist = sorted(set(p), reverse=name.endswith("highest"))
        rank = 2 if name.startswith("second") else 1
        return [i for i in range(N) if p[i] == dist[rank - 1]] if rank <= len(dist) else []
    if fam == "R-VEL":
        name = sub.split("=")[1]
        order = sorted(range(N), key=lambda i: -v[i])
        vs = [v[i] for i in order]
        if name == "loudest":
            return [order[0]] if N >= 2 and vs[0] - vs[1] >= VEL_MARGIN else None
        if name == "softest":
            return [order[-1]] if N >= 2 and vs[-2] - vs[-1] >= VEL_MARGIN else None
        return [order[1]] if N == 3 and vs[0] - vs[1] >= VEL_MARGIN and vs[1] - vs[2] >= VEL_MARGIN else None
    if fam == "P-PAIR":
        if N != 3:
            return None
        d = sorted((abs(p[a] - p[b]), (a, b)) for a in range(3) for b in range(a + 1, 3))
        return list(d[0][1]) if d[0][0] < d[1][0] else None
    if fam == "P-PDIFF":
        D = prm["delta"]
        pairs = [(a, b) for a in range(N) for b in range(a + 1, N) if abs(p[a] - p[b]) == D]
        return sorted({x for pr in pairs for x in pr}) if len(pairs) <= 1 else None
    if fam == "P-CONTOUR":
        if N != 3 or 0 in (p[1] - p[0], p[2] - p[1]):
            return None
        t = "-".join("up" if x > 0 else "down" for x in (p[1] - p[0], p[2] - p[1]))
        return [0, 1, 2] if t == sub.split("=")[1] else []
    if fam == "P-INT":
        if N != 3:
            return None
        a, b = abs(p[1] - p[0]), abs(p[2] - p[1])
        t = "widen" if b - a >= INT_MARGIN else "narrow" if a - b >= INT_MARGIN else "same" if a == b else None
        return None if t is None else ([0, 1, 2] if t == sub.split("=")[1] else [])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {s: json.load(open(f"{a.qa}/llm_{s}.json")) for s in SPLITS if os.path.exists(f"{a.qa}/llm_{s}.json")}
    rep = {"oracle": {}, "contract": {}, "empty": {}, "median_span": {}, "text_only_empty": {}}
    res = defaultdict(Counter)
    bad = defaultdict(list)
    for sp, recs in data.items():
        for r in recs:
            fam = r["v2"]["family"]
            res[fam]["n"] += 1
            ok = all(k in r for k in ("question", "answer", "answer_token", "answer_spans", "query_key", "audio_path", "notes"))
            ok &= all(int(round(s0 * 75)) == f0 for (s0, _), (f0, _) in zip(r["answer_spans_sec"], r["answer_spans"]))
            n = sorted(r["notes"], key=lambda x: (x["start"], x["end"]))
            exp = expected_idx(r)
            if not ok:
                res[fam]["structure_fail"] += 1
                continue
            if exp is None:
                res[fam]["indep_skip"] += 1; bad[fam].append((sp, r["sample_id"], "skip")); continue
            got = sorted(i for i, x in enumerate(n) if any(abs(x["start"] - s) < 2e-5 and abs(x["end"] - e) < 2e-5 for s, e in r["answer_spans_sec"]))
            if got == sorted(exp):
                res[fam]["ok"] += 1
            else:
                res[fam]["fail"] += 1; bad[fam].append((sp, r["sample_id"], r["v2"]["sub"]))
            # text contract
            q = r["question"]; prm = r["v2"]["params"]; errs = []
            if fam == "A-PITCH" and not all(str(x) in q for x in prm["vals"]):
                errs.append("pitch value missing")
            if fam == "P-PDIFF" and not re.search(rf"\b{prm['delta']} semitone", q):
                errs.append("delta missing")
            if fam == "R-VEL" and not any(w in q for w in ("loud", "soft", "quiet")):
                errs.append("dynamics wording")
            if errs:
                rep["contract"].setdefault(fam, []).append((r["sample_id"], errs))
    rep["oracle"] = {f: dict(c) for f, c in sorted(res.items())}
    rep["oracle_examples"] = {f: v[:5] for f, v in bad.items()}
    for sp, recs in data.items():
        by = defaultdict(lambda: [0, 0])
        for r in recs:
            by[r["v2"]["family"]][0] += 1; by[r["v2"]["family"]][1] += not r["answer_spans"]
        rep["empty"][sp] = {f: round(e / n, 3) for f, (n, e) in by.items()}
    tr, te = data["train"], data["test"]
    for fam in sorted({r["v2"]["family"] for r in tr}):
        st = [s for r in tr if r["v2"]["family"] == fam for s, _ in r["answer_spans_sec"]]; en = [e for r in tr if r["v2"]["family"] == fam for _, e in r["answer_spans_sec"]]
        items = [r for r in te if r["v2"]["family"] == fam]
        pred = (float(np.median(st)), float(np.median(en)))
        hit = float(np.mean([any(_iou(pred, tuple(g)) >= 0.5 for g in r["answer_spans_sec"]) for r in items]))
        rep["median_span"][fam] = {"n_test": len(items), "hit@0.5": round(hit, 3), "ok": hit <= 0.40}
        # can the question text alone tell whether the answer is empty?
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        ya = [int(not r["answer_spans"]) for r in tr if r["v2"]["family"] == fam]; yb = [int(not r["answer_spans"]) for r in items]
        if len(set(ya)) > 1 and len(yb) > 30:
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3)
            Xa = v.fit_transform([r["question"] for r in tr if r["v2"]["family"] == fam]); Xb = v.transform([r["question"] for r in items])
            acc = float(np.mean(LogisticRegression(max_iter=300).fit(Xa, ya).predict(Xb) == np.array(yb)))
            maj = max(np.mean(yb), 1 - np.mean(yb)); z = (acc - maj) / max(np.sqrt(maj * (1 - maj) / len(yb)), 1e-9)
            rep["text_only_empty"][fam] = {"acc": round(acc, 3), "majority": round(float(maj), 3), "z": round(float(z), 2), "ok": bool(acc <= maj + 0.03 or z < 1.65)}
    json.dump(rep, open(f"{a.out}/gates_3n.json", "w"), indent=1)
    L = ["# v2 3N gates (auto-generated)", ""]
    tot = Counter()
    for c in res.values():
        tot.update(c)
    L.append(f"items={tot['n']} structure_fail={tot['structure_fail']} independent ok={tot['ok']} fail={tot['fail']} skip={tot['indep_skip']}")
    L += [f"- {f}: {dict(c)}" for f, c in sorted(res.items()) if c["fail"] or c["indep_skip"] or c["structure_fail"]]
    L += ["", "contract failures: " + json.dumps({f: len(v) for f, v in rep["contract"].items()}), "", "empty share (test): " + json.dumps(rep["empty"].get("test", {})), "",
          "median-span baseline: " + json.dumps(rep["median_span"]), "", "text-only(empty vs non-empty): " + json.dumps(rep["text_only_empty"])]
    open(f"{a.out}/GATES_REPORT_3n.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
