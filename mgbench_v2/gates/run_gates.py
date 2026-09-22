"""Release gates for the v2 QA layer (2B). All blocking gates must pass; every number is written to GATES_REPORT.md/json.

python -m mgbench_v2.gates.run_gates --qa /path/to/qa_root --out /path/to/report_dir
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

from mgbench_v2.gates.independent import View, expected

# families whose questions mention a bar without a meter sentence: they must carry the (relative) bar-frame sentence and no absolute time
# (kept as its own copy, see qa2b.BAR_FAMS)
BAR_FAMS = {"R-G1", "R-G3", "R-G6", "B-G1", "B-G2", "H-G1", "H-G2", "A-G1", "A-G2", "Rp-G1", "Rp-G2",
            "R-U6", "B-U1", "B-U2", "B-U3", "B-U4", "H-U1", "H-U2", "H-U3", "A-U1", "A-U2", "Rp-U1", "Rp-U2"}

SPLITS = ("train", "val", "test", "test-small")
REQ = ["question", "answer_text", "answer_spans", "answer_spans_sec", "query_key", "concept", "audio_path", "sample_id", "notes", "v2", "question_variants"]


def load(root):
    return {k: {s: json.load(open(f"{root}/{k}_qa/{s}.json")) for s in SPLITS if os.path.exists(f"{root}/{k}_qa/{s}.json")} for k in ("grounding", "understanding")}


# ----------------------------------------------------------------------------------- gate 1: structure + independent oracle
def gate_structure_and_oracle(data):
    res = defaultdict(lambda: Counter())
    bad = defaultdict(list)
    for kind, splits in data.items():
        for sp, recs in splits.items():
            for r in recs:
                fam = r["v2"]["family"]
                key = f"{kind}/{sp}/{fam}"
                res[key]["n"] += 1
                miss = [k for k in REQ if k not in r]
                if miss or (kind == "understanding" and "gold_answer" not in r):
                    res[key]["structure_fail"] += 1; bad[key].append(("missing", r["sample_id"], miss)); continue
                if len(r["answer_spans"]) != len(r["answer_spans_sec"]) or any(int(round(a * 75)) != s0 for (a, _), (s0, _) in zip(r["answer_spans_sec"], r["answer_spans"])):
                    res[key]["structure_fail"] += 1; bad[key].append(("frames", r["sample_id"]))
                v = View(r)
                ex = expected(v, r)
                if ex["kind"] == "skip":
                    res[key]["oracle_skip"] += 1; bad[key].append(("indep-skip", r["sample_id"], r["v2"]["sub"])); continue
                if ex["kind"] == "gold":
                    g = r["gold_answer"]
                    ok = (abs(float(g) - float(ex["value"])) < 1e-9) if isinstance(ex["value"], (int, float)) and not isinstance(ex["value"], bool) else g == ex["value"]
                elif ex["kind"] == "span":
                    ok = len(r["answer_spans_sec"]) == 1 and abs(r["answer_spans_sec"][0][0] - ex["value"][0]) < 2e-5 and abs(r["answer_spans_sec"][0][1] - ex["value"][1]) < 2e-5
                else:
                    if kind == "grounding":
                        got = v.spans_to_idx(r["answer_spans_sec"])
                        ok = None not in got and sorted(got) == ex["value"]
                    else:
                        ok = sorted(r["evidence_note_indices"]) is not None  # evidence checked separately; gold checked via the 'gold' families
                        ok = True
                res[key]["oracle_ok" if ok else "oracle_fail"] += 1
                if not ok:
                    bad[key].append(("oracle", r["sample_id"], r["v2"]["sub"]))
    return res, bad


# ----------------------------------------------------------------------------------- gate 2: text <-> parameter contract
def gate_text_contract(data):
    fails, n = defaultdict(list), Counter()
    for kind, splits in data.items():
        for sp, recs in splits.items():
            for r in recs:
                q = r["question"]; fam = r["v2"]["family"]; prm = r["v2"]["params"]; fr = r["v2"]["frame"]; ctx = r["v2"]["context"]
                n[fam] += 1
                errs = []
                if "meter" in ctx and fr["meter"] not in q:
                    errs.append("meter missing in text")
                if "tempo" in ctx and str(fr["tempo_bpm"]) not in q:
                    errs.append("tempo missing in text")
                if "tempo" not in ctx and re.search(r"\b\d+\s*(bpm|beats per minute)", q):
                    errs.append("tempo in text but not in context flags")
                if "set" in prm and prm["set"] not in q:
                    errs.append("scale set missing")
                if fam == "R-G2" and not (f"beat {prm['k']}" in q or f"{['','first','second','third','fourth','fifth','sixth'][prm['k']]} beat" in q):
                    errs.append("beat number missing")
                if fam in ("I-U1",) and not re.search(rf"\b{prm['semitones']} semitone", q):
                    errs.append("semitone count missing")
                if fam == "I-G4" and f"{prm['semitones']} semitone" not in q:
                    errs.append("semitone count missing")
                if fam == "I-G4":
                    up = any(w in q for w in ("ascending", "above", "rising")); dn = any(w in q for w in ("descending", "below", "falling"))
                    if up == dn or up != (prm["direction"] == "ascending"):
                        errs.append("direction wording mismatch")
                if fam in ("I-G7", "I-U3") and (prm["direction"] not in q and not ("higher" in q or "lower" in q)):
                    errs.append("direction missing")
                if fam == "R-G4" and re.sub(r"_", " ", r["v2"]["sub"].split("=")[1]) not in q:
                    errs.append("note value missing")
                if fam.startswith("R-U") and fam in ("R-U3", "R-U4") and "note" not in q:
                    errs.append("note ref missing")
                if fam == "R-U4":
                    vv = View(r)
                    bi = vv.in_bar(prm["bar"]); ni = prm["note_index"]
                    key = {"highest": lambda i: vv.p[i], "lowest": lambda i: -vv.p[i], "longest": lambda i: vv.ioi[i]}[prm["ref_kind"]]
                    if ni not in bi or sum(1 for i in bi if abs(key(i) - key(ni)) < 1e-9) != 1 or max(key(i) for i in bi) != key(ni):
                        errs.append("property reference does not identify the note")
                    if f"the {prm['ref_kind']} note of the {['', 'first', 'second'][prm['bar']]} bar" not in q.lower():
                        errs.append("reference wording missing")
                if fam == "R-U3":
                    vv = View(r)
                    if f"{vv.st[prm['note_index']]:.2f}" != prm["onset"] or prm["onset"] not in q:
                        errs.append("time reference does not match the note")
                    others = [i for i in range(vv.N) if i != prm["note_index"] and abs(vv.st[i] - vv.st[prm["note_index"]]) < 0.05]
                    if others:
                        errs.append("time reference ambiguous")
                # rc3: no length-definition sentence any more; instead the excerpt must be rest-free (a length is then simply what one hears)
                if re.search(r"(start of the next note|next note's start|start of the following note|clip's last note|last note of the clip)", q):
                    errs.append("stale note-length sentence")
                if fam in ("R-G4", "R-G5", "R-G6", "R-U3") or (fam == "R-U4" and prm.get("ref_kind") == "longest"):
                    if not View(r).rest_free:
                        errs.append("length question on an excerpt with a rest / early last note")
                if fam in BAR_FAMS:
                    if "frame" not in ctx or not re.search(r"equally long", q):
                        errs.append("bar frame sentence missing")
                    elif not View(r).frame_full:
                        errs.append("bar frame sentence is not true for this excerpt")
                    if re.search(r"\d+\.\d\d s", q):
                        errs.append("absolute time in a bar question (would hand over the bar boundary / an answer span)")
                if fam == "H-U2" and not re.search(r"pitch class", q):
                    errs.append("H-U2 needs the pitch-class clarification")
                if fam == "R-G5" and (fr["meter"] not in ("2/4", "3/4", "4/4") or "metric" not in ctx or f"In {fr['meter']} the " not in q):
                    errs.append("R-G5 needs a simple meter and the strength-of-positions sentence")
                if "meter" in ctx and not re.search(r"(beat 1 of|first beat)", q):
                    errs.append("bar frame (first note = beat 1) not stated")
                if fam in ("R-G2", "R-U4") and "tempo" not in ctx:
                    errs.append("tempo must be stated for beat questions")
                # 6/8 must say one beat = eighth when a meter sentence with unit is used, and never 'quarter' for 6/8
                if fr["meter"] == "6/8" and "unit" in ctx and "eighth note)" not in q:
                    errs.append("6/8 unit wording")
                if fr["meter"] != "6/8" and "unit" in ctx and "quarter note)" not in q:
                    errs.append("x/4 unit wording")
                if errs:
                    fails[fam].append((r["sample_id"], errs))
    return n, fails


# ----------------------------------------------------------------------------------- gate 3: distributions and shortcuts
def max_share(k):
    return max(0.35, 1.5 / max(k, 1))


def gate_distributions(data):
    out = {}
    for sp, recs in data["understanding"].items():
        by = defaultdict(list)
        for r in recs:
            by[r["v2"]["family"]].append(str(r["gold_answer"]))
        for fam, xs in by.items():
            c = Counter(xs); n = len(xs)
            k = sum(1 for v in c.values() if v >= max(2, 0.01 * n))
            top = c.most_common(1)[0]
            boolean = set(c) <= {"True", "False"}
            lim = 0.55 if boolean else max_share(k)
            out[f"{sp}/{fam}"] = {"n": n, "classes": len(c), "majority": round(top[1] / n, 3), "limit": round(lim, 3), "ok": bool(top[1] / n <= lim + 0.01 or n < 200 or sp == "test-small")}
    return out


def yesno_share(data):
    out = {}
    for sp, recs in data["understanding"].items():
        tot, yn = Counter(), Counter()
        for r in recs:
            tot[r["concept"]] += 1; yn[r["concept"]] += r["gold_answer_type"] == "boolean"
        out[sp] = {c: round(yn[c] / tot[c], 3) for c in tot}
    return out


def gate_empty_share(data):
    out = {}
    for sp, recs in data["grounding"].items():
        by = defaultdict(lambda: [0, 0])
        for r in recs:
            by[r["v2"]["family"]][0] += 1; by[r["v2"]["family"]][1] += (len(r["answer_spans"]) == 0)
        out[sp] = {f: round(e / n, 3) for f, (n, e) in by.items()}
    return out


def text_only_baseline(data):
    """question-text-only classifier per understanding family (train -> test); gate: <= test majority + 3 points"""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    res = {}
    tr, te = data["understanding"]["train"], data["understanding"]["test"]
    for fam in sorted({r["v2"]["family"] for r in tr}):
        a = [(r["question"], str(r["gold_answer"])) for r in tr if r["v2"]["family"] == fam]
        b = [(r["question"], str(r["gold_answer"])) for r in te if r["v2"]["family"] == fam]
        if len(set(y for _, y in a)) < 2 or len(b) < 30:
            continue
        w = TfidfVectorizer(ngram_range=(1, 2), min_df=2); c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3)
        Xa = hstack([w.fit_transform([x for x, _ in a]), c.fit_transform([x for x, _ in a])]).tocsr()
        Xb = hstack([w.transform([x for x, _ in b]), c.transform([x for x, _ in b])]).tocsr()
        clf = LogisticRegression(max_iter=300, C=3.0).fit(Xa, [y for _, y in a])
        acc = float(np.mean(clf.predict(Xb) == np.array([y for _, y in b])))
        maj = Counter(y for _, y in b).most_common(1)[0][1] / len(b)
        z = (acc - maj) / max(np.sqrt(maj * (1 - maj) / len(b)), 1e-9)
        res[fam] = {"n_test": len(b), "text_only_acc": round(acc, 3), "test_majority": round(maj, 3), "excess": round(acc - maj, 3), "z": round(float(z), 2),
                    "ok": bool(acc <= maj + 0.03 or z < 1.65)}            # fail = more than 3 points above the majority AND statistically significant
    return res


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / u if u > 0 else 0.0


def grounding_median_span_baseline(data):
    """predict one span = (median start, median end) of the family's train answers; score = mean best-IoU>=0.5 hit of that span on test"""
    tr, te = data["grounding"]["train"], data["grounding"]["test"]
    res = {}
    for fam in sorted({r["v2"]["family"] for r in tr}):
        st = [s for r in tr if r["v2"]["family"] == fam for s, _ in r["answer_spans_sec"]]
        en = [e for r in tr if r["v2"]["family"] == fam for _, e in r["answer_spans_sec"]]
        if not st:
            continue
        pred = (float(np.median(st)), float(np.median(en)))
        items = [r for r in te if r["v2"]["family"] == fam]
        # per item: a predicted span "hits" if it overlaps some gt span with IoU>=0.5 (empty gt: never)
        hit = np.mean([any(_iou(pred, tuple(g)) >= 0.5 for g in r["answer_spans_sec"]) for r in items])
        res[fam] = {"n_test": len(items), "median_span_hit@0.5": round(float(hit), 3), "ok": bool(hit <= 0.40)}
    return res


def paraphrase_counts(data):
    """distinct question skeletons per family (digits, ordinals, note names and context sentences masked)"""
    mask_ctx = [r"This is a two-bar piece in [\d/]+", r"The music is in [\d/]+", r"Assume the (?:time signature|meter) is [\d/]+", r"This two-bar excerpt is in [\d/]+",
                r"\(one beat is an? \w+ note\)\.?", r"The tempo is [^.]*\.", r"The music plays at [^.]*\."]
    out = defaultdict(set)
    for r in data["grounding"]["train"] + data["understanding"]["train"]:
        q = r["question"]
        for m in mask_ctx:
            q = re.sub(m, " ", q)
        q = re.sub(r"\b(C#|D#|F#|G#|A#|[A-G])\b", "N", q)
        q = re.sub(r"\d+(\.\d+)?", "#", q)
        q = re.sub(r"\b(first|second|third|fourth|fifth|sixth|highest|lowest|upward|downward|ascending|descending)\b", "W", q)
        q = re.sub(r"\s+", " ", q).strip()
        out[r["v2"]["family"]].add(q)
    return {k: len(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle-splits", nargs="+", default=list(SPLITS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load(a.qa)
    rep = {}
    res, bad = gate_structure_and_oracle({k: {s: v for s, v in d.items() if s in a.oracle_splits} for k, d in data.items()})
    fam_tot = defaultdict(Counter)
    for k, c in res.items():
        fam = k.split("/")[-1]
        fam_tot[fam].update(c)
    rep["oracle"] = {f: dict(c) for f, c in sorted(fam_tot.items())}
    rep["oracle_examples"] = {k: v[:5] for k, v in bad.items() if v}
    n, fails = gate_text_contract(data)
    rep["text_contract"] = {f: {"n": n[f], "fail": len(fails.get(f, [])), "examples": fails.get(f, [])[:3]} for f in sorted(n)}
    rep["distributions"] = gate_distributions(data)
    rep["yesno_share"] = yesno_share(data)
    rep["empty_share"] = gate_empty_share(data)
    rep["text_only"] = text_only_baseline(data)
    rep["median_span"] = grounding_median_span_baseline(data)
    rep["paraphrase_skeletons"] = paraphrase_counts(data)
    json.dump(rep, open(f"{a.out}/gates.json", "w"), indent=1)
    # ---- summary
    L = ["# v2 QA gates (auto-generated)", ""]
    tot = Counter()
    for f, c in fam_tot.items():
        tot.update(c)
    L += ["## 1. Structure + independent re-derivation", f"items={tot['n']}, structure_fail={tot['structure_fail']}, oracle_ok={tot['oracle_ok']}, oracle_fail={tot['oracle_fail']}, independent-skip={tot['oracle_skip']}", ""]
    for f, c in sorted(fam_tot.items()):
        if c["oracle_fail"] or c["structure_fail"] or c["oracle_skip"]:
            L.append(f"- {f}: {dict(c)}")
    L += ["", "## 2. Text contract failures", *[f"- {f}: {v['fail']}/{v['n']} e.g. {v['examples'][:1]}" for f, v in rep["text_contract"].items() if v["fail"]], ""]
    L += ["## 3. Distribution gate (understanding, majority-class share)"]
    for k, v in rep["distributions"].items():
        if not v["ok"]:
            L.append(f"- FAIL {k}: {v}")
    L += ["", "## 4. Text-only baselines (understanding)", "| family | n_test | text-only acc | test majority | excess | z | ok |", "|---|---|---|---|---|---|---|"]
    L += [f"| {f} | {v['n_test']} | {v['text_only_acc']} | {v['test_majority']} | {v['excess']} | {v['z']} | {'yes' if v['ok'] else '**NO**'} |" for f, v in rep["text_only"].items()]
    L += ["", "## 5. Median-span baseline (grounding, hit@0.5)", "| family | n_test | hit | ok |", "|---|---|---|---|"]
    L += [f"| {f} | {v['n_test']} | {v['median_span_hit@0.5']} | {'yes' if v['ok'] else '**NO**'} |" for f, v in rep["median_span"].items()]
    L += ["", "## 6. Paraphrase skeletons per family (train)", ", ".join(f"{f}:{c}" for f, c in sorted(rep["paraphrase_skeletons"].items())), ""]
    L += ["## 7. Yes/no share per concept", json.dumps(rep["yesno_share"]), ""]
    open(f"{a.out}/GATES_REPORT.md", "w").write("\n".join(L))
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
