import json, sys, random
from collections import Counter, defaultdict
root, split = sys.argv[1], sys.argv[2]
nshow = int(sys.argv[3]) if len(sys.argv) > 3 else 1
random.seed(1)
for kind in ("grounding", "understanding"):
    d = json.load(open(f"{root}/{kind}_qa/{split}.json"))
    by = defaultdict(list)
    for r in d: by[r["v2"]["family"]].append(r)
    print(f"===== {kind} {split}: {len(d)} items, {len(set(r['sample_id'] for r in d))} clips")
    for fam in sorted(by):
        rs = by[fam]; c = Counter(r["v2"]["answer_class"] for r in rs); n = len(rs)
        emp = sum(1 for r in rs if not r["answer_spans"]) / n
        top = c.most_common(1)[0]
        print(f"{fam:6s} n={n:5d} classes={len(c):3d} majority={top[1]/n:.2f} ({top[0]}) empty={emp:.2f} distinct_q={len(set(r['question'] for r in rs))}")
        for r in random.sample(rs, min(nshow, n)):
            print("    Q:", r["question"]); print("    A:", r["answer_text"][:140], "| spans", len(r["answer_spans"]))
