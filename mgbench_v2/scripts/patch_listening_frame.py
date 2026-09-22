"""rc4: add the bar-frame sentence to the bar-referring items of the listening page without losing the ratings.

* item on a frame_full clip, not flagged 'unclear' by the rater -> same id, question gets the frame sentence (the verdict is about the audio / answer, unchanged)
* item flagged 'unclear' (the flag was about the missing frame) -> new id `<set>-<fam>-c<NN>`, question patched, so it is rated afresh
* item on a clip where the sentence would be false -> dropped (a replacement is picked from the rc4 QA by build_listening_check --quota-json)
usage: patch_listening_frame.py <listening_dir> <ratings_dir> ; writes <listening_dir>/items.json and <listening_dir>/replace_quota.json
"""
import glob, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mgbench_v2 import phrasing as P
from mgbench_v2.oracle2b import Excerpt, Frame, Note
from mgbench_v2.qa2b import BAR_FAMS

d, rdir = sys.argv[1], sys.argv[2]
items = json.load(open(f"{d}/items.json"))
rat = {}
for f in glob.glob(f"{rdir}/*.json"):
    r = json.load(open(f)); rat[r["item"]] = r
out, dropped, renamed, patched = [], {}, 0, 0
nxt = {}
for it in items:
    if it["set"] != "MGBench-2B" or it["family"] not in BAR_FAMS or it["id"].split("-")[-1].startswith("c"):
        out.append(it); continue
    fr = it["frame"]
    e = Excerpt([Note(a, b, p, v) for a, b, p, v in it["notes"]], Frame(fr["meter"], fr["tempo_bpm"], fr["music_start"], fr["bar_dur"]))
    if not e.frame_full:
        dropped[it["family"]] = dropped.get(it["family"], 0) + 1; continue
    rng = random.Random(it["id"])
    s = P.frame_sentence(rng)
    it = dict(it); it["question"] = f"{s} {it['question']}" if rng.random() < 0.6 else f"{it['question']} {s}"
    if (rat.get(it["id"]) or {}).get("unclear"):
        fam = it["family"]; nxt[fam] = nxt.get(fam, 0) + 1
        it["id"] = f"2B-{fam}-c{nxt[fam]:02d}"; renamed += 1
    else:
        patched += 1
    out.append(it)
json.dump(out, open(f"{d}/items.json", "w"))
json.dump(dropped, open(f"{d}/replace_quota.json", "w"))
print(f"kept+patched {patched}, re-identified (unclear flag) {renamed}, dropped (frame not true) {sum(dropped.values())}: {dropped}; total {len(out)}")
