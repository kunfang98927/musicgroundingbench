"""WP4 (3N): rebuild the 3-note QA layer (llm_{train,val,test,test_small}.json) from the released MIDI files.

python -m mgbench_v2.scripts.build_qa_3n --out DIR [--seed 20260919]
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

import pretty_midi

from mgbench_v2 import phrasing as P
from mgbench_v2.oracle3n import Clip, N
from mgbench_v2.qa2b import ORACLE_VERSION
from mgbench_v2.qa3n import SLOTS, candidates3
from mgbench_v2.scripts.build_qa_2b import select_family

ROOT = "/lustre09/project/6002780/kunfang/music_grounding/note-level-temporal-grounding/datasets/piano-melody-3notes"
FPS = 75


def load_clips(split):
    out, maxdiff = {}, 0.0
    for r in csv.DictReader(open(f"{ROOT}/metadata.csv")):
        if r["split"] != split:
            continue
        pm = pretty_midi.PrettyMIDI(f"{ROOT}/{r['midi_path']}")
        notes = [N(n.start, n.end, n.pitch, n.velocity) for inst in pm.instruments for n in inst.notes]
        meta = json.load(open(f"{ROOT}/{r['json_path']}"))
        ev = sorted(meta["events"], key=lambda e: e["onset"])
        cl = Clip(notes)
        assert cl.n == len(ev) and [x.pitch for x in cl.notes] == [e["pitch"] for e in ev], r["audio_path"]
        maxdiff = max([maxdiff] + [abs(x.start - e["onset"]) for x, e in zip(cl.notes, ev)] + [abs(x.end - e["offset"]) for x, e in zip(cl.notes, ev)])
        cl.audio_path = r["audio_path"]
        out[os.path.basename(r["audio_path"])[:-4]] = cl
    return out, maxdiff


def frames(spans):
    return [[int(round(a * FPS)), max(int(round(b * FPS)), int(round(a * FPS)) + 1)] for a, b in spans]


def render(cand, clip, split, sid, rng):
    w = rng.choice(cand.whats)
    what, plural = w if isinstance(w, tuple) else (w, cand.plural)
    spans = [(round(a, 6), round(b, 6)) for a, b in clip.spans(cand.idx)]
    q, variants = P.render_grounding_question(rng, what, plural, [])
    ans = "; ".join(f"From {a:.2f} second to {b:.2f} second" for a, b in spans) + "." if spans else "No time span."
    fr = frames(spans)
    tok = "".join(f"<box><|T{s}|><|T{e}|></box>" for s, e in fr) if fr else "<no_time_span>"
    return {
        "question": q, "question_variants": variants, "answer": ans, "answer_token": tok,
        "query_key": f"Q/{cand.cat}/{cand.concept.upper()}/{cand.fam}:{cand.sub.split('=')[0]}",
        "answer_spans": fr, "answer_spans_sec": [list(s) for s in spans],
        "audio_path": clip.audio_path, "sample_id": sid, "split": split,
        "notes": [{"pitch": n.pitch, "velocity": n.vel, "start": round(n.start, 6), "end": round(n.end, 6)} for n in clip.notes],
        "v2": {"oracle": ORACLE_VERSION + "-3n", "family": cand.fam, "sub": cand.sub, "params": cand.prov, "answer_class": cand.akey, "n_notes": clip.n},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    all_recs, report = {}, {}
    train_sigs = None
    def sig_full(cl):
        t0 = cl.notes[0].start
        return tuple((n.pitch, n.vel, round((n.start - t0) / 0.05), round((n.end - n.start) / 0.05)) for n in cl.notes)
    def sig_pitch(cl):
        return tuple(sorted(n.pitch for n in cl.notes))
    for split in ("train", "val", "test"):
        clips, maxdiff = load_clips(split)
        if split == "train":
            train_sigs = ({sig_pitch(c) for c in clips.values()}, {sig_full(c) for c in clips.values()})
        fam_cands = defaultdict(dict)
        for sid, cl in clips.items():
            for fam, lst in candidates3(cl, sid).items():
                fam_cands[fam][sid] = lst
        recs = []
        for fam in sorted(fam_cands):
            remaining = {sid: list(l) for sid, l in fam_cands[fam].items()}
            for slot in range(SLOTS[fam]):
                cur = {sid: l for sid, l in remaining.items() if l}
                if not cur:
                    break
                sel = select_family(cur, random.Random(f"{a.seed}|{split}|{fam}|{slot}"))
                for sid, c in sel.items():
                    recs.append(render(c, clips[sid], split, sid, random.Random(f"{a.seed}|{split}|{sid}|{fam}|{slot}|render")))
                    remaining[sid] = [x for x in remaining[sid] if x.prov["key"] != c.prov["key"]]
        if split != "train":                                # melody-twin flags (reporting aid)
            for r in recs:
                cl = clips[r["sample_id"]]
                r["v2"]["twin"] = {"pitch_multiset_in_train": sig_pitch(cl) in train_sigs[0], "pitch_velocity_timing_in_train": sig_full(cl) in train_sigs[1]}
        recs.sort(key=lambda r: (r["sample_id"], r["query_key"], r["question"]))
        all_recs[split] = recs
        report[split] = {"clips": len(clips), "items": len(recs), "max_midi_vs_json_sec": round(maxdiff, 6),
                         "families": dict(Counter(r["v2"]["family"] for r in recs)), "empty_share": round(sum(1 for r in recs if not r["answer_spans"]) / len(recs), 3),
                         "by_n_notes": dict(Counter(r["v2"]["n_notes"] for r in recs))}
        print(split, report[split], flush=True)
    # test_small: one item per clip, greedily balanced over families
    rng = random.Random(f"{a.seed}|test_small")
    by_clip = defaultdict(list)
    for r in all_recs["test"]:
        by_clip[r["sample_id"]].append(r)
    cnt, cat, small = Counter(), Counter(), []
    sids = sorted(by_clip)
    rng.shuffle(sids)
    for sid in sids:
        r = min(by_clip[sid], key=lambda x: (cat[x["query_key"].split("/")[1]], cnt[x["v2"]["family"]], not x["answer_spans"] and sum(1 for s in small if not s["answer_spans"]) > 0.15 * (len(small) + 1), rng.random()))
        cnt[r["v2"]["family"]] += 1
        cat[r["query_key"].split("/")[1]] += 1
        small.append(r)
    all_recs["test_small"] = sorted(small, key=lambda r: r["sample_id"])
    keys = sorted({r["query_key"] for v in all_recs.values() for r in v})
    vocab = {k: i for i, k in enumerate(keys)}
    for split, lst in all_recs.items():
        for r in lst:
            r["query_id"] = vocab[r["query_key"]]
            r["question_token"] = f"<|Q{r['query_id']}|>"
        json.dump(lst, open(f"{a.out}/llm_{split}.json", "w"), ensure_ascii=False)
    json.dump(vocab, open(f"{a.out}/query_vocab_v2_3n.json", "w"), indent=1)
    json.dump(report, open(f"{a.out}/build_report_3n.json", "w"), indent=1)
    print("query keys", len(vocab), "test_small", len(small), dict(cat))


if __name__ == "__main__":
    main()
