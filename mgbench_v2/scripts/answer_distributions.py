"""Majority-class share and answer spread of every understanding-type answer the oracle can give (G3a feasibility)."""
import json, os
from collections import Counter, defaultdict
from mgbench_v2.oracle2b import Excerpt, frame_from_basic_info, load_notes, NOTE_NAMES

MR = "/scratch/kunfang/two_bar_dataset"; ME = "/scratch/kunfang/mgbench_regen_check/two_bar_dataset"

U = {
    "R-U1 beats_per_bar": lambda e: e.beats_per_bar_answer(),
    "R-U2 tempo": lambda e: e.tempo_answer(),
    "R-U3 note_value(bar2 first)": lambda e: e.value_name(e.bars[2][0]) if e.notes_with_value(0.5) is not None else None,
    "R-U5 shortest_fill": lambda e: e.shortest_notes_per_bar(),
    "R-U6 n_second_half_bar1": lambda e: len(e.notes_in_second_half(1)) if e.notes_in_second_half(1) is not None else None,
    "T-U1 outside C major": lambda e: len(e.notes_outside_scale(0)) if e.usable else None,
    "T-U3 distinct pcs": lambda e: len(e.distinct_pitch_classes()) if e.usable else None,
    "I-U1 count 4-semitone": lambda e: e.count_moves_of(4) if e.usable else None,
    "I-U2 largest jump": lambda e: e.largest_jump() if e.usable and e.n > 1 else None,
    "I-U3 downward": lambda e: len(e.downward_notes()) if e.usable else None,
    "B-U1 bar1 notes": lambda e: e.bar_note_count(1) if e.usable else None,
    "B-U2 bar2-bar1 notes": lambda e: e.bar_note_count(2) - e.bar_note_count(1) if e.usable else None,
    "B-U3 highest bar2-bar1": lambda e: e.highest_pitch(2) - e.highest_pitch(1) if e.usable else None,
    "H-U1 triad bar1": lambda e: e.triad_name(e.triad_of_bar(1)) if e.usable and e.triad_of_bar(1) else None,
    "H-U2 root motion": lambda e: e.root_motion() if e.usable else None,
    "Rp-U1 shift": lambda e: (e.bar_shift() or (None, None))[0] if e.usable and e.bar_shift() and e.bar_shift()[0] is not None else None,
    "Rp-U2 n differing": lambda e: e.bar_shift()[1] if e.usable and e.bar_shift() else None,
    "A-U1 drops bar1": lambda e: e.drops_to_lowest(1) if e.usable else None,
}
res = defaultdict(lambda: defaultdict(Counter))
for sp in ("train", "test"):
    for f in sorted(os.listdir(f"{MR}/{sp}/midi")):
        sid = f[:-4]; meta = json.load(open(f"{ME}/{sp}/meta/{sid}.json"))
        e = Excerpt(load_notes(f"{MR}/{sp}/midi/{sid}.mid"), frame_from_basic_info(meta["basic_info"]))
        for k, fn in U.items():
            a = fn(e)
            if a is not None:
                res[k][sp][a] += 1
for k in U:
    for sp in ("train", "test"):
        c = res[k][sp]; n = sum(c.values())
        if not n: print(f"{k:30s} {sp:5s} (none)"); continue
        top = c.most_common(3)
        print(f"{k:30s} {sp:5s} n={n:5d} classes={len(c):3d} majority={top[0][1]/n:.2f}  top3={[(a, round(v/n,2)) for a, v in top]}")
