"""WP2: run the oracle over every released 2B MIDI and write the keep manifest + per-type eligibility.

usage: python -m mgbench_v2.scripts.build_keep_manifest [--out DIR]
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

from mgbench_v2.oracle2b import Excerpt, frame_from_basic_info, load_notes

MIDI_ROOT = "/scratch/kunfang/two_bar_dataset"
META_ROOT = "/scratch/kunfang/mgbench_regen_check/two_bar_dataset"
SPLITS = ("train", "val", "test")

# question types -> callable(excerpt) -> answer-or-None ("eligible" iff not None)
TYPES = {
    "R-G1 bar_start": lambda e: e.bar_initial_notes(),
    "R-G2 beat1": lambda e: e.notes_on_beat(1),
    "R-G2 beat2": lambda e: e.notes_on_beat(2),
    "R-G2 beat3": lambda e: e.notes_on_beat(3) if e.frame.beats_per_bar >= 3 else None,
    "R-G3 bar_middle": lambda e: e.notes_at_bar_middle(),
    "R-G4 eighth_notes": lambda e: e.notes_with_value(0.5),
    "R-G5 syncopated": lambda e: e.syncopated_notes(),
    "R-G6 longest_bar2": lambda e: e.longest_note_of_bar(2, 1.3),
    "R-U1 beats_per_bar": lambda e: e.beats_per_bar_answer(),
    "R-U2 tempo": lambda e: e.tempo_answer(),
    "R-U3 note_value": lambda e: e.value_name(e.bars[2][0]) if e.notes_with_value(0.5) is not None else None,
    "R-U5 shortest_fill": lambda e: e.shortest_notes_per_bar(),
    "R-U6 second_half_bar1": lambda e: e.notes_in_second_half(1),
    "T-U1 outside_scale": lambda e: e.notes_outside_scale(0) if e.usable else None,
    "I-G1 move1-2": lambda e: e.notes_reached_by(1, 2) if e.usable else None,
    "I-G3 move>=6": lambda e: e.notes_reached_by(6, None) if e.usable else None,
    "I-U2 largest_jump": lambda e: e.largest_jump() if e.usable and e.n > 1 else None,
    "B-U1 bar1_notes": lambda e: e.bar_note_count(1) if e.usable else None,
    "H-U1 triad_bar1": lambda e: e.triad_of_bar(1) if e.usable else None,
    "H-U2 root_motion": lambda e: e.root_motion() if e.usable else None,
    "Rp-U1 bar_shift": lambda e: e.bar_shift() if e.usable else None,
    "A-U1 drops": lambda e: e.drops_to_lowest(1) if e.usable else None,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/lustre09/project/6002780/kunfang/music_grounding/mgbench_v2_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    manifest, elig, flags = {}, defaultdict(Counter), defaultdict(Counter)
    nom_overflow_disagree = []
    per_id = {}
    for sp in SPLITS:
        ids = sorted(f[:-4] for f in os.listdir(f"{MIDI_ROOT}/{sp}/midi") if f.endswith(".mid"))
        for sid in ids:
            meta = json.load(open(f"{META_ROOT}/{sp}/meta/{sid}.json"))
            fr = frame_from_basic_info(meta["basic_info"])
            e = Excerpt(load_notes(f"{MIDI_ROOT}/{sp}/midi/{sid}.mid"), fr)
            # nominal (control-side) overflow / ultra-short criterion used in the audit, for cross-checking the measured flags
            nominal_bad = any(x["duration_beats"] * fr.sec_per_beat < 0.10 or x["within_bar_onset_beats"] + x["duration_beats"] > fr.beats_per_bar + 1e-6
                              for x in meta["events"])
            flags[sp]["nominal_bad"] += nominal_bad
            flags[sp]["nominal_bad&usable"] += nominal_bad and e.usable
            flags[sp]["nominal_ok&unusable"] += (not nominal_bad) and not e.usable
            flags[sp]["n"] += 1
            for k in ("ultra_short", "overlap", "frame_clear", "grid_ok", "values_ok", "usable"):
                flags[sp][k] += bool(getattr(e, k))
            flags[sp]["meter_" + fr.meter] += 1
            flags[sp]["conc_" + str(meta["control_attributes"].get("harmonic_pattern_type", "none"))] += 1
            row = {"usable": bool(e.usable), "frame_clear": bool(e.frame_clear), "grid_ok": bool(e.grid_ok), "values_ok": bool(e.values_ok),
                   "ultra_short": bool(e.ultra_short), "overlap": bool(e.overlap), "nominal_bad": bool(nominal_bad), "eligible": []}
            for name, fn in TYPES.items():
                try:
                    r = fn(e)
                except Exception as exc:                               # an oracle bug must be visible, not silent
                    r = None
                    elig[sp]["ERR " + name] += 1
                if r is not None:
                    elig[sp][name] += 1
                    row["eligible"].append(name)
            per_id[sid] = row
    json.dump(per_id, open(f"{a.out}/v2_keep_manifest_2b.json", "w"))
    json.dump({sp: dict(c) for sp, c in flags.items()}, open(f"{a.out}/v2_flags_2b.json", "w"), indent=1)
    json.dump({sp: dict(c) for sp, c in elig.items()}, open(f"{a.out}/v2_eligibility_2b.json", "w"), indent=1)
    print("== integrity flags (count of excerpts)")
    for sp in SPLITS:
        print(sp, {k: v for k, v in flags[sp].items() if not k.startswith(("meter_", "conc_"))})
    print("== eligibility per question type (train / val / test)")
    for name in list(TYPES) + [k for k in elig["test"] if k.startswith("ERR")]:
        print(f"{name:24s}", *(f"{elig[sp][name]:6d}" for sp in SPLITS))


if __name__ == "__main__":
    main()
