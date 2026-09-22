"""WP7: build ~200 *true* transposed two-bar excerpts (bar 2 = bar 1 shifted by s semitones, same rhythm) from existing excerpts.

Source excerpt = usable, frame-clear, grid-clean; only its first bar is kept, the second bar is bar 1 moved by exactly one bar length in time and by
s semitones in pitch (s in +-1..7, register kept within MIDI 36..96). Output is verified with the oracle (bar_shift() == (s, n_notes)).
"""
import argparse, json, os, random
import pretty_midi
from mgbench_v2.oracle2b import Excerpt, frame_from_basic_info, load_notes

MIDI_ROOT = "/scratch/kunfang/two_bar_dataset"
META_ROOT = "/scratch/kunfang/mgbench_regen_check/two_bar_dataset"
QUOTA = {"train": 120, "val": 20, "test": 60}
SHIFTS = [s for s in range(-7, 8) if s != 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/lustre09/project/6002780/kunfang/music_grounding/mgbench_v2_out/new_transposed")
    ap.add_argument("--seed", type=int, default=20260920)
    a = ap.parse_args()
    manifest = []
    for split, quota in QUOTA.items():
        rng = random.Random(f"{a.seed}|{split}")
        ids = sorted(f[:-4] for f in os.listdir(f"{MIDI_ROOT}/{split}/midi") if f.endswith(".mid"))
        rng.shuffle(ids)
        shifts = [SHIFTS[i % len(SHIFTS)] for i in range(quota)]
        rng.shuffle(shifts)
        os.makedirs(f"{a.out}/{split}/midi", exist_ok=True)
        os.makedirs(f"{a.out}/{split}/meta", exist_ok=True)
        made = 0
        for sid in ids:
            if made == quota:
                break
            meta = json.load(open(f"{META_ROOT}/{split}/meta/{sid}.json"))
            fr = frame_from_basic_info(meta["basic_info"])
            e = Excerpt(load_notes(f"{MIDI_ROOT}/{split}/midi/{sid}.mid"), fr)
            if not (e.usable and e.frame_clear and e.grid_ok and e.values_ok and e.tempo_answer() is not None):
                continue
            b1 = [e.notes[i] for i in e.bars[1]]
            if not (4 <= len(b1) <= 8) or len({n.pitch for n in b1}) < 3:
                continue
            if max(n.end for n in b1) > fr.music_start + fr.bar_dur - 0.02:      # bar 1 must end inside its own bar
                continue
            s = shifts[made]
            if min(n.pitch for n in b1) + s < 36 or max(n.pitch for n in b1) + s > 96:
                alt = [x for x in SHIFTS if 36 <= min(n.pitch for n in b1) + x and max(n.pitch for n in b1) + x <= 96]
                if not alt:
                    continue
                s = min(alt, key=lambda x: abs(x - s))
            pm = pretty_midi.PrettyMIDI(resolution=220, initial_tempo=120)
            inst = pretty_midi.Instrument(program=0)
            for n in b1:
                inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end))
            for n in b1:
                inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch + s, start=n.start + fr.bar_dur, end=n.end + fr.bar_dur))
            pm.instruments.append(inst)
            new_id = f"{split}_tr{made + 1:04d}"
            path = f"{a.out}/{split}/midi/{new_id}.mid"
            pm.write(path)
            # verify with the oracle on the written file
            e2 = Excerpt(load_notes(path), fr)
            ok = (e2.usable and e2.frame_clear and e2.grid_ok and e2.values_ok and e2.tempo_answer() is not None and e2.bar_shift() == (s, len(b1))
                  and len(e2.bars[1]) == len(e2.bars[2]) == len(b1))
            if not ok:
                os.remove(path)
                continue
            new_meta = {"sample_id": new_id, "split": split, "filename": f"{new_id}.mid", "source_sample_id": sid,
                        "basic_info": dict(meta["basic_info"], num_notes=2 * len(b1), music_end_sec=e2.notes[-1].end, clip_duration_sec=e2.notes[-1].end),
                        "construction": {"kind": "true_transposition", "semitones": s, "bar2_time_offset_sec": fr.bar_dur, "source_bar": 1}}
            json.dump(new_meta, open(f"{a.out}/{split}/meta/{new_id}.json", "w"), indent=1)
            manifest.append({"id": new_id, "split": split, "source": sid, "semitones": s, "meter": fr.meter, "tempo_bpm": fr.tempo_bpm, "n_notes": 2 * len(b1)})
            made += 1
        print(split, made, "of", quota)
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
    from collections import Counter
    print(Counter(m["semitones"] for m in manifest))


if __name__ == "__main__":
    main()
