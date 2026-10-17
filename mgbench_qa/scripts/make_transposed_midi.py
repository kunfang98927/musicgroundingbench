"""How the 200 transposed MusicGroundingBench-2B clips (ids *_tr0001...) were built.

A source clip of the same split (usable, clear bar frame, notes on the metrical grid) keeps its first bar; the second bar
repeats it exactly one bar later and transposed by s semitones (s in +-1..7, register kept within MIDI 36..96). Every
output is re-read and checked with the oracle (bar_shift() == (s, number of notes)).

The released clips are in mgbench_2b/reference/transposed, and the QA build uses those files. They were selected with an
earlier revision of the clip checks in oracle2b.py, so running this script with the current oracle picks different
source clips.

    python -m mgbench_qa.scripts.make_transposed_midi --clips two_bar_dataset --out transposed
"""
import argparse, json, os, random
import pretty_midi
from mgbench_qa.oracle2b import Excerpt, frame_from_basic_info, load_notes

QUOTA = {"train": 120, "val": 20, "test": 60}
SHIFTS = [s for s in range(-7, 8) if s != 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="2B clips from mgbench_2b/run_pipeline.sh (<split>/midi, <split>/meta)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260920)
    a = ap.parse_args()
    manifest = []
    for split, quota in QUOTA.items():
        rng = random.Random(f"{a.seed}|{split}")
        ids = sorted(f[:-4] for f in os.listdir(f"{a.clips}/{split}/midi") if f.endswith(".mid"))
        rng.shuffle(ids)
        shifts = [SHIFTS[i % len(SHIFTS)] for i in range(quota)]
        rng.shuffle(shifts)
        os.makedirs(f"{a.out}/{split}/midi", exist_ok=True)
        os.makedirs(f"{a.out}/{split}/meta", exist_ok=True)
        made = 0
        for sid in ids:
            if made == quota:
                break
            meta = json.load(open(f"{a.clips}/{split}/meta/{sid}.json"))
            fr = frame_from_basic_info(meta["basic_info"])
            e = Excerpt(load_notes(f"{a.clips}/{split}/midi/{sid}.mid"), fr)
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
