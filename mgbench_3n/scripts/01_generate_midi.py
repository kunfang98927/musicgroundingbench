"""Step 3N-1: generate the MGBench-3N MIDI clips (up to three notes each).

Much simpler than the 2B generator: no control-attribute sampling, just three
template pools (single/two/three-note clips built from fixed timing centers +
small jitter) mixed 1:1:1 and split 70/15/15. Uses the GLOBAL `random`/`np.random`
modules (seeded once in `build_dataset`), matching the original exactly - this
is deliberately not rewritten to use a local `random.Random` instance, since
that would change the draw sequence and silently produce a different dataset.

Usage:
    python scripts/01_generate_midi.py --outdir piano-melody-3notes [--target 6000]
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pretty_midi

PITCH_MIN = 21      # A0
PITCH_MAX = 108     # C8
ALL_PITCHES = list(range(PITCH_MIN, PITCH_MAX + 1))  # 88 keys

MAX_CLIP_LEN = 5.0
MIN_CLIP_LEN = 1.5
MIN_NOTE_DUR = 0.08

VEL_MIN = 40
VEL_MAX = 110

ONSET_JITTER_SEC = 0.015
GAP_JITTER_SEC = 0.05
DUR_REL_JITTER = 0.12

RNG_SEED = 20260123

SINGLE_DUR_CENTERS = [0.30, 0.60, 1.00, 1.60]
SINGLE_ONSET_CENTERS = [0.30, 1.30, 2.30]

TWO_ONSET_CENTERS = [0.30, 0.50, 0.80]
TWO_DUR_CENTERS = [0.30, 0.60, 1.00]
TWO_GAP_CENTERS = [0.15, 0.40, 0.80]
INTERVALS_TWO = [0, 1, 2, 3, 5, 7, 12, 19]

THREE_ONSET_CENTERS = [0.30, 0.50, 0.80]
THREE_GAP_CENTERS = [0.15, 0.30, 0.45]
THREE_DUR_CENTERS = [0.25, 0.50, 0.80]
THREE_CONTOURS = ["up-up", "down-down", "up-down", "down-up"]
S, M, L = [1, 2, 3], [5, 7], [12, 19]

_stats = {"clip_total": 0, "fallback_used": 0}


def jitter_abs(center: float, abs_j: float) -> float:
    return center + random.uniform(-abs_j, abs_j)


def jitter_rel(center: float, rel_j: float) -> float:
    return center * (1.0 + random.uniform(-rel_j, rel_j))


def sample_velocity() -> int:
    return random.randint(VEL_MIN, VEL_MAX)


def ensure_monophonic(events: List[Dict]) -> None:
    ev = sorted(events, key=lambda e: e["onset"])
    prev_off = -1e18
    for e in ev:
        if e["onset"] < prev_off - 1e-9:
            raise ValueError("Overlap detected (should not happen).")
        prev_off = e["offset"]


def sanitize_events(events: List[Dict]) -> List[Dict]:
    out = []
    for e in events:
        on = max(0.0, float(e["onset"]))
        off = max(on + MIN_NOTE_DUR, float(e["offset"]))
        vel = max(1, min(127, int(e.get("velocity", 100))))
        out.append({"pitch": int(e["pitch"]), "onset": on, "offset": off, "velocity": vel})
    out = sorted(out, key=lambda x: x["onset"])
    ensure_monophonic(out)
    return out


def compress_gaps_and_trim_to_max(events: List[Dict], max_len: float = MAX_CLIP_LEN) -> List[Dict]:
    """Absolute fallback: guarantee last offset <= max_len. (1) pack with 0 gap,
    (2) trim durations from the end backward, (3) shift left as a last resort."""
    if not events:
        return events

    events = sanitize_events(events)
    for i in range(1, len(events)):
        prev, cur = events[i - 1], events[i]
        dur = cur["offset"] - cur["onset"]
        if cur["onset"] > prev["offset"]:
            cur["onset"] = prev["offset"]
            cur["offset"] = prev["offset"] + dur

    events = sanitize_events(events)
    last_off = max(e["offset"] for e in events)
    if last_off <= max_len + 1e-9:
        return events

    remaining = last_off - max_len
    for e in reversed(events):
        dur = e["offset"] - e["onset"]
        trim = min(remaining, max(0.0, dur - MIN_NOTE_DUR))
        if trim > 0:
            e["offset"] -= trim
            remaining -= trim
        if remaining <= 1e-9:
            break

    events = sanitize_events(events)
    last_off = max(e["offset"] for e in events)
    if last_off <= max_len + 1e-9:
        return events

    earliest_on = min(e["onset"] for e in events)
    shift_left = min(last_off - max_len + 0.02, earliest_on - 0.02)
    if shift_left > 0:
        for e in events:
            e["onset"] -= shift_left
            e["offset"] -= shift_left

    events = sanitize_events(events)
    last_off = max(e["offset"] for e in events)
    if last_off > max_len + 1e-9:
        e = events[max(range(len(events)), key=lambda i: events[i]["offset"])]
        e["offset"] = max(e["onset"] + MIN_NOTE_DUR, max_len)

    events = sanitize_events(events)
    assert max(e["offset"] for e in events) <= max_len + 1e-6
    return events


def fit_to_length(events: List[Dict], min_len: float = MIN_CLIP_LEN, max_len: float = MAX_CLIP_LEN) -> List[Dict]:
    if not events:
        return events

    events = sanitize_events(events)
    last_off = max(e["offset"] for e in events)

    if last_off < min_len:
        shift = (min_len - last_off) + 0.10
        for e in events:
            e["onset"] += shift
            e["offset"] += shift
        events = sanitize_events(events)

    if max(e["offset"] for e in events) > max_len + 1e-9:
        _stats["fallback_used"] += 1
        events = compress_gaps_and_trim_to_max(events, max_len=max_len)

    for e in events:
        e["onset"] = round(float(e["onset"]), 6)
        e["offset"] = round(float(e["offset"]), 6)
        if e["offset"] <= e["onset"]:
            e["offset"] = round(e["onset"] + MIN_NOTE_DUR, 6)
        e["velocity"] = int(e.get("velocity", 100))

    events = sanitize_events(events)
    assert max(e["offset"] for e in events) <= max_len + 1e-6
    return events


def events_to_pretty_midi(events: List[Dict], program: int = 0) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, is_drum=False)
    for e in events:
        inst.notes.append(pretty_midi.Note(velocity=int(e.get("velocity", 100)), pitch=int(e["pitch"]),
                                            start=float(e["onset"]), end=float(e["offset"])))
    pm.instruments.append(inst)
    return pm


def sample_evenly(valid: List[int], k: int) -> List[int]:
    if not valid:
        return []
    if k >= len(valid):
        return valid[:]
    idxs = np.linspace(0, len(valid) - 1, num=k, dtype=int)
    return [valid[i] for i in idxs]


# =============================================================================
# The three template pools (single / two / three notes)
# =============================================================================

def generate_single_pool(target_n: int) -> List[List[Dict]]:
    pool: List[List[Dict]] = []
    combos = [(d, o) for d in SINGLE_DUR_CENTERS for o in SINGLE_ONSET_CENTERS]

    i = 0
    while len(pool) < target_n:
        pitch = ALL_PITCHES[i % 88]
        d_center, o_center = combos[i % len(combos)]
        onset = max(0.05, jitter_abs(o_center, ONSET_JITTER_SEC))
        dur = max(MIN_NOTE_DUR, jitter_rel(d_center, DUR_REL_JITTER))

        events = [{"pitch": pitch, "onset": onset, "offset": onset + dur, "velocity": sample_velocity()}]
        pool.append(fit_to_length(events))
        _stats["clip_total"] += 1
        i += 1
    return pool


def generate_two_pool(target_n: int, k_per_interval: int = 26) -> List[List[Dict]]:
    pool: List[List[Dict]] = []

    signed_intervals = []
    for m in INTERVALS_TWO:
        signed_intervals += [m] if m == 0 else [m, -m]

    bases_by_iv = {}
    for iv in signed_intervals:
        valid_bases = [p for p in ALL_PITCHES if PITCH_MIN <= p + iv <= PITCH_MAX]
        bases_by_iv[iv] = sample_evenly(valid_bases, k_per_interval)

    i = 0
    while len(pool) < target_n:
        iv = signed_intervals[i % len(signed_intervals)]
        bases = bases_by_iv[iv]
        base = bases[(i // len(signed_intervals)) % len(bases)]

        d_center = TWO_DUR_CENTERS[i % len(TWO_DUR_CENTERS)]
        g_center = TWO_GAP_CENTERS[(i // 3) % len(TWO_GAP_CENTERS)]
        onset1_center = TWO_ONSET_CENTERS[i % len(TWO_ONSET_CENTERS)] + (i % 4) * 0.15
        onset1 = max(0.05, jitter_abs(onset1_center, ONSET_JITTER_SEC))
        dur1 = max(MIN_NOTE_DUR, jitter_rel(d_center, DUR_REL_JITTER))
        gap = max(0.05, jitter_abs(g_center, GAP_JITTER_SEC))
        onset2 = onset1 + dur1 + gap
        dur2 = max(MIN_NOTE_DUR, jitter_rel(d_center, DUR_REL_JITTER))

        events = [
            {"pitch": base, "onset": onset1, "offset": onset1 + dur1, "velocity": sample_velocity()},
            {"pitch": base + iv, "onset": onset2, "offset": onset2 + dur2, "velocity": sample_velocity()},
        ]
        pool.append(fit_to_length(events))
        _stats["clip_total"] += 1
        i += 1
    return pool


def allowed_interval_pairs() -> List[Tuple[int, int]]:
    pairs = []
    for a in (S + M + L):
        for b in (S + M + L):
            if a in L and b in L:
                continue
            pairs.append((a, b))
    return sorted(set(pairs))


def generate_three_pool(target_n: int, k_per_combo: int = 12) -> List[List[Dict]]:
    pool: List[List[Dict]] = []
    pairs = allowed_interval_pairs()

    def signed_pair(contour: str, a: int, b: int) -> Tuple[int, int]:
        return {"up-up": (a, b), "down-down": (-a, -b), "up-down": (a, -b), "down-up": (-a, b)}[contour]

    combos = [(c, *signed_pair(c, a, b)) for c in THREE_CONTOURS for (a, b) in pairs]

    bases_by_combo = {}
    for (c, ia, ib) in combos:
        valid = [p for p in ALL_PITCHES if PITCH_MIN <= p + ia <= PITCH_MAX and PITCH_MIN <= p + ia + ib <= PITCH_MAX]
        bases_by_combo[(c, ia, ib)] = sample_evenly(valid, min(k_per_combo, max(1, len(valid))))

    i = 0
    while len(pool) < target_n:
        c, ia, ib = combos[i % len(combos)]
        bases = bases_by_combo[(c, ia, ib)]
        if not bases:
            i += 1
            continue
        base = bases[(i // len(combos)) % len(bases)]

        d_center = THREE_DUR_CENTERS[i % len(THREE_DUR_CENTERS)]
        g_center = THREE_GAP_CENTERS[(i // 3) % len(THREE_GAP_CENTERS)]
        onset1_center = THREE_ONSET_CENTERS[i % len(THREE_ONSET_CENTERS)] + (i % 4) * 0.15
        onset1 = max(0.05, jitter_abs(onset1_center, ONSET_JITTER_SEC))
        dur1 = max(MIN_NOTE_DUR, jitter_rel(d_center, DUR_REL_JITTER))
        gap1 = max(0.05, jitter_abs(g_center, GAP_JITTER_SEC))
        onset2 = onset1 + dur1 + gap1
        dur2 = max(MIN_NOTE_DUR, jitter_rel(d_center, DUR_REL_JITTER))
        gap2 = max(0.05, jitter_abs(g_center, GAP_JITTER_SEC))
        onset3 = onset2 + dur2 + gap2
        dur3 = max(MIN_NOTE_DUR, jitter_rel(d_center, DUR_REL_JITTER))

        p1, p2, p3 = base, base + ia, base + ia + ib
        events = [
            {"pitch": p1, "onset": onset1, "offset": onset1 + dur1, "velocity": sample_velocity()},
            {"pitch": p2, "onset": onset2, "offset": onset2 + dur2, "velocity": sample_velocity()},
            {"pitch": p3, "onset": onset3, "offset": onset3 + dur3, "velocity": sample_velocity()},
        ]
        pool.append(fit_to_length(events))
        _stats["clip_total"] += 1
        i += 1
    return pool


# =============================================================================
# IO + splitting + main
# =============================================================================

def write_split(entries: List[List[Dict]], outdir: Path, program: int = 0, split: str = "") -> None:
    import json
    midi_dir, meta_dir = outdir / "midi", outdir / "meta"
    midi_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    for idx, events in enumerate(entries, start=1):
        n = len(events)
        pitches = "-".join(str(e["pitch"]) for e in events)
        base = f"{split}_{idx:06d}_n{n}_p{pitches}"

        events_to_pretty_midi(events, program=program).write(str(midi_dir / f"{base}.mid"))
        meta = {"filename": f"{base}.mid", "num_notes": n, "events": events}
        with open(meta_dir / f"{base}.json", "w", encoding="utf8") as f:
            json.dump(meta, f, indent=2)


def split_entries(entries: List[List[Dict]], seed: int, splits=(0.7, 0.15, 0.15)):
    rng = random.Random(seed)
    rng.shuffle(entries)
    n = len(entries)
    n_train, n_val = int(n * splits[0]), int(n * splits[1])
    return entries[:n_train], entries[n_train:n_train + n_val], entries[n_train + n_val:]


def build_dataset(outdir: Path, target_total: int, splits=(0.7, 0.15, 0.15)) -> None:
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    _stats["clip_total"] = 0
    _stats["fallback_used"] = 0

    target_each = target_total // 3
    remainder = target_total - 3 * target_each
    print(f"[INFO] target_total={target_total}, target_each={target_each}, remainder={remainder}")

    need = target_each + (1 if remainder > 0 else 0)
    single_pool = generate_single_pool(need)
    two_pool = generate_two_pool(need)
    three_pool = generate_three_pool(need)

    rng = random.Random(RNG_SEED)
    rng.shuffle(single_pool)
    rng.shuffle(two_pool)
    rng.shuffle(three_pool)

    balanced = single_pool[:target_each] + two_pool[:target_each] + three_pool[:target_each]

    extras: List[List[Dict]] = []
    if remainder > 0:
        sources = [single_pool[target_each:], two_pool[target_each:], three_pool[target_each:]]
        si = 0
        while remainder > 0:
            if sources[si]:
                extras.append(sources[si].pop())
                remainder -= 1
            si = (si + 1) % 3
    balanced += extras
    rng.shuffle(balanced)

    train, val, test = split_entries(balanced, RNG_SEED, splits=splits)

    outdir.mkdir(parents=True, exist_ok=True)
    write_split(train, outdir / "train", split="train")
    write_split(val, outdir / "val", split="val")
    write_split(test, outdir / "test", split="test")

    print(f"[DONE] wrote train={len(train)}, val={len(val)}, test={len(test)}")
    if _stats["clip_total"] > 0:
        rate = 100.0 * _stats["fallback_used"] / _stats["clip_total"]
        print(f"[STATS] fallback_used={_stats['fallback_used']} / {_stats['clip_total']} ({rate:.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--target", type=int, default=6000, help="Total clips (~1:1:1 split across n=1/2/3 notes)")
    args = ap.parse_args()
    build_dataset(outdir=Path(args.outdir), target_total=args.target)


if __name__ == "__main__":
    main()
