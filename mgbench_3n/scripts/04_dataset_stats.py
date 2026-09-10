"""Step 3N-4 (optional check): dataset statistics and sanity checks -
per-split note-count distribution, pitch coverage, interval/contour
histograms, clip-length distribution, and monophonic/timing sanity. Read-only.

Usage:
    python scripts/04_dataset_stats.py --data_root piano-melody-3notes
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

PITCH_MIN = 21
PITCH_MAX = 108


def contour_from_pitches(p1: int, p2: int, p3: int) -> str:
    def sign(d):
        return "up" if d > 0 else ("down" if d < 0 else "eq")
    return f"{sign(p2 - p1)}-{sign(p3 - p2)}"


def read_split(meta_dir: Path):
    return [json.load(open(fp, encoding="utf8")) for fp in sorted(meta_dir.glob("*.json"))]


def summarize(rows, split_name: str):
    n_dist, pitch_counts, interval_counts, contour_counts = Counter(), Counter(), Counter(), Counter()
    lengths = []
    overlap_violations = negative_time = too_long = 0

    for r in rows:
        n = int(r["num_notes"])
        n_dist[n] += 1
        events = sorted(r["events"], key=lambda e: e["onset"])

        for e in events:
            pitch_counts[int(e["pitch"])] += 1

        length = float(max(e["offset"] for e in events))
        lengths.append(length)
        if length > 5.0 + 1e-6:
            too_long += 1

        prev_off = -1e9
        for e in events:
            on, off = float(e["onset"]), float(e["offset"])
            if on < -1e-9 or off < -1e-9 or off <= on:
                negative_time += 1
            if on < prev_off - 1e-9:
                overlap_violations += 1
            prev_off = off

        if n == 2:
            interval_counts[int(events[1]["pitch"]) - int(events[0]["pitch"])] += 1
        elif n == 3:
            p1, p2, p3 = (int(events[i]["pitch"]) for i in range(3))
            contour_counts[contour_from_pitches(p1, p2, p3)] += 1

    lengths = np.array(lengths, dtype=float) if lengths else np.array([0.0])
    pitch_covered = sum(1 for p in range(PITCH_MIN, PITCH_MAX + 1) if pitch_counts[p] > 0)

    return {
        "split": split_name, "num_clips": len(rows), "n_dist": dict(n_dist),
        "pitch_covered_88": pitch_covered,
        "length_min": float(np.min(lengths)), "length_mean": float(np.mean(lengths)),
        "length_median": float(np.median(lengths)), "length_p95": float(np.percentile(lengths, 95)),
        "length_max": float(np.max(lengths)),
        "overlap_violations": overlap_violations, "negative_time_records": negative_time,
        "too_long_>5s": too_long,
        "interval_counts": dict(interval_counts), "contour_counts": dict(contour_counts),
    }, pitch_counts


def print_report(stats) -> None:
    print(f"\n=== Split: {stats['split']} ===")
    print(f"Clips: {stats['num_clips']}")
    print(f"n_dist: {stats['n_dist']}")
    print(f"Pitch covered (21..108): {stats['pitch_covered_88']} / 88")
    print(f"Max note offset: min={stats['length_min']:.3f}, mean={stats['length_mean']:.3f}, "
          f"median={stats['length_median']:.3f}, p95={stats['length_p95']:.3f}, max={stats['length_max']:.3f}")
    print(f"Sanity: overlap_violations={stats['overlap_violations']}, "
          f"negative_time_records={stats['negative_time_records']}, "
          f"max_offset_too_large_>5s={stats['too_long_>5s']}")
    if stats["interval_counts"]:
        print(f"Top intervals (2-note) [interval:count] (top10): {Counter(stats['interval_counts']).most_common(10)}")
    if stats["contour_counts"]:
        print(f"Contours (3-note): {stats['contour_counts']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=str, required=True)
    args = ap.parse_args()
    root = Path(args.data_root)

    all_pitch_counts = Counter()
    for split in ["train", "val", "test"]:
        meta_dir = root / split / "meta"
        if not meta_dir.exists():
            raise FileNotFoundError(f"Missing: {meta_dir}")
        stats, pitch_counts = summarize(read_split(meta_dir), split)
        print_report(stats)
        all_pitch_counts.update(pitch_counts)

    covered = sum(1 for p in range(PITCH_MIN, PITCH_MAX + 1) if all_pitch_counts[p] > 0)
    counts = np.array([all_pitch_counts[p] for p in range(PITCH_MIN, PITCH_MAX + 1)], dtype=int)
    print("\n=== Global Pitch Coverage (21..108) ===")
    print(f"Covered: {covered}/88")
    print(f"Counts per pitch: min={counts.min()}, median={int(np.median(counts))}, max={counts.max()}")


if __name__ == "__main__":
    main()
