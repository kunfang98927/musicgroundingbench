"""Step 3N-3 (optional check): confirm train/val/test don't overlap.

Two clips "overlap" if they have identical note counts, pitches, velocities,
onsets and offsets - i.e. they're really the same clip. Read-only.

Usage:
    python scripts/03_check_splits.py --data_dir piano-melody-3notes
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_metadata_csv(csv_path: Path) -> List[Dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_splits_no_overlap(data_dir: Path, entries: List[Dict]) -> None:
    split_suffixes = {"train": set(), "val": set(), "test": set()}
    for entry in entries:
        json_path = entry["json_path"]
        with open(data_dir / json_path, "r", encoding="utf-8") as f:
            events = json.load(f)["events"]

        velocities = [n["velocity"] for n in events]
        onsets = [n["onset"] for n in events]
        offsets = [n["offset"] for n in events]

        filename = json_path.split("/")[-1]
        suffix = (filename[filename.index("_n"):].split(".json")[0]
                  + "_v" + "-".join(str(v) for v in velocities)
                  + "_o" + "-".join(f"{o:.2f}" for o in onsets)
                  + "_f" + "-".join(f"{o:.2f}" for o in offsets))
        split_suffixes[entry["split"]].add(suffix)

    overlap_with_train = split_suffixes["train"] & split_suffixes["test"]
    print(f"Overlap found between train and test sets: {len(overlap_with_train)} items"
          if overlap_with_train else "No overlap between train and test sets.")

    overlap_with_val = split_suffixes["val"] & split_suffixes["test"]
    print(f"Overlap found between val and test sets: {len(overlap_with_val)} items"
          if overlap_with_val else "No overlap between val and test sets.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=str, required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    entries = read_metadata_csv(data_dir / "metadata.csv")
    print(f"Total entries in metadata: {len(entries)}")
    check_splits_no_overlap(data_dir, entries)


if __name__ == "__main__":
    main()
