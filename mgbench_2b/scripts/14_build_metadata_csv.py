"""Step 14 (optional): write a flat metadata.csv, one row per audio clip,
with paths to every modality file and per-sample-id QA counts - handy for
a quick spreadsheet view or as a HuggingFace dataset index.

Usage:
    python scripts/14_build_metadata_csv.py <dataset_root>
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def infer_split_from_path(path: Path) -> str:
    for split in ["train", "val", "test"]:
        if split in path.parts:
            return split
    return "unknown"


def collect_wav_files(dataset_root: Path) -> List[Path]:
    files = []
    for split in ["train", "val", "test"]:
        wav_dir = dataset_root / split / "wav"
        if wav_dir.exists():
            files.extend(sorted(wav_dir.glob("*.wav")))
    return files


def maybe_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path.exists() else ""


def load_qa_counts(qa_json_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    if not qa_json_path.exists():
        return counts
    for item in load_json(qa_json_path):
        sample_id = item.get("sample_id")
        if sample_id is not None:
            counts[str(sample_id)] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    wav_files = collect_wav_files(dataset_root)
    if not wav_files:
        raise FileNotFoundError(
            f"No wav files found under {dataset_root}. Audio synthesis is out of scope for this repo - "
            "run this script against your own render, or skip it (metadata.csv is optional)."
        )

    grounding_counts: Dict[str, int] = defaultdict(int)
    understanding_counts: Dict[str, int] = defaultdict(int)
    for split in ["train", "val", "test"]:
        for k, v in load_qa_counts(dataset_root / "grounding_qa" / f"{split}.json").items():
            grounding_counts[k] += v
        for k, v in load_qa_counts(dataset_root / "understanding_qa" / f"{split}.json").items():
            understanding_counts[k] += v

    fieldnames = ["sample_id", "split", "audio_path", "midi_path", "meta_path", "meta_slim_path",
                  "grounding_facts_path", "understanding_facts_path", "grounding_qa_count", "understanding_qa_count"]
    rows = []
    split_counts: Dict[str, int] = defaultdict(int)

    for wav_path in wav_files:
        split = infer_split_from_path(wav_path)
        sample_id = wav_path.stem
        split_root = dataset_root / split
        split_counts[split] += 1

        midi_mid = split_root / "midi" / f"{sample_id}.mid"
        midi_path = midi_mid if midi_mid.exists() else split_root / "midi" / f"{sample_id}.midi"

        rows.append({
            "sample_id": sample_id, "split": split,
            "audio_path": maybe_relpath(wav_path, dataset_root),
            "midi_path": maybe_relpath(midi_path, dataset_root),
            "meta_path": maybe_relpath(split_root / "meta" / f"{sample_id}.json", dataset_root),
            "meta_slim_path": maybe_relpath(split_root / "meta_slim" / f"{sample_id}.json", dataset_root),
            "grounding_facts_path": maybe_relpath(split_root / "grounding_facts" / f"{sample_id}.grounding.json", dataset_root),
            "understanding_facts_path": maybe_relpath(split_root / "understanding_facts" / f"{sample_id}.understanding.json", dataset_root),
            "grounding_qa_count": grounding_counts[sample_id],
            "understanding_qa_count": understanding_counts[sample_id],
        })

    rows.sort(key=lambda x: (x["split"], x["sample_id"]))

    out_csv = dataset_root / "metadata.csv"
    with open(out_csv, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] wrote {out_csv}")
    print(f"num_rows: {len(rows)}")
    for split, count in split_counts.items():
        print(f"num_{split}_rows: {count}")


if __name__ == "__main__":
    main()
