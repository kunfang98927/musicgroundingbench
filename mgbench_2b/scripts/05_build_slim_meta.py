"""Step 5: build a "slim" per-sample metadata file (meta_slim/) that keeps only
the fields QA generation actually needs, plus the derived qa_summary block
(see src/qa_summary.py) that the QA-fact builder reads from directly.

Usage:
    python scripts/05_build_slim_meta.py <dataset_root> [--overwrite]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import src.*` works regardless of cwd

from src.qa_summary import build_qa_summary, enrich_event_with_beat_fields

CORE_BASIC_FIELDS = [
    "num_bars", "num_notes", "time_signature", "beats_per_bar", "beat_unit", "meter_class",
    "tempo_bpm", "tempo_class", "bar_duration_sec", "clip_duration_sec",
    "leading_silence_sec", "music_start_sec", "music_end_sec",
]
CORE_CONTROL_FIELDS = [
    "tonal_mode", "tonic_pc", "bar1_chord_type", "bar2_chord_type",
    "structure_mode", "repetition_type", "harmonic_pattern_type",
]
CORE_DERIVED_FIELDS = [
    "bar1_note_count", "bar2_note_count", "pitch_span", "unique_pitch_count",
    "mean_duration_beats", "step_count", "skip_count", "leap_count",
    "direction_changes", "mean_velocity",
]
CORE_EVENT_FIELDS = [
    "note_index", "pitch", "onset_sec", "offset_sec", "duration_beats", "duration_symbol",
    "velocity", "bar_index", "beat_index_in_bar", "within_bar_onset_beats", "beat_phase",
    "is_on_beat", "is_offbeat", "is_downbeat", "is_strong_beat",
    "interval_from_prev", "interval_class_from_prev", "direction_from_prev",
]
TOP_LEVEL_OPTIONAL_FIELDS = ["sample_id", "split", "filename"]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def keep_fields(obj: Dict[str, Any], allowed_fields: List[str]) -> Dict[str, Any]:
    return {k: obj[k] for k in allowed_fields if k in obj}


def build_slim_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    slim: Dict[str, Any] = {k: sample[k] for k in TOP_LEVEL_OPTIONAL_FIELDS if k in sample}

    basic_info = sample["basic_info"]
    time_signature = str(basic_info["time_signature"])
    events = [enrich_event_with_beat_fields(e, time_signature) for e in sample["events"]]

    slim["basic_info"] = keep_fields(basic_info, CORE_BASIC_FIELDS)
    slim["control_attributes"] = keep_fields(sample["control_attributes"], CORE_CONTROL_FIELDS)
    slim["derived_attributes"] = keep_fields(sample["derived_attributes"], CORE_DERIVED_FIELDS)
    slim["events"] = [keep_fields(e, CORE_EVENT_FIELDS) for e in events]
    slim["qa_summary"] = build_qa_summary(sample["basic_info"], sample["control_attributes"], slim["events"])
    return slim


def collect_meta_files(dataset_root: Path) -> List[Path]:
    files: List[Path] = []
    for split in ["train", "val", "test"]:
        meta_dir = dataset_root / split / "meta"
        if meta_dir.exists():
            files.extend(sorted(meta_dir.glob("*.json")))
    return files


def infer_split(path: Path) -> str:
    for split in ("train", "val", "test"):
        if split in path.parts:
            return split
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str, help="Path to dataset root, e.g. two_bar_dataset")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing meta_slim files")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    meta_files = collect_meta_files(dataset_root)
    if not meta_files:
        raise FileNotFoundError(f"No meta json files found under {dataset_root}")

    written, skipped, split_counts = 0, 0, Counter()
    for meta_path in meta_files:
        output_path = dataset_root / infer_split(meta_path) / "meta_slim" / meta_path.name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        save_json(output_path, build_slim_sample(load_json(meta_path)))
        written += 1
        split_counts[infer_split(meta_path)] += 1

    print("=== build_slim_meta ===")
    print(f"dataset_root: {dataset_root}")
    print(f"meta files found: {len(meta_files)}")
    print(f"written: {written} ({dict(split_counts)})")
    print(f"skipped (already existed): {skipped}")


if __name__ == "__main__":
    main()
