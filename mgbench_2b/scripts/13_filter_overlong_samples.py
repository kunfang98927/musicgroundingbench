"""Step 13 (requires rendered audio - see docs/REPRODUCING.md): remove
samples whose *rendered* clip is too long for the MERT audio encoder
(feature_frames = duration_sec * fps > max_frames), across every modality
directory for that sample_id.

Audio synthesis (PianoTeq) is out of scope for this repo, so this script
only runs if you already have `<split>/wav/*.wav` files from your own
render. The official release used --max_frames 750 --fps 75.0 (10.0s) and
removed exactly 32 samples (27 train / 2 val / 3 test) from the pre-filter
9034/1130/1127 split, giving the published 9007/1128/1124 counts - see
reference/official_overlong_excluded_ids.txt for that exact list, which
lets you reproduce the official final counts without rendering audio
yourself, as long as you generated with the documented default seed.

Usage:
    python scripts/13_filter_overlong_samples.py <dataset_root> [--dry_run]
    # no rendered audio? reproduce the official counts directly instead:
    python scripts/13_filter_overlong_samples.py <dataset_root> \\
        --from_id_list reference/official_overlong_excluded_ids.txt
"""

import argparse
import json
import shutil
import wave
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_MODALITY_DIRS = ["wav", "midi", "meta", "meta_slim", "grounding_facts", "understanding_facts"]


def get_wav_num_frames_and_sr(wav_path: Path) -> Tuple[int, int]:
    with wave.open(str(wav_path), "rb") as wf:
        return int(wf.getnframes()), int(wf.getframerate())


def safe_move(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def candidate_files_for_sample(split_root: Path, stem: str) -> List[Path]:
    mapping = {
        "wav": [f"{stem}.wav"], "midi": [f"{stem}.mid", f"{stem}.midi"],
        "meta": [f"{stem}.json"], "meta_slim": [f"{stem}.json"],
        "grounding_facts": [f"{stem}.grounding.json"], "understanding_facts": [f"{stem}.understanding.json"],
    }
    return [split_root / subdir / name for subdir, names in mapping.items() for name in names]


def load_id_list(path: Path) -> List[str]:
    with open(path, "r", encoding="utf8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str, help="Path to dataset root, e.g. two_bar_dataset")
    ap.add_argument("--max_frames", type=int, default=750, help="Maximum allowed audio frames at feature fps")
    ap.add_argument("--fps", type=float, default=75.0, help="Feature fps used by MERT")
    ap.add_argument("--quarantine_dirname", type=str, default="filtered_overlong")
    ap.add_argument("--dry_run", action="store_true", help="Only report what would be moved, without moving files")
    ap.add_argument("--from_id_list", type=str, default=None,
                    help="Skip measuring audio; quarantine exactly these sample_ids instead "
                         "(e.g. reference/official_overlong_excluded_ids.txt, if you have no rendered audio)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    quarantine_root = dataset_root / args.quarantine_dirname
    max_sec = float(args.max_frames) / float(args.fps)

    manifest: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    total_overlong = 0

    print("=== Filter Overlong Samples ===")
    print(f"max_frames: {args.max_frames}  fps: {args.fps}  max_sec: {max_sec:.6f}  dry_run: {args.dry_run}")

    if args.from_id_list:
        ids_by_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        for sample_id in load_id_list(Path(args.from_id_list)):
            split = sample_id.split("_", 1)[0]
            if split in ids_by_split:
                ids_by_split[split].append(sample_id)
        print(f"from_id_list: {args.from_id_list} ({sum(len(v) for v in ids_by_split.values())} ids)")
    else:
        ids_by_split = None

    for split in ["train", "val", "test"]:
        if ids_by_split is not None:
            stems = ids_by_split[split]
        else:
            wav_dir = dataset_root / split / "wav"
            if not wav_dir.exists():
                print(f"[WARN] missing wav dir: {wav_dir} (audio synthesis is out of scope for this repo - "
                      f"pass --from_id_list reference/official_overlong_excluded_ids.txt instead)")
                continue
            stems = []
            for wav_path in sorted(wav_dir.glob("*.wav")):
                n_samples, sr = get_wav_num_frames_and_sr(wav_path)
                feature_frames = (float(n_samples) / float(sr)) * args.fps
                if feature_frames > args.max_frames:
                    stems.append(wav_path.stem)

        split_overlong = 0
        for stem in stems:
            split_overlong += 1
            total_overlong += 1
            record = {"sample_id": stem, "moved_files": []}

            for src in candidate_files_for_sample(dataset_root / split, stem):
                if not src.exists():
                    continue
                rel = src.relative_to(dataset_root)
                dst = quarantine_root / rel
                if safe_move(src, dst) if not args.dry_run else True:
                    record["moved_files"].append(str(rel))

            manifest[split].append(record)

        print(f"[{split}] overlong samples: {split_overlong}")

    quarantine_root.mkdir(parents=True, exist_ok=True)
    with open(quarantine_root / "overlong_manifest.json", "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(quarantine_root / "summary.json", "w", encoding="utf8") as f:
        json.dump({"dataset_root": str(dataset_root), "max_frames": args.max_frames, "fps": args.fps, "max_sec": max_sec,
                    "dry_run": args.dry_run, "counts": {s: len(manifest[s]) for s in ["train", "val", "test"]},
                    "total_overlong": total_overlong}, f, indent=2, ensure_ascii=False)

    print(f"\nTotal overlong samples: {total_overlong}")
    print(f"[DONE] wrote {quarantine_root}")
    if not args.dry_run and total_overlong > 0:
        print("\nRecommended next steps (the re-entrant loop - see docs/REPRODUCING.md):\n"
              "1. Re-run scripts/07_build_qa_facts.py --overwrite\n"
              "2. Re-run scripts/08_render_grounding_qa.py and scripts/09_render_understanding_qa.py --overwrite\n"
              "3. Then extract MERT features on the filtered dataset")


if __name__ == "__main__":
    main()
