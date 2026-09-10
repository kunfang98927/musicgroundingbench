"""Step 3: physically move every sample listed in soft_blacklist.txt (written
by 02_inspect_and_blacklist.py) out of train/val/test and into a quarantine
folder, so downstream steps never see them again.

This is the only blacklist that's actually acted on automatically - see the
module docstring in 02_inspect_and_blacklist.py for why hard_blacklist and
review_list are diagnostic-only.

Usage:
    python scripts/03_quarantine_flagged_samples.py <dataset_root> [--dry_run]
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import shutil


def load_soft_blacklist(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"soft blacklist file not found: {path}\n"
            f"(did you run scripts/02_inspect_and_blacklist.py first?)"
        )
    return [line.strip() for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def infer_split_from_sample_id(sample_id: str) -> str:
    for split in ("train", "val", "test"):
        if sample_id.startswith(f"{split}_"):
            return split
    raise ValueError(f"Cannot infer split from sample_id: {sample_id}")


def move_one_file(src: Path, dst: Path, dry_run: bool) -> Tuple[bool, str]:
    if not src.exists():
        return False, "missing_source"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False, "destination_exists"
    if not dry_run:
        shutil.move(str(src), str(dst))
    return True, "moved"


def process_sample(dataset_root: Path, quarantine_root: Path, sample_id: str, dry_run: bool) -> Dict:
    split = infer_split_from_sample_id(sample_id)
    src_json, src_midi = dataset_root / split / "meta" / f"{sample_id}.json", dataset_root / split / "midi" / f"{sample_id}.mid"
    dst_json, dst_midi = quarantine_root / split / "meta" / f"{sample_id}.json", quarantine_root / split / "midi" / f"{sample_id}.mid"

    json_ok, json_status = move_one_file(src_json, dst_json, dry_run)
    midi_ok, midi_status = move_one_file(src_midi, dst_midi, dry_run)

    return {
        "sample_id": sample_id,
        "split": split,
        "status": "ok" if json_ok and midi_ok else "partial_or_failed",
        "json": {"status": json_status}, "midi": {"status": midi_status},
    }


def write_summary(quarantine_root: Path, total: int, dry_run: bool, results: List[Dict]) -> None:
    split_counter, status_counter = Counter(r["split"] for r in results), Counter(r["status"] for r in results)
    with open(quarantine_root / "move_summary.txt", "w", encoding="utf8") as f:
        f.write(f"=== Soft Blacklist Quarantine Summary ===\ndry_run: {dry_run}\n")
        f.write(f"total_requested: {total}\ntotal_processed: {len(results)}\n\n")
        f.write("=== By split ===\n" + "".join(f"{k}: {v}\n" for k, v in split_counter.items()))
        f.write("\n=== Overall status ===\n" + "".join(f"{k}: {v}\n" for k, v in status_counter.items()))
        failed = [r for r in results if r["status"] != "ok"]
        if failed:
            f.write("\n=== Failed / Partial samples ===\n")
            f.write("".join(f"{r['sample_id']} | json={r['json']['status']} | midi={r['midi']['status']}\n" for r in failed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str)
    ap.add_argument("--soft_list", type=str, default=None,
                     help="Default: <dataset_root>/inspection_results/soft_blacklist.txt")
    ap.add_argument("--quarantine_dir", type=str, default="quarantine_soft_blacklist")
    ap.add_argument("--dry_run", action="store_true", help="Only simulate moves, do not actually move files")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    soft_list_path = Path(args.soft_list) if args.soft_list else dataset_root / "inspection_results" / "soft_blacklist.txt"
    quarantine_root = dataset_root / args.quarantine_dir
    quarantine_root.mkdir(parents=True, exist_ok=True)

    sample_ids = load_soft_blacklist(soft_list_path)
    results = [process_sample(dataset_root, quarantine_root, sid, args.dry_run) for sid in sample_ids]

    with open(quarantine_root / "move_log.json", "w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    write_summary(quarantine_root, len(sample_ids), args.dry_run, results)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print("=== Quarantine soft-blacklisted samples ===")
    print(f"soft_list: {soft_list_path}")
    print(f"dry_run: {args.dry_run}")
    print(f"requested: {len(sample_ids)}  ok: {ok_count}  failed_or_partial: {len(results) - ok_count}")
    print(f"[DONE] wrote {quarantine_root / 'move_summary.txt'}")


if __name__ == "__main__":
    main()
