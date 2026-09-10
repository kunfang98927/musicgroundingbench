"""Step 6: sanity-check meta_slim/ against its own MIDI file and against a
fresh recomputation of qa_summary - catches the two ways meta_slim could go
stale: hand-edited JSON that no longer matches the audio, or qa_summary
fields that predate a change to src/qa_summary.py.

Usage:
    python scripts/06_check_meta_slim_consistency.py <dataset_root>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import src.*` works regardless of cwd

from src.qa_summary import build_qa_summary

EPS = 1e-6


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def approx_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def infer_split(path: Path) -> str:
    for split in ("train", "val", "test"):
        if split in path.parts:
            return split
    return "unknown"


def load_midi_notes(midi_path: Path) -> Tuple[List[pretty_midi.Note], List[str]]:
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        return [], [f"midi:load_failed:{type(e).__name__}"]

    notes = sorted((n for inst in pm.instruments for n in inst.notes), key=lambda n: (n.start, n.end, n.pitch))
    if not notes:
        return [], ["midi:no_notes"]
    return notes, []


def check_events_match_midi(events: List[Dict[str, Any]], midi_notes: List[pretty_midi.Note]) -> List[str]:
    if len(events) != len(midi_notes):
        return ["events_midi:note_count_mismatch"]
    for e, n in zip(events, midi_notes):
        if int(e["pitch"]) != int(n.pitch):
            return ["events_midi:pitch_mismatch"]
        if not approx_equal(e["onset_sec"], n.start, tol=2e-3):
            return ["events_midi:onset_mismatch"]
        if not approx_equal(e["offset_sec"], n.end, tol=2e-3):
            return ["events_midi:offset_mismatch"]
    return []


def check_qa_summary_matches_recomputed(sample: Dict[str, Any]) -> List[str]:
    """The one thing this script actually re-derives: recompute qa_summary
    from basic_info/control_attributes/events and diff every key against
    what's stored, so a change to build_qa_summary() that isn't backward
    compatible gets caught here instead of silently propagating into QA text."""
    recomputed = build_qa_summary(sample["basic_info"], sample["control_attributes"], sample["events"])
    stored = sample.get("qa_summary", {})
    return [f"qa_summary:{key}_mismatch" for key in recomputed if stored.get(key) != recomputed[key]]


def inspect_one_sample(dataset_root: Path, meta_slim_path: Path) -> Dict[str, Any]:
    sample = load_json(meta_slim_path)
    midi_path = dataset_root / infer_split(meta_slim_path) / "midi" / f"{meta_slim_path.stem}.mid"

    issues: List[str] = []
    if not midi_path.exists():
        issues = ["pair:midi_missing"]
    else:
        midi_notes, midi_issues = load_midi_notes(midi_path)
        issues.extend(midi_issues)
        if not issues:
            issues.extend(check_events_match_midi(sample["events"], midi_notes))
            issues.extend(check_qa_summary_matches_recomputed(sample))

    return {"sample_id": meta_slim_path.stem, "split": infer_split(meta_slim_path), "issues": issues}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    meta_slim_files = [
        p for split in ["train", "val", "test"]
        for p in sorted((dataset_root / split / "meta_slim").glob("*.json"))
        if (dataset_root / split / "meta_slim").exists()
    ]
    if not meta_slim_files:
        raise FileNotFoundError(f"No meta_slim files found under {dataset_root}")

    results = [inspect_one_sample(dataset_root, p) for p in meta_slim_files]
    bad_results = [r for r in results if r["issues"]]
    issue_counter: Counter = Counter()
    for r in bad_results:
        issue_counter.update(r["issues"])

    output_root = dataset_root / "meta_slim_check_results"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "inconsistencies.json", "w", encoding="utf8") as f:
        json.dump(bad_results, f, indent=2, ensure_ascii=False)
    with open(output_root / "bad_samples.txt", "w", encoding="utf8") as f:
        f.write("\n".join(r["sample_id"] for r in bad_results) + ("\n" if bad_results else ""))

    total, bad = len(meta_slim_files), len(bad_results)
    print("=== Meta Slim Consistency Check ===")
    print(f"total_samples: {total}")
    print(f"bad_samples: {bad} ({100.0 * bad / max(1, total):.2f}%)")
    print("\n=== Top issues ===")
    for k, v in issue_counter.most_common(30):
        print(f"{k}: {v}")
    print(f"\n[DONE] wrote {output_root / 'inconsistencies.json'} and 'bad_samples.txt'")


if __name__ == "__main__":
    main()
