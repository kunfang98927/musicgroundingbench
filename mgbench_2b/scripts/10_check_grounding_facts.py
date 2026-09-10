"""Step 10 (optional sanity check): recompute what each grounding fact's
`note_indices` *should* be from qa_summary/events, and flag any fact where
the stored value disagrees - catches a wrong branch in build_qa_facts.py
that produces a plausible-looking but semantically incorrect answer span.

Read-only; can be run any time after step 07.

Usage:
    python scripts/10_check_grounding_facts.py <dataset_root>
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def infer_split_from_path(path: Path) -> str:
    for split in ["train", "val", "test"]:
        if split in path.parts:
            return split
    return "unknown"


def collect_grounding_fact_files(dataset_root: Path) -> List[Path]:
    files: List[Path] = []
    for split in ["train", "val", "test"]:
        d = dataset_root / split / "grounding_facts"
        if d.exists():
            files.extend(sorted(d.glob("*.grounding.json")))
    return files


def corresponding_meta_slim_path(dataset_root: Path, grounding_path: Path) -> Path:
    split = infer_split_from_path(grounding_path)
    stem = grounding_path.name.replace(".grounding.json", "")
    return dataset_root / split / "meta_slim" / f"{stem}.json"


def sorted_unique_ints(xs: List[Any]) -> List[int]:
    return sorted(set(int(x) for x in xs))


def expected_grounding_note_indices(fact: Dict[str, Any], sample: Dict[str, Any]) -> Optional[List[int]]:
    control = sample["control_attributes"]
    events = sample["events"]
    qa = sample["qa_summary"]

    fact_id = str(fact.get("fact_id", ""))
    target = str(fact.get("target", ""))
    structure_mode = str(control["structure_mode"])
    repetition_type = str(control["repetition_type"])
    harmonic_pattern_type = str(control["harmonic_pattern_type"])

    if fact_id == "bar_1":
        return sorted_unique_ints(qa["bar1_note_indices"])
    if fact_id == "bar_2":
        return sorted_unique_ints(qa["bar2_note_indices"])
    if fact_id == "highest_note":
        return sorted_unique_ints(qa["highest_note_indices"])
    if fact_id == "lowest_note":
        return sorted_unique_ints(qa["lowest_note_indices"])
    if fact_id == "first_note":
        return [int(events[0]["note_index"])] if events else []
    if fact_id == "last_note":
        return [int(events[-1]["note_index"])] if events else []

    if fact_id.startswith("note_"):
        try:
            return [int(fact_id.split("_", 1)[1])]
        except Exception:
            return None

    if fact_id == "downbeat_notes":
        return sorted_unique_ints(qa["downbeat_note_indices"])
    if fact_id == "on_beat_notes":
        return sorted_unique_ints(qa["on_beat_note_indices"])
    if fact_id == "offbeat_notes":
        return sorted_unique_ints(qa["offbeat_note_indices"])

    if fact_id.startswith("beat_") and fact_id.endswith("_notes"):
        try:
            beat = fact_id.split("_")[1]
            return sorted_unique_ints(qa["beat_note_indices"].get(str(beat), []))
        except Exception:
            return None

    if fact_id == "step_target_notes":
        return sorted_unique_ints(qa["step_note_indices"])
    if fact_id == "leap_target_notes":
        return sorted_unique_ints(qa["leap_note_indices"])

    if fact_id.startswith("interval_"):
        return sorted_unique_ints(qa["interval_name_to_note_indices"].get(target, []))

    if fact_id == "bar1_chord_tone_notes":
        return sorted_unique_ints(qa["bar1_chord_tone_note_indices"])
    if fact_id == "bar2_chord_tone_notes":
        return sorted_unique_ints(qa["bar2_chord_tone_note_indices"])
    if fact_id == "bar1_non_chord_tone_notes":
        return sorted_unique_ints(qa["bar1_non_chord_tone_note_indices"])
    if fact_id == "bar2_non_chord_tone_notes":
        return sorted_unique_ints(qa["bar2_non_chord_tone_note_indices"])

    if fact_id == "repetition_pattern_notes":
        if structure_mode != "explicit_repetition":
            return None
        return sorted_unique_ints(qa["bar1_note_indices"] + qa["bar2_note_indices"])
    if fact_id == "exact_repeat_pattern_notes":
        if structure_mode != "explicit_repetition" or repetition_type != "exact_bar_repeat":
            return None
        return sorted_unique_ints(qa["bar1_note_indices"] + qa["bar2_note_indices"])
    if fact_id == "transposed_repeat_pattern_notes":
        if structure_mode != "explicit_repetition" or repetition_type != "transposed_bar_repeat":
            return None
        return sorted_unique_ints(qa["bar1_note_indices"] + qa["bar2_note_indices"])
    if fact_id == "harmonic_pattern_notes":
        if structure_mode != "explicit_harmonic_pattern":
            return None
        return sorted_unique_ints(qa["bar1_note_indices"] + qa["bar2_note_indices"])
    if fact_id == "ascending_arpeggio_pattern_notes":
        if structure_mode != "explicit_harmonic_pattern" or harmonic_pattern_type != "ascending_arpeggio":
            return None
        return sorted_unique_ints(qa["bar1_note_indices"] + qa["bar2_note_indices"])
    if fact_id == "broken_chord_repeat_pattern_notes":
        if structure_mode != "explicit_harmonic_pattern" or harmonic_pattern_type != "broken_chord_repeat":
            return None
        return sorted_unique_ints(qa["bar1_note_indices"] + qa["bar2_note_indices"])

    return None


def check_one_fact(fact: Dict[str, Any], sample: Dict[str, Any]) -> List[str]:
    expected = expected_grounding_note_indices(fact, sample)
    if expected is None:
        return ["unsupported_fact_for_semantic_check"]
    actual = sorted_unique_ints(fact.get("note_indices", []))
    return ["wrong_note_indices_for_fact"] if actual != expected else []


def inspect_one_file(dataset_root: Path, grounding_path: Path) -> Dict[str, Any]:
    sample_id = grounding_path.name.replace(".grounding.json", "")
    split = infer_split_from_path(grounding_path)
    meta_slim_path = corresponding_meta_slim_path(dataset_root, grounding_path)

    if not meta_slim_path.exists():
        return {"sample_id": sample_id, "split": split, "issues": ["pair:meta_slim_missing"], "fact_level_issues": []}

    facts_payload = load_json(grounding_path)
    sample = load_json(meta_slim_path)

    fact_level_issues: List[Dict[str, Any]] = []
    for bucket in ["high_confidence_facts", "low_confidence_facts"]:
        for i, fact in enumerate(facts_payload.get(bucket, [])):
            fact_issues = check_one_fact(fact, sample)
            if fact_issues and fact_issues != ["unsupported_fact_for_semantic_check"]:
                fact_level_issues.append({"bucket": bucket, "index": i, "fact_id": fact.get("fact_id"), "issues": fact_issues})

    issues = ["file:has_semantically_bad_facts"] if fact_level_issues else []
    return {"sample_id": sample_id, "split": split, "issues": issues, "fact_level_issues": fact_level_issues}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    files = collect_grounding_fact_files(dataset_root)
    if not files:
        raise FileNotFoundError(f"No grounding facts found under {dataset_root}")

    output_root = dataset_root / "grounding_fact_semantic_check_results"
    bad_results, issue_counter, fact_issue_counter = [], Counter(), Counter()

    for path in files:
        result = inspect_one_file(dataset_root, path)
        if result["issues"] or result["fact_level_issues"]:
            bad_results.append(result)
            issue_counter.update(result["issues"])
            for item in result["fact_level_issues"]:
                fact_issue_counter.update(item["issues"])

    total, bad = len(files), len(bad_results)
    save_json(output_root / "bad_files.json", bad_results)
    with open(output_root / "bad_files.txt", "w", encoding="utf8") as f:
        f.write("".join(f"{r['sample_id']}\n" for r in bad_results))

    print("=== Grounding Fact Semantic Check ===")
    print(f"total_files: {total}\ngood_files: {total - bad}\nbad_files: {bad}\nbad_rate_pct: {100.0 * bad / max(1, total):.2f}")
    print("\n=== Fact-level issues ===")
    for k, v in (fact_issue_counter.most_common(20) or [("none", "")]):
        print(f"{k}: {v}")
    print(f"\n[DONE] wrote {output_root}")


if __name__ == "__main__":
    main()
