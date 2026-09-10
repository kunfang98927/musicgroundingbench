"""Step 11 (optional sanity check): recompute what each understanding fact's
answer and evidence_note_indices *should* be from qa_summary/derived_attributes,
and flag any fact where the stored value disagrees.

Read-only; can be run any time after step 07.

Usage:
    python scripts/11_check_understanding_facts.py <dataset_root>
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def collect_understanding_fact_files(dataset_root: Path) -> List[Path]:
    files: List[Path] = []
    for split in ["train", "val", "test"]:
        d = dataset_root / split / "understanding_facts"
        if d.exists():
            files.extend(sorted(d.glob("*.understanding.json")))
    return files


def corresponding_meta_slim_path(dataset_root: Path, understanding_path: Path) -> Path:
    split = infer_split_from_path(understanding_path)
    stem = understanding_path.name.replace(".understanding.json", "")
    return dataset_root / split / "meta_slim" / f"{stem}.json"


def sorted_unique_ints(xs: List[Any]) -> List[int]:
    return sorted(set(int(x) for x in xs))


def all_note_indices(qa: Dict[str, Any]) -> List[int]:
    return sorted_unique_ints(qa["bar1_note_indices"] + qa["bar2_note_indices"])


def interval_evidence(qa: Dict[str, Any]) -> List[int]:
    ev = sorted_unique_ints(qa["step_note_indices"] + qa["leap_note_indices"])
    return ev if ev else all_note_indices(qa)


def expected_answer_and_evidence(fact: Dict[str, Any], sample: Dict[str, Any]) -> Tuple[Optional[Any], Optional[List[int]]]:
    basic, control, derived, qa = sample["basic_info"], sample["control_attributes"], sample["derived_attributes"], sample["qa_summary"]
    fact_id = str(fact.get("fact_id", ""))
    structure_mode = str(control["structure_mode"])
    all_notes = all_note_indices(qa)
    interval_notes = interval_evidence(qa)

    if fact_id == "time_signature":
        ev = sorted_unique_ints(qa["downbeat_note_indices"] + qa["on_beat_note_indices"]) or all_notes
        return basic["time_signature"], ev
    if fact_id == "tempo_bpm":
        return basic["tempo_bpm"], all_notes
    if fact_id == "tempo_class":
        return basic["tempo_class"], all_notes
    if fact_id == "tonal_mode":
        return control["tonal_mode"], all_notes
    if fact_id == "tonic_pc":
        ev = sorted_unique_ints(qa["highest_note_indices"] + qa["lowest_note_indices"]) or all_notes
        return control["tonic_pc"], ev
    if fact_id == "num_notes":
        return basic["num_notes"], all_notes
    if fact_id == "bar1_note_count":
        return derived["bar1_note_count"], all_notes
    if fact_id == "bar2_note_count":
        return derived["bar2_note_count"], all_notes
    if fact_id == "pitch_span":
        ev = sorted_unique_ints(qa["highest_note_indices"] + qa["lowest_note_indices"]) or all_notes
        return derived["pitch_span"], ev
    if fact_id == "unique_pitch_count":
        return derived["unique_pitch_count"], all_notes
    if fact_id == "mean_duration_beats":
        return derived["mean_duration_beats"], all_notes
    if fact_id == "step_count":
        return derived["step_count"], interval_notes
    if fact_id == "skip_count":
        return derived["skip_count"], interval_notes
    if fact_id == "leap_count":
        return derived["leap_count"], interval_notes
    if fact_id == "direction_changes":
        return derived["direction_changes"], interval_notes
    if fact_id == "mean_velocity":
        return derived["mean_velocity"], all_notes
    if fact_id == "highest_pitch":
        ev = sorted_unique_ints(qa["highest_note_indices"] + qa["lowest_note_indices"]) or all_notes
        return qa["highest_pitch"], ev
    if fact_id == "lowest_pitch":
        ev = sorted_unique_ints(qa["highest_note_indices"] + qa["lowest_note_indices"]) or all_notes
        return qa["lowest_pitch"], ev
    if fact_id == "ascending_interval_count":
        return qa["ascending_interval_count"], interval_notes
    if fact_id == "descending_interval_count":
        return qa["descending_interval_count"], interval_notes
    if fact_id == "repeated_pitch_count":
        return qa["repeated_pitch_count"], interval_notes

    if fact_id == "bar1_chord_type":
        if structure_mode == "explicit_harmonic_pattern":
            ev = sorted_unique_ints(qa["bar1_chord_tone_note_indices"]) or sorted_unique_ints(qa["bar1_note_indices"])
        else:
            ev = sorted_unique_ints(qa["bar1_note_indices"])
        return control["bar1_chord_type"], ev
    if fact_id == "bar2_chord_type":
        if structure_mode == "explicit_harmonic_pattern":
            ev = sorted_unique_ints(qa["bar2_chord_tone_note_indices"]) or sorted_unique_ints(qa["bar2_note_indices"])
        else:
            ev = sorted_unique_ints(qa["bar2_note_indices"])
        return control["bar2_chord_type"], ev
    if fact_id == "chord_progression":
        if structure_mode == "explicit_harmonic_pattern":
            ev = sorted_unique_ints(qa["bar1_chord_tone_note_indices"] + qa["bar2_chord_tone_note_indices"]) or all_notes
        else:
            ev = all_notes
        return qa["chord_progression"], ev
    if fact_id == "harmonic_relation_type":
        if structure_mode == "explicit_harmonic_pattern":
            ev = sorted_unique_ints(qa["bar1_chord_tone_note_indices"] + qa["bar2_chord_tone_note_indices"]) or all_notes
        else:
            ev = all_notes
        return qa["harmonic_relation_type"], ev
    if fact_id == "bar1_chord_tone_ratio":
        if structure_mode == "explicit_harmonic_pattern":
            ev = sorted_unique_ints(qa["bar1_chord_tone_note_indices"]) or sorted_unique_ints(qa["bar1_note_indices"])
        else:
            ev = sorted_unique_ints(qa["bar1_note_indices"])
        return qa["bar1_chord_tone_ratio"], ev
    if fact_id == "bar2_chord_tone_ratio":
        if structure_mode == "explicit_harmonic_pattern":
            ev = sorted_unique_ints(qa["bar2_chord_tone_note_indices"]) or sorted_unique_ints(qa["bar2_note_indices"])
        else:
            ev = sorted_unique_ints(qa["bar2_note_indices"])
        return qa["bar2_chord_tone_ratio"], ev
    if fact_id == "repetition_type":
        return control["repetition_type"], all_notes
    if fact_id == "harmonic_pattern_type":
        return control["harmonic_pattern_type"], all_notes
    if fact_id == "bar_pitch_sequence_equal":
        return qa["bar1_pitch_sequence"] == qa["bar2_pitch_sequence"], all_notes
    if fact_id == "bar_duration_sequence_equal":
        return qa["bar1_duration_sequence"] == qa["bar2_duration_sequence"], all_notes
    if fact_id == "bar_note_count_equal":
        return derived["bar1_note_count"] == derived["bar2_note_count"], all_notes
    if fact_id == "which_bar_has_more_notes":
        b1, b2 = derived["bar1_note_count"], derived["bar2_note_count"]
        return ("bar_1" if b1 > b2 else "bar_2" if b2 > b1 else "equal"), all_notes
    if fact_id == "which_bar_has_higher_peak":
        p1, p2 = qa["bar1_highest_pitch"], qa["bar2_highest_pitch"]
        return ("bar_1" if p1 > p2 else "bar_2" if p2 > p1 else "equal"), all_notes
    if fact_id == "has_downbeat_notes":
        ev = sorted_unique_ints(qa["downbeat_note_indices"] + qa["on_beat_note_indices"]) or all_notes
        return len(sorted_unique_ints(qa["downbeat_note_indices"])) > 0, ev
    if fact_id == "has_offbeat_notes":
        ev = sorted_unique_ints(qa["offbeat_note_indices"] + qa["bar1_note_indices"] + qa["bar2_note_indices"]) or all_notes
        return len(sorted_unique_ints(qa["offbeat_note_indices"])) > 0, ev
    if fact_id == "named_interval_sequence":
        return qa["named_interval_sequence"], interval_notes
    if fact_id == "directed_named_interval_sequence":
        return qa["directed_named_interval_sequence"], interval_notes
    if fact_id == "interval_name_counts":
        return qa["interval_name_counts"], interval_notes
    if fact_id.startswith("contains_interval_"):
        interval_name = fact.get("interval_name")
        return qa["interval_name_counts"].get(interval_name, 0) > 0, interval_notes
    if fact_id.startswith("count_interval_"):
        interval_name = fact.get("interval_name")
        return qa["interval_name_counts"].get(interval_name, 0), interval_notes
    if fact_id.startswith("beat_") and fact_id.endswith("_note_count"):
        beat = fact.get("beat")
        return len(sorted_unique_ints(qa["beat_note_indices"].get(str(beat), []))), all_notes

    return None, None


def check_one_fact(fact: Dict[str, Any], sample: Dict[str, Any]) -> List[str]:
    expected_answer, expected_evidence = expected_answer_and_evidence(fact, sample)
    if expected_answer is None and expected_evidence is None:
        return ["unsupported_fact_for_semantic_check"]

    issues = []
    if fact.get("answer") != expected_answer:
        issues.append("wrong_answer_for_fact")
    if sorted_unique_ints(fact.get("evidence_note_indices", [])) != expected_evidence:
        issues.append("wrong_evidence_note_indices_for_fact")
    return issues


def inspect_one_file(dataset_root: Path, understanding_path: Path) -> Dict[str, Any]:
    sample_id = understanding_path.name.replace(".understanding.json", "")
    split = infer_split_from_path(understanding_path)
    meta_slim_path = corresponding_meta_slim_path(dataset_root, understanding_path)

    if not meta_slim_path.exists():
        return {"sample_id": sample_id, "split": split, "issues": ["pair:meta_slim_missing"], "fact_level_issues": []}

    facts_payload = load_json(understanding_path)
    sample = load_json(meta_slim_path)

    fact_level_issues: List[Dict[str, Any]] = []
    for bucket in ["high_confidence_facts", "low_confidence_facts"]:
        for i, fact in enumerate(facts_payload.get(bucket, [])):
            fact_issues = check_one_fact(fact, sample)
            if fact_issues and fact_issues != ["unsupported_fact_for_semantic_check"]:
                fact_level_issues.append({"bucket": bucket, "index": i, "fact_id": fact.get("fact_id"),
                                           "concept": fact.get("concept"), "issues": fact_issues})

    issues = ["file:has_semantically_bad_facts"] if fact_level_issues else []
    return {"sample_id": sample_id, "split": split, "issues": issues, "fact_level_issues": fact_level_issues}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    files = collect_understanding_fact_files(dataset_root)
    if not files:
        raise FileNotFoundError(f"No understanding facts found under {dataset_root}")

    output_root = dataset_root / "understanding_fact_semantic_check_results"
    bad_results = []
    issue_counter, fact_issue_counter = Counter(), Counter()
    bad_fact_id_counter, bad_concept_counter = Counter(), Counter()
    bad_fact_id_issue_counter: Dict[str, Counter] = defaultdict(Counter)

    for path in files:
        result = inspect_one_file(dataset_root, path)
        if result["issues"] or result["fact_level_issues"]:
            bad_results.append(result)
            issue_counter.update(result["issues"])
            for item in result["fact_level_issues"]:
                fact_issue_counter.update(item["issues"])
                bad_fact_id_counter.update([str(item.get("fact_id"))])
                bad_concept_counter.update([str(item.get("concept"))])
                bad_fact_id_issue_counter[str(item.get("fact_id"))].update(item["issues"])

    total, bad = len(files), len(bad_results)
    save_json(output_root / "bad_files.json", bad_results)
    with open(output_root / "bad_files.txt", "w", encoding="utf8") as f:
        f.write("".join(f"{r['sample_id']}\n" for r in bad_results))
    save_json(output_root / "debug_summary.json", {
        "bad_fact_id_counter": dict(bad_fact_id_counter),
        "bad_concept_counter": dict(bad_concept_counter),
        "bad_fact_id_issue_counter": {k: dict(v) for k, v in bad_fact_id_issue_counter.items()},
    })

    print("=== Understanding Fact Semantic Check ===")
    print(f"total_files: {total}\ngood_files: {total - bad}\nbad_files: {bad}\nbad_rate_pct: {100.0 * bad / max(1, total):.2f}")
    print("\n=== Fact-level issues ===")
    for k, v in (fact_issue_counter.most_common(20) or [("none", "")]):
        print(f"{k}: {v}")
    print("\n=== Most problematic fact_ids ===")
    for k, v in (bad_fact_id_counter.most_common(20) or [("none", "")]):
        print(f"{k}: {v}")
    print(f"\n[DONE] wrote {output_root}")


if __name__ == "__main__":
    main()
