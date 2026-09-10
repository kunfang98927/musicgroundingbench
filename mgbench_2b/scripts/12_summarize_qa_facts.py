"""Step 12 (optional report): tally grounding/understanding facts by
qa_category, concept, answer_type, and a few fine-grained attributes
(fact_id, target, interval_name, beat, answer value) - useful for
sanity-checking the label distribution and for paper-table numbers.

Read-only; can be run any time after step 07.

Usage:
    python scripts/12_summarize_qa_facts.py <dataset_root>
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_fact_files(dataset_root: Path, subdir_name: str, suffix: str) -> List[Path]:
    files: List[Path] = []
    for split in ["train", "val", "test"]:
        fact_dir = dataset_root / split / subdir_name
        if fact_dir.exists():
            files.extend(sorted(fact_dir.glob(suffix)))
    return files


def infer_split_from_path(path: Path) -> str:
    for split in ["train", "val", "test"]:
        if split in path.parts:
            return split
    return "unknown"


def update_nested_counter(nested: Dict[str, Counter], split: str, key: str, amount: int = 1) -> None:
    nested[split][key] += amount
    nested["all"][key] += amount


def normalize_for_counting(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def extract_attribute_keys(fact: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Fine-grained attribute keys for paper statistics - more specific than concept/category alone."""
    fact_id = str(fact.get("fact_id", ""))
    concept = str(fact.get("concept", ""))
    qa_category = str(fact.get("qa_category", ""))
    target = fact.get("target", None)

    out: List[Tuple[str, str]] = [("fact_id", fact_id)]
    if target is not None:
        out.append(("target", normalize_for_counting(target)))
    if "interval_name" in fact:
        out.append(("interval_name", normalize_for_counting(fact["interval_name"])))
    if "beat" in fact:
        out.append(("beat", normalize_for_counting(fact["beat"])))
    if concept in {"meter", "tempo", "mode", "tonic", "chord", "chord_progression", "harmonic_relation",
                    "repetition", "harmonic_pattern", "interval_presence", "interval_count", "beat_count",
                    "bar_comparison", "bar_sequence_relation", "pitch_extreme", "pitch_summary", "interval_summary"}:
        out.append(("answer", normalize_for_counting(fact.get("answer"))))
    out.append(("concept__category", f"{concept}__{qa_category}"))
    return out


def summarize_one_task(dataset_root: Path, subdir_name: str, suffix: str, task_name: str) -> Dict[str, Any]:
    files = collect_fact_files(dataset_root, subdir_name, suffix)

    counts_by_split, files_by_split = Counter(), Counter()
    confidence_counts: Dict[str, Counter] = defaultdict(Counter)
    qa_category_counts: Dict[str, Counter] = defaultdict(Counter)
    concept_counts: Dict[str, Counter] = defaultdict(Counter)
    answer_type_counts: Dict[str, Counter] = defaultdict(Counter)
    fine_attribute_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    structure_mode_counts: Dict[str, Counter] = defaultdict(Counter)
    focus_tag_counts: Dict[str, Counter] = defaultdict(Counter)
    total_high, total_low = 0, 0

    for path in files:
        split = infer_split_from_path(path)
        files_by_split[split] += 1
        files_by_split["all"] += 1

        payload = load_json(path)
        meta_summary = payload.get("meta_summary", {})

        structure_mode = meta_summary.get("structure_mode", None)
        if structure_mode is not None:
            update_nested_counter(structure_mode_counts, split, str(structure_mode))
        for tag in meta_summary.get("qa_focus_tags", []):
            update_nested_counter(focus_tag_counts, split, str(tag))

        high_facts = payload.get("high_confidence_facts", [])
        low_facts = payload.get("low_confidence_facts", [])
        counts_by_split[split] += len(high_facts) + len(low_facts)
        counts_by_split["all"] += len(high_facts) + len(low_facts)
        total_high += len(high_facts)
        total_low += len(low_facts)

        for confidence, facts in [("high", high_facts), ("low", low_facts)]:
            for fact in facts:
                update_nested_counter(confidence_counts, split, confidence)
                update_nested_counter(qa_category_counts, split, str(fact.get("qa_category", "UNKNOWN")))
                update_nested_counter(concept_counts, split, str(fact.get("concept", "UNKNOWN")))
                update_nested_counter(answer_type_counts, split, str(fact.get("answer_type", "UNKNOWN")))
                for attr_name, attr_value in extract_attribute_keys(fact):
                    update_nested_counter(fine_attribute_counts[attr_name], split, attr_value)

    return {
        "task_name": task_name,
        "num_files": dict(files_by_split),
        "num_facts": dict(counts_by_split),
        "confidence_counts": {k: dict(v) for k, v in confidence_counts.items()},
        "qa_category_counts": {k: dict(v) for k, v in qa_category_counts.items()},
        "concept_counts": {k: dict(v) for k, v in concept_counts.items()},
        "answer_type_counts": {k: dict(v) for k, v in answer_type_counts.items()},
        "structure_mode_counts": {k: dict(v) for k, v in structure_mode_counts.items()},
        "focus_tag_counts": {k: dict(v) for k, v in focus_tag_counts.items()},
        "fine_attribute_counts": {attr: {k: dict(v) for k, v in m.items()} for attr, m in fine_attribute_counts.items()},
        "totals": {"high_confidence_facts": total_high, "low_confidence_facts": total_low, "all_facts": total_high + total_low},
    }


def format_counter_lines(counter_dict: Dict[str, int], top_k: int = 50) -> List[str]:
    c = Counter(counter_dict)
    return [f"{k}: {v}" for k, v in c.most_common(top_k)] if c else ["none"]


def write_text_report(out_path: Path, grounding_summary: Dict[str, Any], understanding_summary: Dict[str, Any], top_k: int = 30) -> None:
    def write_task_section(f, summary: Dict[str, Any]) -> None:
        f.write(f"=== {summary['task_name'].upper()} ===\n")
        f.write(f"files_all: {summary['num_files'].get('all', 0)}\n")
        f.write(f"facts_all: {summary['num_facts'].get('all', 0)}\n")
        f.write(f"high_confidence: {summary['totals']['high_confidence_facts']}\n")
        f.write(f"low_confidence: {summary['totals']['low_confidence_facts']}\n\n")

        for split in ["train", "val", "test", "all"]:
            f.write(f"[split={split}]\nfiles: {summary['num_files'].get(split, 0)}\nfacts: {summary['num_facts'].get(split, 0)}\n\n")

        for section, key in [("qa_category_counts", "qa_category_counts"), ("concept_counts", "concept_counts"),
                              ("answer_type_counts", "answer_type_counts"), ("structure_mode_counts", "structure_mode_counts"),
                              ("focus_tag_counts", "focus_tag_counts")]:
            f.write(f"=== {section} (all) ===\n")
            f.write("\n".join(format_counter_lines(summary[key].get("all", {}), top_k)) + "\n\n")

        for attr_name in ["fact_id", "target", "interval_name", "beat", "answer", "concept__category"]:
            f.write(f"=== fine_attribute_counts:{attr_name} (all) ===\n")
            attr_counts = summary["fine_attribute_counts"].get(attr_name, {})
            f.write("\n".join(format_counter_lines(attr_counts.get("all", {}), top_k)) + "\n\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf8") as f:
        f.write("QA FACTS SUMMARY REPORT\n\n")
        write_task_section(f, grounding_summary)
        f.write("\n")
        write_task_section(f, understanding_summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str)
    ap.add_argument("--top_k", type=int, default=30, help="How many top items to keep in the text report")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    grounding_summary = summarize_one_task(dataset_root, "grounding_facts", "*.grounding.json", "grounding")
    understanding_summary = summarize_one_task(dataset_root, "understanding_facts", "*.understanding.json", "understanding")

    out_dir = dataset_root / "qa_facts_summary"
    save_json(out_dir / "grounding_summary.json", grounding_summary)
    save_json(out_dir / "understanding_summary.json", understanding_summary)
    save_json(out_dir / "combined_summary.json", {"grounding": grounding_summary, "understanding": understanding_summary})
    write_text_report(out_dir / "summary_report.txt", grounding_summary, understanding_summary, top_k=args.top_k)

    print("=== summarize_qa_facts ===")
    print(f"grounding facts (all): {grounding_summary['totals']['all_facts']}")
    print(f"understanding facts (all): {understanding_summary['totals']['all_facts']}")
    print(f"[DONE] wrote {out_dir}")


if __name__ == "__main__":
    main()
