"""Step 15 (optional): pick a small, concept-balanced subset of an existing
QA json - one entry per sample_id, greedily choosing whichever remaining
sample_id/entry pair currently most under-represents a rare concept. Useful
for a quick human-eval or demo split without the full test set.

Like the render scripts, the selection order depends on an exact sequence
of `rng.shuffle()` calls - preserved closely here rather than restructured.

Usage:
    python scripts/15_sample_small_testset.py <input_json> <output_json> --target_num_samples 100
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def group_by_sample_id(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in data:
        out[str(item.get("sample_id", "unknown"))].append(item)
    return out


def concept_counter(data: List[Dict[str, Any]]) -> Counter:
    return Counter(str(x.get("concept", "unknown")) for x in data)


def ratio_dict(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    return {k: v / total for k, v in sorted(counter.items())} if total else {}


def choose_balanced_subset_one_per_sample(
    data: List[Dict[str, Any]], target_num_samples: int, seed: int = 20260417,
) -> List[Dict[str, Any]]:
    """Select target_num_samples sample_ids (or all, if fewer are available),
    keeping exactly one entry per selected sample_id, greedily preferring
    the entry whose concept is currently most under-selected (and, among
    ties, the globally rarer concept)."""
    rng = random.Random(seed)

    by_sample = group_by_sample_id(data)
    all_sample_ids = list(by_sample.keys())
    target_num_samples = min(target_num_samples, len(all_sample_ids))
    if target_num_samples <= 0:
        return []

    global_concept_freq = concept_counter(data)

    selected_items: List[Dict[str, Any]] = []
    selected_sample_ids = set()
    selected_concepts: Counter = Counter()

    while len(selected_items) < target_num_samples:
        remaining_sample_ids = [sid for sid in all_sample_ids if sid not in selected_sample_ids]
        if not remaining_sample_ids:
            break

        best_sid, best_item, best_score = None, None, None
        rng.shuffle(remaining_sample_ids)

        for sid in remaining_sample_ids:
            candidates = by_sample[sid][:]
            rng.shuffle(candidates)
            candidates.sort(key=lambda x: (
                selected_concepts[str(x.get("concept", "unknown"))],
                global_concept_freq[str(x.get("concept", "unknown"))],
            ))
            candidate = candidates[0]
            concept = str(candidate.get("concept", "unknown"))
            score = (selected_concepts[concept], global_concept_freq[concept], rng.random())

            if best_score is None or score < best_score:
                best_score, best_sid, best_item = score, sid, candidate

        selected_sample_ids.add(best_sid)
        selected_items.append(best_item)
        selected_concepts[str(best_item.get("concept", "unknown"))] += 1

    return selected_items


def summarize_source_and_sampled(source_data: List[Dict[str, Any]], sampled_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_by_sample = group_by_sample_id(source_data)
    sampled_by_sample = group_by_sample_id(sampled_data)
    source_concepts = concept_counter(source_data)
    sampled_concepts = concept_counter(sampled_data)

    selected_sample_id_to_concept = {str(x.get("sample_id", "unknown")): str(x.get("concept", "unknown")) for x in sampled_data}

    return {
        "source": {"num_entries": len(source_data), "num_sample_ids": len(source_by_sample),
                    "concept_counts": dict(sorted(source_concepts.items())), "concept_ratios": ratio_dict(source_concepts)},
        "sampled": {"num_entries": len(sampled_data), "num_sample_ids": len(sampled_by_sample),
                     "concept_counts": dict(sorted(sampled_concepts.items())), "concept_ratios": ratio_dict(sampled_concepts)},
        "selected_sample_id_to_concept": dict(sorted(selected_sample_id_to_concept.items())),
        "checks": {"all_sample_ids_unique_in_sampled": len(sampled_data) == len(sampled_by_sample)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_json", type=str, help="Path to original QA json")
    ap.add_argument("output_json", type=str, help="Path to sampled output json")
    ap.add_argument("--target_num_samples", type=int, required=True, help="How many sample_ids to keep")
    ap.add_argument("--seed", type=int, default=20260417)
    args = ap.parse_args()

    input_path, output_path = Path(args.input_json), Path(args.output_json)
    data = load_json(input_path)
    sampled = choose_balanced_subset_one_per_sample(data, args.target_num_samples, args.seed)
    save_json(output_path, sampled)

    stats = summarize_source_and_sampled(data, sampled)
    stats.update({"input_json": str(input_path), "output_json": str(output_path),
                  "target_num_samples": args.target_num_samples, "seed": args.seed})
    stats_path = output_path.with_suffix(".summary.json")
    save_json(stats_path, stats)

    print(f"[DONE] wrote sampled json: {output_path}")
    print(f"[DONE] wrote summary json: {stats_path}")
    print("=== Summary ===")
    print(f"source num_entries: {stats['source']['num_entries']}  num_sample_ids: {stats['source']['num_sample_ids']}")
    print(f"sampled num_entries: {stats['sampled']['num_entries']}  num_sample_ids: {stats['sampled']['num_sample_ids']}")
    print(f"unique check: {stats['checks']['all_sample_ids_unique_in_sampled']}")
    print("\n[Sampled concept counts]")
    for k, v in stats["sampled"]["concept_counts"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
