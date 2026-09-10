"""Step 16 (optional): plot sample-level and QA-level distributions
(structure mode, time signature, tempo, concept/category counts, ...) for
a final visual sanity check of the built dataset.

Usage:
    python scripts/16_plot_final_stats.py <dataset_root>
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_meta_slim_files(dataset_root: Path) -> List[Path]:
    files = []
    for split in ["train", "val", "test"]:
        d = dataset_root / split / "meta_slim"
        if d.exists():
            files.extend(sorted(d.glob("*.json")))
    return files


def load_qa_entries(dataset_root: Path, qa_dirname: str) -> List[Dict[str, Any]]:
    entries = []
    for split in ["train", "val", "test"]:
        p = dataset_root / qa_dirname / f"{split}.json"
        if p.exists():
            entries.extend(load_json(p))
    return entries


def plot_bar(counter: Counter, title: str, xlabel: str, ylabel: str, save_path: Path, top_k: int = None, rotate_xticks: bool = False) -> None:
    items = counter.most_common(top_k) if top_k is not None else list(counter.items())
    if not items:
        return
    labels, values = [k for k, _ in items], [v for _, v in items]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45 if rotate_xticks else 0, ha="right" if rotate_xticks else "center")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_hist(values: List[float], title: str, xlabel: str, ylabel: str, save_path: Path, bins: int = 40) -> None:
    if not values:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(np.array(values, dtype=np.float64), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def summarize_counter(counter: Counter) -> Dict[str, int]:
    return dict(counter.most_common())


def plot_qa_task(dataset_root: Path, qa_dirname: str, task_label: str, out_root: Path, top_k_query_keys: int) -> Dict[str, Any]:
    entries = load_qa_entries(dataset_root, qa_dirname)
    split_counter, family_counter, concept_counter, query_key_counter = Counter(), Counter(), Counter(), Counter()

    for x in entries:
        split_counter.update([str(x.get("split", "unknown"))])
        family_counter.update([str(x.get("qa_category", "unknown"))])
        concept_counter.update([str(x.get("concept", "unknown"))])
        query_key_counter.update([str(x.get("query_key", "unknown"))])

    plot_bar(split_counter, f"{task_label} QA Count by Split", "Split", "Count", out_root / f"{qa_dirname.split('_')[0]}_split_distribution.png")
    plot_bar(family_counter, f"{task_label} QA Family Distribution", "Family", "Count", out_root / f"{qa_dirname.split('_')[0]}_family_distribution.png")
    plot_bar(concept_counter, f"{task_label} QA Concept Distribution", "Concept", "Count",
              out_root / f"{qa_dirname.split('_')[0]}_concept_distribution.png", rotate_xticks=True)
    plot_bar(query_key_counter, f"{task_label} QA Top-{top_k_query_keys} Query Keys", "Query Key", "Count",
              out_root / f"{qa_dirname.split('_')[0]}_query_key_topk_distribution.png", top_k=top_k_query_keys, rotate_xticks=True)

    return {"num_entries": len(entries), "split_counts": summarize_counter(split_counter),
            "family_counts": summarize_counter(family_counter), "concept_counts": summarize_counter(concept_counter),
            "query_key_topk": dict(query_key_counter.most_common(top_k_query_keys))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str)
    ap.add_argument("--top_k_query_keys", type=int, default=30)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = dataset_root / "final_statistics"
    out_root.mkdir(parents=True, exist_ok=True)

    meta_files = collect_meta_slim_files(dataset_root)
    if not meta_files:
        raise FileNotFoundError(f"No meta_slim files found under {dataset_root}")

    split_counter, structure_mode_counter = Counter(), Counter()
    time_signature_counter, tempo_class_counter, tonal_mode_counter = Counter(), Counter(), Counter()
    repetition_type_counter, harmonic_pattern_type_counter = Counter(), Counter()
    num_notes_values, clip_duration_values = [], []

    for meta_path in meta_files:
        sample = load_json(meta_path)
        basic, control = sample["basic_info"], sample["control_attributes"]

        split_counter.update([str(sample.get("split", "unknown"))])
        structure_mode_counter.update([str(control.get("structure_mode", "unknown"))])
        time_signature_counter.update([str(basic.get("time_signature", "unknown"))])
        tempo_class_counter.update([str(basic.get("tempo_class", "unknown"))])
        tonal_mode_counter.update([str(control.get("tonal_mode", "unknown"))])
        repetition_type_counter.update([str(control.get("repetition_type", "unknown"))])
        harmonic_pattern_type_counter.update([str(control.get("harmonic_pattern_type", "unknown"))])

        if basic.get("num_notes") is not None:
            num_notes_values.append(int(basic["num_notes"]))
        if basic.get("clip_duration_sec") is not None:
            clip_duration_values.append(float(basic["clip_duration_sec"]))

    plot_bar(split_counter, "Sample Count by Split", "Split", "Count", out_root / "sample_split_distribution.png")
    plot_bar(structure_mode_counter, "Structure Mode Distribution", "Structure Mode", "Count",
              out_root / "sample_structure_mode_distribution.png", rotate_xticks=True)
    plot_bar(time_signature_counter, "Time Signature Distribution", "Time Signature", "Count", out_root / "sample_time_signature_distribution.png")
    plot_bar(tempo_class_counter, "Tempo Class Distribution", "Tempo Class", "Count", out_root / "sample_tempo_class_distribution.png")
    plot_bar(tonal_mode_counter, "Tonal Mode Distribution", "Mode", "Count", out_root / "sample_tonal_mode_distribution.png")
    plot_bar(repetition_type_counter, "Repetition Type Distribution", "Repetition Type", "Count",
              out_root / "sample_repetition_type_distribution.png", rotate_xticks=True)
    plot_bar(harmonic_pattern_type_counter, "Harmonic Pattern Type Distribution", "Harmonic Pattern Type", "Count",
              out_root / "sample_harmonic_pattern_type_distribution.png", rotate_xticks=True)
    plot_hist(num_notes_values, "Number of Notes Distribution", "Number of Notes", "Count", out_root / "sample_num_notes_hist.png", bins=30)
    plot_hist(clip_duration_values, "Clip Duration Distribution", "Duration (sec)", "Count", out_root / "sample_clip_duration_hist.png", bins=40)

    grounding_summary = plot_qa_task(dataset_root, "grounding_qa", "Grounding", out_root, args.top_k_query_keys)
    understanding_summary = plot_qa_task(dataset_root, "understanding_qa", "Understanding", out_root, args.top_k_query_keys)

    summary = {
        "sample_level": {
            "num_samples": len(meta_files), "split_counts": summarize_counter(split_counter),
            "structure_mode_counts": summarize_counter(structure_mode_counter),
            "time_signature_counts": summarize_counter(time_signature_counter),
            "tempo_class_counts": summarize_counter(tempo_class_counter),
            "tonal_mode_counts": summarize_counter(tonal_mode_counter),
            "repetition_type_counts": summarize_counter(repetition_type_counter),
            "harmonic_pattern_type_counts": summarize_counter(harmonic_pattern_type_counter),
            "num_notes_summary": {"min": int(np.min(num_notes_values)) if num_notes_values else 0,
                                    "max": int(np.max(num_notes_values)) if num_notes_values else 0,
                                    "mean": round(float(np.mean(num_notes_values)), 4) if num_notes_values else 0.0,
                                    "median": round(float(np.median(num_notes_values)), 4) if num_notes_values else 0.0},
            "clip_duration_summary": {"min_sec": round(float(np.min(clip_duration_values)), 4) if clip_duration_values else 0.0,
                                        "max_sec": round(float(np.max(clip_duration_values)), 4) if clip_duration_values else 0.0,
                                        "mean_sec": round(float(np.mean(clip_duration_values)), 4) if clip_duration_values else 0.0,
                                        "median_sec": round(float(np.median(clip_duration_values)), 4) if clip_duration_values else 0.0},
        },
        "grounding_qa": grounding_summary,
        "understanding_qa": understanding_summary,
    }
    save_json(out_root / "final_statistics_summary.json", summary)

    print("=== Final Statistics Plots ===")
    print(f"[DONE] wrote plots and summary to: {out_root}")


if __name__ == "__main__":
    main()
