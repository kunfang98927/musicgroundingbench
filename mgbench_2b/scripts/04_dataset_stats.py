"""Step 4 (optional): print a distribution report over every sample currently
in train/val/test - control-variable counts, note-count/duration percentiles,
and the most common concept tags. Purely informational: nothing downstream
reads this report, so it's safe to run at any point in the pipeline to sanity
-check where things stand.

Usage:
    python scripts/04_dataset_stats.py <dataset_root> [--output_txt path]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ReportWriter:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def add_header(self, title: str) -> None:
        self.lines += ["", f"=== {title} ==="]

    def render(self) -> str:
        return "\n".join(self.lines).strip() + "\n"


def counter_to_sorted_list(counter: Counter) -> List[Tuple[Any, int]]:
    return sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))


def summarize_numeric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)
    n = len(vals)

    def percentile(p: float) -> float:
        idx = p * (n - 1)
        lo, frac = int(idx), idx - int(idx)
        return vals[lo] * (1 - frac) + vals[min(lo + 1, n - 1)] * frac

    return {"count": n, "min": vals[0], "p25": percentile(0.25), "median": percentile(0.50),
            "p75": percentile(0.75), "max": vals[-1], "mean": sum(vals) / n}


def add_counter_section(report: ReportWriter, title: str, counter: Counter, total: int) -> None:
    report.add_header(title)
    for k, v in counter_to_sorted_list(counter):
        report.add(f"{k}: {v} ({100.0 * v / total if total else 0.0:.2f}%)")


def add_numeric_section(report: ReportWriter, title: str, values: List[float]) -> None:
    report.add_header(title)
    stats = summarize_numeric(values)
    if not stats:
        report.add("No values")
        return
    for k, v in stats.items():
        report.add(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")


CONTROL_COUNTER_FIELDS = [
    "time_signature", "tempo_class", "density_level", "tonal_mode", "interval_profile",
    "bar_relation_type", "structure_mode", "meter_clarity", "repetition_type",
    "accent_pattern_type", "harmonic_pattern_type",
]
NUMERIC_FIELDS = [
    "num_notes", "bar_duration_sec", "clip_duration_sec", "leading_silence_sec",
    "music_start_sec", "music_end_sec", "bar1_note_count", "bar2_note_count", "pitch_span",
    "unique_pitch_count", "mean_duration_beats", "step_count", "skip_count", "leap_count",
    "direction_changes", "mean_velocity",
]


def build_report(dataset_root: Path) -> str:
    json_files = sorted(dataset_root.glob("*/meta/*.json"))
    if not json_files:
        raise FileNotFoundError(f"No metadata json files found under: {dataset_root}")

    split_counter: Counter = Counter()
    counters: Dict[str, Counter] = defaultdict(Counter)
    numeric: Dict[str, List[float]] = defaultdict(list)
    concept_counter: Counter = Counter()
    total_events = 0

    for path in json_files:
        data = json.loads(path.read_text(encoding="utf8"))
        basic, control, derived = data["basic_info"], data["control_attributes"], data["derived_attributes"]

        split_counter[data.get("split", path.parts[-3])] += 1
        for field in CONTROL_COUNTER_FIELDS:
            source = basic if field in ("time_signature", "tempo_class") else control
            counters[field][source[field]] += 1
        for field in NUMERIC_FIELDS:
            source = basic if field in basic else derived
            numeric[field].append(float(source[field]))
        for c in data.get("concepts", []):
            concept_counter[c] += 1
        total_events += len(data["events"])

    total_samples = len(json_files)
    report = ReportWriter()
    report.add(f"Dataset root: {dataset_root}")
    report.add(f"Total samples: {total_samples}")
    report.add(f"Total note events: {total_events}")

    add_counter_section(report, "Split distribution", split_counter, total_samples)
    for field in CONTROL_COUNTER_FIELDS:
        add_counter_section(report, f"{field} distribution", counters[field], total_samples)
    for field in NUMERIC_FIELDS:
        add_numeric_section(report, f"{field} summary", numeric[field])

    report.add_header("Top concepts")
    for k, v in counter_to_sorted_list(concept_counter)[:60]:
        report.add(f"{k}: {v} ({100.0 * v / total_samples if total_samples else 0.0:.2f}%)")

    return report.render()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str)
    ap.add_argument("--output_txt", type=str, default=None)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    output_txt = Path(args.output_txt) if args.output_txt else dataset_root / "dataset_summary.txt"

    report = build_report(dataset_root)
    output_txt.write_text(report, encoding="utf8")
    print(report)
    print(f"[DONE] wrote {output_txt}")
