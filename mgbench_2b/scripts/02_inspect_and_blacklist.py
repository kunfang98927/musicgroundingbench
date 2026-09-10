"""Step 2: inspect every generated sample and classify it as clean / hard /
soft / review.

  hard   - structural problems (missing fields, broken JSON/MIDI pairing,
           corrupt events). Nothing currently reads hard_blacklist back in;
           it's a diagnostic dump. In practice this is always empty, because
           validate_sample() in src/validation.py already rejects anything
           this broken before it's ever written to disk.
  soft   - passes structurally but fails a quality threshold (e.g. fewer
           than 5 notes, a single unique pitch). soft_blacklist.txt IS read
           by 03_quarantine_flagged_samples.py, which physically moves these
           samples out of the dataset.
  review - borderline quality (e.g. an unusually wide pitch span). Written
           for a human to skim; nothing in the pipeline reads it back in, so
           leaving review_list.json un-acted-on does not change the dataset.

Usage:
    python scripts/02_inspect_and_blacklist.py <dataset_root>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import src.*` works regardless of cwd

from src.music_theory import chord_pitch_classes

EPS = 1e-6

HARD_PREFIXES = ("missing_top:", "missing_basic:", "json:", "events:", "basic:", "midi:", "pair:")
SOFT_QUALITY_FLAGS = {
    "quality:too_few_notes", "quality:zero_pitch_span", "quality:single_unique_pitch",
    "quality:exact_repeat_bar_note_count_mismatch", "quality:harmonic_low_chord_tone_ratio_hard",
}
REVIEW_QUALITY_FLAGS = {
    "quality:mean_velocity_extreme", "quality:mean_duration_too_long", "quality:very_large_pitch_span",
    "quality:very_low_unique_pitch_count", "quality:harmonic_low_chord_tone_ratio_review",
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def get_sample_paths(split_dir: Path) -> List[Tuple[str, Path, Path]]:
    meta_dir, midi_dir = split_dir / "meta", split_dir / "midi"
    if not meta_dir.exists():
        return []
    return [(p.stem, p, midi_dir / f"{p.stem}.mid") for p in sorted(meta_dir.glob("*.json"))]


def check_json_basic(sample: Dict[str, Any]) -> List[str]:
    issues = [f"missing_top:{k}" for k in ["basic_info", "control_attributes", "derived_attributes", "events"] if k not in sample]
    if issues:
        return issues

    basic, events = sample["basic_info"], sample["events"]
    issues += [f"missing_basic:{k}" for k in [
        "num_bars", "num_notes", "time_signature", "tempo_bpm", "tempo_class", "bar_duration_sec",
        "clip_duration_sec", "leading_silence_sec", "music_start_sec", "music_end_sec",
    ] if k not in basic]

    if basic.get("num_bars") != 2:
        issues.append("basic:num_bars_not_2")
    if "num_notes" in basic and basic["num_notes"] != len(events):
        issues.append("basic:num_notes_mismatch")
    if len(events) == 0:
        issues.append("events:empty")
    return issues


def check_time_consistency(sample: Dict[str, Any]) -> List[str]:
    issues = []
    basic = sample["basic_info"]
    leading, music_start, music_end = float(basic["leading_silence_sec"]), float(basic["music_start_sec"]), float(basic["music_end_sec"])
    clip, bar_duration = float(basic["clip_duration_sec"]), float(basic["bar_duration_sec"])

    if abs(music_start - leading) > 1e-4:
        issues.append("basic:music_start_mismatch")
    if abs(music_end - clip) > 1e-4:
        issues.append("basic:music_end_mismatch")
    if abs((leading + 2.0 * bar_duration) - clip) > 1e-3:
        issues.append("basic:clip_duration_inconsistent")
    if leading < -EPS:
        issues.append("basic:negative_leading_silence")
    if clip <= 0:
        issues.append("basic:non_positive_clip_duration")
    return issues


def check_events(sample: Dict[str, Any]) -> List[str]:
    clip_duration_sec = float(sample["basic_info"]["clip_duration_sec"])
    prev_on, prev_off = -1e18, -1e18

    for i, e in enumerate(sample["events"]):
        for key in ["pitch", "onset_sec", "offset_sec", "velocity", "bar_index"]:
            if key not in e:
                return [f"events:missing_key:{key}"]

        pitch, onset, offset = int(e["pitch"]), float(e["onset_sec"]), float(e["offset_sec"])
        velocity, bar_index = int(e["velocity"]), int(e["bar_index"])
        issues = []
        if not (0 <= pitch <= 127):
            issues.append("events:pitch_out_of_midi_range")
        if not (1 <= velocity <= 127):
            issues.append("events:velocity_out_of_range")
        if offset <= onset + EPS:
            issues.append("events:non_positive_duration")
        if onset < -EPS:
            issues.append("events:negative_onset")
        if offset > clip_duration_sec + 1e-4:
            issues.append("events:offset_after_clip_end")
        if bar_index not in {1, 2}:
            issues.append("events:bar_index_invalid")
        if onset < prev_on - EPS:
            issues.append("events:not_sorted_by_onset")
        if onset < prev_off - EPS:
            issues.append("events:polyphonic_overlap")
        if i > 0 and onset - prev_off > 2.0:
            issues.append("events:very_large_gap_sec")
        if issues:
            return issues
        prev_on, prev_off = onset, offset
    return []


def read_midi_notes(midi_path: Path) -> Tuple[List[pretty_midi.Note], List[str]]:
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        return [], [f"midi:load_failed:{type(e).__name__}"]

    if len(pm.instruments) == 0:
        return [], ["midi:no_instruments"]

    notes = sorted((n for inst in pm.instruments for n in inst.notes), key=lambda n: (n.start, n.end, n.pitch))
    if not notes:
        return [], ["midi:no_notes"]

    prev_end = -1e18
    for n in notes:
        if n.end <= n.start + EPS:
            return notes, ["midi:non_positive_duration"]
        if n.start < prev_end - EPS:
            return notes, ["midi:polyphonic_overlap"]
        prev_end = n.end
    return notes, []


def compare_json_midi(sample: Dict[str, Any], midi_notes: List[pretty_midi.Note]) -> List[str]:
    events = sample["events"]
    if len(events) != len(midi_notes):
        return ["pair:note_count_mismatch"]

    issues = []
    if any(int(e["pitch"]) != int(n.pitch) for e, n in zip(events, midi_notes)):
        issues.append("pair:pitch_mismatch")
    if max((abs(float(e["onset_sec"]) - float(n.start)) for e, n in zip(events, midi_notes)), default=0.0) > 2e-3:
        issues.append("pair:onset_mismatch")
    if max((abs(float(e["offset_sec"]) - float(n.end)) for e, n in zip(events, midi_notes)), default=0.0) > 2e-3:
        issues.append("pair:offset_mismatch")
    return issues


def heuristic_quality_flags(sample: Dict[str, Any]) -> List[str]:
    """The actual quality thresholds behind soft/review classification."""
    flags = []
    control, derived, events, basic = sample["control_attributes"], sample["derived_attributes"], sample["events"], sample["basic_info"]
    num_notes = int(basic["num_notes"])
    pitch_span = int(derived.get("pitch_span", 0))
    unique_pitch_count = int(derived.get("unique_pitch_count", 0))
    mean_velocity = float(derived.get("mean_velocity", 0.0))
    mean_duration_beats = float(derived.get("mean_duration_beats", 0.0))

    if num_notes < 5:
        flags.append("quality:too_few_notes")
    if pitch_span == 0:
        flags.append("quality:zero_pitch_span")
    if unique_pitch_count <= 1:
        flags.append("quality:single_unique_pitch")
    elif unique_pitch_count <= 2:
        flags.append("quality:very_low_unique_pitch_count")
    if pitch_span >= 24:
        flags.append("quality:very_large_pitch_span")
    if mean_velocity < 45 or mean_velocity > 115:
        flags.append("quality:mean_velocity_extreme")
    if mean_duration_beats > 2.2:
        flags.append("quality:mean_duration_too_long")

    if control.get("structure_mode") == "explicit_harmonic_pattern" and control.get("harmonic_pattern_type", "none") != "none":
        tonal_mode, tonic_pc = control["tonal_mode"], int(control["tonic_pc"])
        chord_hits = 0
        for e in events:
            chord_type = control["bar1_chord_type"] if int(e["bar_index"]) == 1 else control["bar2_chord_type"]
            if int(e["pitch"]) % 12 in set(chord_pitch_classes(tonal_mode, tonic_pc, chord_type)):
                chord_hits += 1
        ratio = chord_hits / max(1, len(events))
        if ratio < 0.65:
            flags.append("quality:harmonic_low_chord_tone_ratio_hard")
        elif ratio < 0.85:
            flags.append("quality:harmonic_low_chord_tone_ratio_review")

    if control.get("repetition_type") == "exact_bar_repeat":
        b1 = sum(1 for e in events if int(e["bar_index"]) == 1)
        b2 = sum(1 for e in events if int(e["bar_index"]) == 2)
        if b1 != b2:
            flags.append("quality:exact_repeat_bar_note_count_mismatch")

    return flags


def classify_sample(issues: List[str], flags: List[str]) -> str:
    if any(x.startswith(HARD_PREFIXES) for x in issues):
        return "hard"
    if any(f in SOFT_QUALITY_FLAGS for f in flags):
        return "soft"
    if any(f in REVIEW_QUALITY_FLAGS for f in flags):
        return "review"
    return "clean"


def inspect_dataset(dataset_root: Path):
    lists: Dict[str, List[Dict[str, Any]]] = {"hard": [], "soft": [], "review": []}
    counters: Dict[str, Counter] = {"hard": Counter(), "soft": Counter(), "review": Counter()}
    total_samples = 0

    for split in ["train", "val", "test"]:
        for sample_id, json_path, midi_path in get_sample_paths(dataset_root / split):
            total_samples += 1
            issues: List[str] = []
            flags: List[str] = []

            if not midi_path.exists():
                issues = ["pair:midi_missing"]
            else:
                try:
                    sample = load_json(json_path)
                except Exception as e:
                    sample = None
                    issues = [f"json:load_failed:{type(e).__name__}"]

                if sample is not None:
                    issues = check_json_basic(sample)
                    if not issues:
                        issues = check_time_consistency(sample) + check_events(sample)

                    midi_notes, midi_issues = read_midi_notes(midi_path)
                    issues += midi_issues
                    if not issues:
                        issues += compare_json_midi(sample, midi_notes)
                    if not issues:
                        flags = heuristic_quality_flags(sample)

            cls = classify_sample(issues, flags)
            if cls == "clean":
                continue

            item = {"sample_id": sample_id, "split": split, "issues": issues, "flags": flags}
            lists[cls].append(item)
            counters[cls].update(issues)
            counters[cls].update(flags)

    return lists, counters, total_samples


def write_list_files(output_root: Path, stem: str, items: List[Dict[str, Any]]) -> None:
    with open(output_root / f"{stem}.txt", "w", encoding="utf8") as f:
        f.write("\n".join(item["sample_id"] for item in items) + ("\n" if items else ""))
    with open(output_root / f"{stem}.json", "w", encoding="utf8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = dataset_root / "inspection_results"
    output_root.mkdir(exist_ok=True)

    lists, counters, total = inspect_dataset(dataset_root)
    for cls in ["hard", "soft", "review"]:
        write_list_files(output_root, f"{cls}_blacklist" if cls != "review" else "review_list", lists[cls])

    counts = {cls: len(lists[cls]) for cls in ["hard", "soft", "review"]}
    with open(output_root / "summary.txt", "w", encoding="utf8") as f:
        f.write(f"total_samples: {total}\n")
        for cls, n in counts.items():
            f.write(f"{cls}: {n} ({100.0 * n / max(1, total):.2f}%)\n")

    print("=== Inspection Summary ===")
    print(f"total_samples: {total}")
    for cls, n in counts.items():
        print(f"{cls}: {n} ({100.0 * n / max(1, total):.2f}%)")
        for reason, count in counters[cls].most_common(15):
            print(f"  {reason}: {count}")
    print(f"\n[DONE] wrote {output_root}/{{hard_blacklist,soft_blacklist,review_list}}.{{txt,json}}")


if __name__ == "__main__":
    main()
