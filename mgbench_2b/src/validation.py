"""Post-processing and acceptance-checking for one generated sample: timing
humanization, the accept/reject rule set (validate_sample), and the MIDI /
JSON writers.

generate_valid_sample() is the retry loop 01_generate_midi.py calls once per
target sample: it keeps calling generate_one_sample_fn(rng) and validating
the result until one passes (or SANITY_MAX_RETRIES_PER_SAMPLE is hit), and
tallies every rejection reason and every accepted sample's control/derived
values into a SanityStats, which becomes two_bar_dataset/sanity_report.txt.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pretty_midi

from src.config import (
    BAR_DRIFT_MAX_BY_LEVEL,
    EPS,
    GLOBAL_DRIFT_MAX_BY_LEVEL,
    MAX_CLIP_DURATION_SEC,
    MIN_NOTE_DUR_SEC,
    ONSET_JITTER_MAX_BY_LEVEL,
    OFFSET_JITTER_MAX_BY_LEVEL,
    PITCH_MAX,
    PITCH_MIN,
    PROGRAM,
    SANITY_MAX_RETRIES_PER_SAMPLE,
    STRONG_BEATS,
    STRUCTURED_HUMANIZATION_SCALE,
    TEMPO_TO_CLASS,
    TIMING_HUMANIZATION_LEVELS,
    TIMING_HUMANIZATION_WEIGHTS,
    VELOCITY_JITTER_MAX_BY_LEVEL,
    VELOCITY_MAX,
    VELOCITY_MIN,
    SanityStats,
    ValidationResult,
)
from src.music_theory import chord_pitch_classes, interval_class, interval_direction, mean, scale_pitch_classes


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def refresh_event_relations(events: List[Dict]) -> List[Dict]:
    """Recompute interval_from_prev / interval_class_from_prev / direction_from_prev
    after sanitize_events may have dropped or reordered notes."""
    for i, e in enumerate(events):
        if i == 0:
            e["interval_from_prev"] = e["interval_class_from_prev"] = e["direction_from_prev"] = None
        else:
            iv = e["pitch"] - events[i - 1]["pitch"]
            e["interval_from_prev"] = int(iv)
            e["interval_class_from_prev"] = interval_class(iv)
            e["direction_from_prev"] = interval_direction(iv)
    return events


def ensure_monophonic(events: List[Dict]) -> None:
    ev = sorted(events, key=lambda e: (e["onset_sec"], e["offset_sec"], e.get("note_index", 0)))
    prev_off = -1e18
    for e in ev:
        if e["offset_sec"] <= e["onset_sec"]:
            raise ValueError("Non-positive duration detected in generated events.")
        if e["onset_sec"] < prev_off - EPS:
            raise ValueError("Overlap detected in generated events.")
        prev_off = e["offset_sec"]


def sanitize_events(events: List[Dict], clip_duration_sec: float) -> List[Dict]:
    """Clip onsets/offsets to [0, clip_duration_sec], enforce a minimum note
    duration and non-overlap by nudging onsets forward past the previous
    note's offset, then drop any note that no longer fits. Run after timing
    jitter, since jitter is what can push a note out of bounds or into its
    neighbor."""
    if not events:
        return []

    src = sorted(events, key=lambda x: (x["onset_sec"], x["offset_sec"], x.get("note_index", 0)))
    out: List[Dict] = []
    prev_off = 0.0
    for e in src:
        onset = max(max(0.0, float(e["onset_sec"])), prev_off)
        if onset + MIN_NOTE_DUR_SEC > clip_duration_sec:
            continue

        offset = max(min(float(e["offset_sec"]), clip_duration_sec), onset + MIN_NOTE_DUR_SEC)
        offset = min(offset, clip_duration_sec)
        if offset <= onset:
            continue

        new_e = dict(e)
        new_e["onset_sec"] = round(onset, 6)
        new_e["offset_sec"] = round(offset, 6)
        out.append(new_e)
        prev_off = new_e["offset_sec"]

    for i, e in enumerate(out):
        e["note_index"] = i

    ensure_monophonic(out)
    return refresh_event_relations(out)


def sample_humanization_level(rng) -> str:
    return rng.choices(TIMING_HUMANIZATION_LEVELS, weights=TIMING_HUMANIZATION_WEIGHTS, k=1)[0]


def humanization_scale_for_control(control: Dict[str, Any]) -> float:
    if control.get("structure_mode") in {"explicit_meter", "explicit_repetition", "explicit_harmonic_pattern"}:
        return STRUCTURED_HUMANIZATION_SCALE
    return 1.0


def apply_timing_jitter(events: List[Dict], rng, control: Dict[str, Any]) -> List[Dict]:
    """Nudge every onset/offset by three independent amounts - a global clip-wide
    drift, a per-bar drift, and per-note jitter - plus a small per-note velocity
    jitter. All four draw from the same `level` (light/medium/strong), scaled
    down for structured excerpts so the labeled structure stays audible."""
    if not events:
        return []

    level = sample_humanization_level(rng)
    scale = humanization_scale_for_control(control)

    onset_max = ONSET_JITTER_MAX_BY_LEVEL[level] * scale
    offset_max = OFFSET_JITTER_MAX_BY_LEVEL[level] * scale
    bar_drift_max = BAR_DRIFT_MAX_BY_LEVEL[level] * scale
    global_drift_max = GLOBAL_DRIFT_MAX_BY_LEVEL[level] * scale
    velocity_jitter_max = max(0, int(round(VELOCITY_JITTER_MAX_BY_LEVEL[level] * scale)))

    global_drift = rng.uniform(-global_drift_max, global_drift_max)
    bar_drifts = {1: rng.uniform(-bar_drift_max, bar_drift_max), 2: rng.uniform(-bar_drift_max, bar_drift_max)}

    out: List[Dict] = []
    for e in events:
        new_e = dict(e)
        bar_drift = bar_drifts.get(int(new_e["bar_index"]), 0.0)
        new_e["onset_sec"] = float(new_e["onset_sec"]) + global_drift + bar_drift + rng.uniform(-onset_max, onset_max)
        new_e["offset_sec"] = float(new_e["offset_sec"]) + global_drift + bar_drift + rng.uniform(-offset_max, offset_max)
        if velocity_jitter_max > 0:
            v = int(new_e["velocity"]) + rng.randint(-velocity_jitter_max, velocity_jitter_max)
            new_e["velocity"] = max(VELOCITY_MIN, min(VELOCITY_MAX, v))
        out.append(new_e)

    return out


def detect_motif_repetition(events: List[Dict]) -> bool:
    """True if any (interval, duration) pair recurs - a cheap proxy for
    "this excerpt has a repeated melodic-rhythmic cell", used to sanity-check
    that repeat_with_variation / exact_bar_repeat samples actually sound repetitive."""
    if len(events) < 4:
        return False
    seen = set()
    for i in range(1, len(events)):
        pair = (events[i]["pitch"] - events[i - 1]["pitch"], round(events[i - 1]["duration_beats"], 3))
        if pair in seen:
            return True
        seen.add(pair)
    return False


def derive_attributes(events: List[Dict]) -> Dict:
    pitches = [e["pitch"] for e in events]
    durs = [e["duration_beats"] for e in events]
    intervals = [e["interval_from_prev"] for e in events[1:]]
    directions = [interval_direction(iv) for iv in intervals if iv is not None and iv != 0]
    direction_changes = sum(1 for a, b in zip(directions, directions[1:]) if a != b)

    return {
        "bar1_note_count": sum(1 for e in events if e["bar_index"] == 1),
        "bar2_note_count": sum(1 for e in events if e["bar_index"] == 2),
        "min_pitch": int(min(pitches)) if pitches else 0,
        "max_pitch": int(max(pitches)) if pitches else 0,
        "pitch_span": int(max(pitches) - min(pitches)) if pitches else 0,
        "unique_pitch_count": len(set(pitches)),
        "has_repeated_pitch": len(set(pitches)) < len(pitches),
        "mean_duration_beats": round(mean(durs), 6),
        "step_count": sum(1 for iv in intervals if iv is not None and abs(iv) in {1, 2}),
        "skip_count": sum(1 for iv in intervals if iv is not None and abs(iv) in {3, 4, 5}),
        "leap_count": sum(1 for iv in intervals if iv is not None and abs(iv) >= 6),
        "direction_changes": int(direction_changes),
        "mean_velocity": round(mean([e["velocity"] for e in events]), 6),
        "has_motif_repetition": bool(detect_motif_repetition(events)),
    }


def recompute_core_from_events(entry: Dict[str, Any]) -> Dict[str, Any]:
    recomputed = derive_attributes(entry["events"])
    recomputed["num_notes"] = len(entry["events"])
    recomputed["clip_duration_sec"] = entry["basic_info"]["clip_duration_sec"]
    recomputed["bar_duration_sec"] = entry["basic_info"]["bar_duration_sec"]
    return recomputed


# Fields validate_sample cross-checks between the entry's own derived_attributes
# and a fresh recomputation from events - catches any generator/jitter bug that
# would otherwise leave stale derived values sitting next to correct events.
_DERIVED_FIELDS_TO_RECHECK = [
    "bar1_note_count", "bar2_note_count", "min_pitch", "max_pitch", "pitch_span",
    "unique_pitch_count", "has_repeated_pitch", "mean_duration_beats", "step_count",
    "skip_count", "leap_count", "direction_changes", "mean_velocity", "has_motif_repetition",
]


def validate_sample(entry: Dict[str, Any]) -> ValidationResult:
    """The accept/reject gate every generated sample must pass before it's
    written to disk. `errors` are hard rejections (structural problems: bad
    timing, out-of-scale pitches, non-monophonic overlap, too few notes);
    `warnings` flag samples that are kept but flagged as borderline (e.g. a
    "conjunct" sample with several leaps anyway) - these don't reject the
    sample, they just get tallied into the sanity report."""
    errors: List[str] = []
    warnings: List[str] = []

    basic, control, derived, events = entry["basic_info"], entry["control_attributes"], entry["derived_attributes"], entry["events"]

    if basic["num_bars"] != 2:
        errors.append("basic:num_bars_not_2")
    if basic["num_notes"] != len(events):
        errors.append("basic:num_notes_mismatch")
    if not events:
        errors.append("events:empty")
    if TEMPO_TO_CLASS.get(basic["tempo_bpm"]) != basic["tempo_class"]:
        errors.append("basic:tempo_class_mismatch")
    if abs(basic["clip_duration_sec"] - (basic["leading_silence_sec"] + 2 * basic["bar_duration_sec"])) > 1e-6:
        errors.append("basic:clip_bar_duration_inconsistent")
    if basic["clip_duration_sec"] > MAX_CLIP_DURATION_SEC + 1e-6:
        errors.append("basic:clip_duration_too_long")

    prev_on, prev_off = -1e18, -1e18
    allowed_scale_pcs = set(scale_pitch_classes(control["tonal_mode"], control["tonic_pc"]))
    large_gaps, max_gap_beats = 0, 0.0

    for i, e in enumerate(events):
        if e["pitch"] < PITCH_MIN or e["pitch"] > PITCH_MAX:
            errors.append("events:pitch_out_of_range")
            break
        if e["offset_sec"] <= e["onset_sec"]:
            errors.append("events:non_positive_duration")
            break
        if e["onset_sec"] < -EPS or e["offset_sec"] > basic["clip_duration_sec"] + 1e-6:
            errors.append("events:time_out_of_clip")
            break
        if e["onset_sec"] < prev_on - EPS:
            errors.append("events:onset_not_sorted")
            break
        if e["onset_sec"] < prev_off - EPS:
            errors.append("events:polyphonic_overlap")
            break
        if control["interval_constraint_mode"] != "free_chromatic" and e["pitch"] % 12 not in allowed_scale_pcs:
            errors.append("events:scale_violation")
            break

        if i > 0:
            gap_beats = (e["onset_sec"] - events[i - 1]["offset_sec"]) / (basic["bar_duration_sec"] / basic["beats_per_bar"])
            max_gap_beats = max(max_gap_beats, gap_beats)
            if gap_beats > 1.5 + 1e-9:
                errors.append("music:gap_too_large")
                break
            if gap_beats > 0.75 + 1e-9:
                large_gaps += 1

        prev_on, prev_off = e["onset_sec"], e["offset_sec"]

    if len(events) < 5:
        errors.append("music:too_few_notes")

    if not errors:
        recomputed = recompute_core_from_events(entry)
        for key in _DERIVED_FIELDS_TO_RECHECK:
            lhs, rhs = derived[key], recomputed[key]
            mismatch = abs(lhs - rhs) > 1e-6 if isinstance(lhs, float) else lhs != rhs
            if mismatch:
                errors.append(f"derived:{key}_mismatch")
                break

    density_to_range = {
        "sparse": (5, 6), "medium": (7, 9),
        "dense": (9, 11) if basic["time_signature"] == "6/8" else (10, 12),
    }
    lo, hi = density_to_range[control["density_level"]]
    if not (lo <= len(events) <= hi):
        warnings.append("control:density_realization_off")
    if control["interval_profile"] == "conjunct" and derived["leap_count"] > max(1, len(events) // 4):
        warnings.append("control:conjunct_too_many_leaps")
    if control["interval_profile"] == "disjunct" and derived["leap_count"] == 0:
        warnings.append("control:disjunct_no_leaps")
    if control["bar_relation_type"] == "repeat_with_variation" and not derived["has_motif_repetition"]:
        warnings.append("control:repeat_with_variation_no_motif_repetition")
    if control["repetition_type"] == "exact_bar_repeat" and not derived["has_motif_repetition"]:
        warnings.append("control:exact_repetition_not_detected")
    if control["repetition_type"] == "transposed_bar_repeat" and not derived["has_motif_repetition"]:
        warnings.append("control:transposed_repetition_not_detected")

    if control["structure_mode"] == "explicit_harmonic_pattern":
        chord_tone_hits = 0
        for e in events:
            chord_type = control["bar1_chord_type"] if e["bar_index"] == 1 else control["bar2_chord_type"]
            if e["pitch"] % 12 in set(chord_pitch_classes(control["tonal_mode"], control["tonic_pc"], chord_type)):
                chord_tone_hits += 1
        if events and chord_tone_hits / len(events) < 0.90:
            warnings.append("control:harmonic_pattern_low_chord_tone_ratio")

    if large_gaps >= 2:
        warnings.append("music:multiple_large_gaps")
    if max_gap_beats > 1.0 + 1e-9:
        warnings.append("music:max_gap_gt_1_beat")

    return ValidationResult(ok=(len(errors) == 0), errors=errors, warnings=warnings)


def events_to_pretty_midi(events: List[Dict], program: int = PROGRAM) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, is_drum=False)
    for e in events:
        inst.notes.append(pretty_midi.Note(
            velocity=int(e["velocity"]), pitch=int(e["pitch"]),
            start=float(e["onset_sec"]), end=float(e["offset_sec"]),
        ))
    pm.instruments.append(inst)
    return pm


def write_split(entries: List[Dict[str, Any]], outdir: Path, split: str) -> None:
    midi_dir, meta_dir = outdir / "midi", outdir / "meta"
    safe_mkdir(midi_dir)
    safe_mkdir(meta_dir)

    for idx, entry in enumerate(entries, start=1):
        sample_id = f"{split}_{idx:06d}"
        entry["filename"], entry["split"], entry["sample_id"] = f"{sample_id}.mid", split, sample_id

        events_to_pretty_midi(entry["events"], program=PROGRAM).write(str(midi_dir / entry["filename"]))
        with open(meta_dir / f"{sample_id}.json", "w", encoding="utf8") as f:
            json.dump(entry, f, indent=2)


def generate_valid_sample(rng, sanity: SanityStats, generate_one_sample_fn):
    """Retry loop: generate + validate until one sample passes, tallying
    every attempt into `sanity`. Never resets/reseeds `rng` between retries,
    so results stay deterministic for a given top-level seed."""
    last_errors: List[str] = []
    for _ in range(SANITY_MAX_RETRIES_PER_SAMPLE):
        sanity.attempts += 1
        entry = generate_one_sample_fn(rng)
        result = validate_sample(entry)

        if result.ok:
            sanity.accepted += 1
            for w in result.warnings:
                sanity.warning_reasons[w] += 1

            basic, control, derived = entry["basic_info"], entry["control_attributes"], entry["derived_attributes"]
            for field in ["time_signature", "tempo_class"]:
                sanity.accepted_control[field][basic[field]] += 1
            for field in ["density_level", "tonal_mode", "interval_profile", "bar_relation_type",
                          "structure_mode", "meter_clarity", "repetition_type", "accent_pattern_type",
                          "harmonic_pattern_type"]:
                sanity.accepted_control[field][control[field]] += 1
            for k in ["num_notes", "bar_duration_sec", "clip_duration_sec"]:
                sanity.accepted_derived_values[k].append(float(basic[k]))
            for k in ["bar1_note_count", "bar2_note_count", "pitch_span", "unique_pitch_count",
                      "mean_duration_beats", "step_count", "skip_count", "leap_count",
                      "direction_changes", "mean_velocity"]:
                sanity.accepted_derived_values[k].append(float(derived[k]))

            return entry

        sanity.rejected += 1
        for e in result.errors:
            sanity.reject_reasons[e] += 1
        last_errors = result.errors

    raise RuntimeError(f"Failed to generate a valid sample after {SANITY_MAX_RETRIES_PER_SAMPLE} retries. Last errors: {last_errors}")


def validate_dataset_distribution(sanity: SanityStats) -> List[str]:
    """Flag any control field where one value ended up dominating >65% of
    accepted samples - a signal the sampling weights or a filter interacted
    unexpectedly, worth a human look before treating the dataset as final."""
    warnings: List[str] = []
    total = max(1, sanity.accepted)
    for field in ["time_signature", "tempo_class", "density_level", "tonal_mode", "interval_profile", "bar_relation_type"]:
        counter = sanity.accepted_control[field]
        if counter and max(counter.values()) / total > 0.65:
            warnings.append(f"dataset:{field}_dominant_class_gt_65pct")
    return warnings


def write_sanity_report(outdir: Path, sanity: SanityStats, dataset_warnings: List[str]) -> None:
    lines: List[str] = ["=== Dataset Sanity Report ===",
                        f"attempts: {sanity.attempts}", f"accepted: {sanity.accepted}", f"rejected: {sanity.rejected}",
                        f"reject_rate_pct: {100.0 * sanity.rejected / max(1, sanity.attempts):.2f}"]

    lines.append("\n=== Top reject reasons ===")
    lines += [f"{k}: {v}" for k, v in sanity.reject_reasons.most_common(30)]
    lines.append("\n=== Top warning reasons ===")
    lines += [f"{k}: {v}" for k, v in sanity.warning_reasons.most_common(30)]

    lines.append("\n=== Accepted control distributions ===")
    for field, counter in sanity.accepted_control.items():
        lines.append(f"\n[{field}]")
        total = sum(counter.values())
        lines += [f"{k}: {v} ({100.0 * v / max(1, total):.2f}%)" for k, v in counter.most_common()]

    lines.append("\n=== Accepted numeric summaries ===")
    for field, values in sanity.accepted_derived_values.items():
        if not values:
            continue
        vals = sorted(values)
        lines.append(f"\n[{field}]")
        lines += [f"count: {len(vals)}", f"min: {vals[0]:.6f}", f"max: {vals[-1]:.6f}", f"mean: {mean(vals):.6f}"]

    lines.append("\n=== Dataset-level warnings ===")
    lines += dataset_warnings if dataset_warnings else ["none"]

    (outdir / "sanity_report.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
