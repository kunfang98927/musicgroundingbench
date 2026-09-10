"""Derives the `qa_summary` block: every note-index list, pitch sequence, and
interval sequence that the QA-fact builder (04_build_qa_facts.py) later turns
into grounding/understanding questions.

This is the one place this derivation is implemented. `03_build_slim_meta.py`
calls it to *produce* qa_summary; `04_check_meta_slim_consistency.py` calls it
again on the same input and diffs the result against what's on disk, so the
"check" step can never silently drift out of sync with the "build" step.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional

from src.music_theory import chord_pitch_classes

STRONG_BEATS = {"2/4": {1}, "3/4": {1}, "4/4": {1, 3}, "6/8": {1, 4}}

INTERVAL_NAME_BY_SEMITONES = {
    0: "unison", 1: "minor_second", 2: "major_second", 3: "minor_third", 4: "major_third",
    5: "perfect_fourth", 6: "tritone", 7: "perfect_fifth", 8: "minor_sixth",
    9: "major_sixth", 10: "minor_seventh", 11: "major_seventh", 12: "octave",
}


def normalize_within_bar_onset(x: float, tol: float = 1e-6) -> float:
    """Snap onsets that are within `tol` beats of an integer (e.g. 0.9999999
    from float accumulation) back onto that integer, so beat-index math below
    doesn't misclassify a downbeat as landing in the previous beat."""
    nearest = round(x)
    return float(nearest) if abs(x - nearest) < tol else x


def interval_name_from_semitones(iv: int) -> str:
    mag = abs(int(iv))
    return INTERVAL_NAME_BY_SEMITONES.get(mag, f"{mag}_semitones") if mag <= 12 else f"{mag}_semitones"


def directed_interval_name(iv: int) -> str:
    iv = int(iv)
    base = interval_name_from_semitones(iv)
    if iv > 0:
        return f"ascending_{base}"
    if iv < 0:
        return f"descending_{base}"
    return "repeated_unison"


def enrich_event_with_beat_fields(event: Dict[str, Any], time_signature: str, tol: float = 1e-6) -> Dict[str, Any]:
    """Add beat_index_in_bar / beat_phase / is_on_beat / is_offbeat / is_downbeat
    / is_strong_beat to one note event, recomputed from within_bar_onset_beats
    (rather than trusted from the generator) so meta_slim stays correct even if
    an event was hand-edited."""
    enriched = dict(event)
    onset_in_bar = normalize_within_bar_onset(float(enriched["within_bar_onset_beats"]), tol=tol)
    beat_index_in_bar = int(math.floor(onset_in_bar)) + 1
    beat_phase = onset_in_bar - math.floor(onset_in_bar)
    if abs(beat_phase) < tol:
        beat_phase = 0.0

    enriched["within_bar_onset_beats"] = round(onset_in_bar, 6)
    enriched["beat_index_in_bar"] = beat_index_in_bar
    enriched["beat_phase"] = round(beat_phase, 6)
    enriched["is_on_beat"] = abs(beat_phase) < tol
    enriched["is_offbeat"] = not enriched["is_on_beat"]
    enriched["is_downbeat"] = abs(onset_in_bar - 0.0) < tol
    enriched["is_strong_beat"] = beat_index_in_bar in STRONG_BEATS.get(time_signature, set())
    return enriched


def build_qa_summary(basic_info: Dict[str, Any], control_attributes: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute every field render_grounding_qa / render_understanding_qa read
    off a clip: note-index lists per bar/beat/extremum, pitch and interval
    sequences per bar, and chord-tone membership per bar."""
    music_start_sec = float(basic_info["music_start_sec"])
    music_end_sec = float(basic_info["music_end_sec"])
    bar_duration_sec = float(basic_info["bar_duration_sec"])
    beats_per_bar = int(basic_info["beats_per_bar"])

    tonal_mode = str(control_attributes["tonal_mode"])
    tonic_pc = int(control_attributes["tonic_pc"])
    bar1_chord_type = str(control_attributes["bar1_chord_type"])
    bar2_chord_type = str(control_attributes["bar2_chord_type"])
    structure_mode = str(control_attributes["structure_mode"])

    bar1_chord_pcs = chord_pitch_classes(tonal_mode, tonic_pc, bar1_chord_type)
    bar2_chord_pcs = chord_pitch_classes(tonal_mode, tonic_pc, bar2_chord_type)

    bar1_start_sec = round(music_start_sec, 6)
    bar1_end_sec = round(bar1_start_sec + bar_duration_sec, 6)
    bar2_start_sec = bar1_end_sec
    bar2_end_sec = round(music_end_sec, 6)

    if structure_mode == "explicit_harmonic_pattern":
        qa_focus_tags = ["harmony", "chord_tone", "harmonic_pattern", "pitch", "interval"]
    elif structure_mode == "explicit_repetition":
        qa_focus_tags = ["repetition", "pattern_relation", "pitch", "interval"]
    elif structure_mode == "explicit_meter":
        qa_focus_tags = ["meter", "beat", "downbeat", "rhythm"]
    else:
        qa_focus_tags = ["pitch", "interval", "melody", "bar_comparison"]

    bar1_events = [e for e in events if int(e["bar_index"]) == 1]
    bar2_events = [e for e in events if int(e["bar_index"]) == 2]
    pitches = [int(e["pitch"]) for e in events]
    max_pitch = max(pitches) if pitches else None
    min_pitch = min(pitches) if pitches else None

    bar1_pitches = [int(e["pitch"]) for e in bar1_events]
    bar2_pitches = [int(e["pitch"]) for e in bar2_events]

    beat_note_indices: Dict[str, List[int]] = {str(i): [] for i in range(1, beats_per_bar + 1)}
    bar1_note_indices, bar2_note_indices = [], []
    highest_note_indices, lowest_note_indices = [], []
    downbeat_note_indices, on_beat_note_indices, offbeat_note_indices = [], [], []
    step_note_indices, leap_note_indices = [], []

    interval_sequence, interval_class_sequence = [], []
    named_interval_sequence, directed_named_interval_sequence = [], []
    bar1_interval_sequence, bar2_interval_sequence = [], []
    bar1_named_interval_sequence, bar2_named_interval_sequence = [], []
    ascending_interval_count = descending_interval_count = repeated_pitch_count = 0
    interval_name_counts: Counter = Counter()
    ascending_interval_name_counts: Counter = Counter()
    descending_interval_name_counts: Counter = Counter()
    interval_name_to_note_indices: Dict[str, List[int]] = {}

    for e in events:
        idx, bar_index, beat_index, pitch = int(e["note_index"]), int(e["bar_index"]), int(e["beat_index_in_bar"]), int(e["pitch"])

        (bar1_note_indices if bar_index == 1 else bar2_note_indices).append(idx)
        if max_pitch is not None and pitch == max_pitch:
            highest_note_indices.append(idx)
        if min_pitch is not None and pitch == min_pitch:
            lowest_note_indices.append(idx)
        if bool(e["is_downbeat"]):
            downbeat_note_indices.append(idx)
        if bool(e["is_on_beat"]):
            on_beat_note_indices.append(idx)
        if bool(e["is_offbeat"]):
            offbeat_note_indices.append(idx)
        beat_note_indices.setdefault(str(beat_index), []).append(idx)

        iv = e.get("interval_from_prev")
        if iv is not None:
            iv = int(iv)
            name = interval_name_from_semitones(iv)
            directed_name = directed_interval_name(iv)

            interval_sequence.append(iv)
            interval_class_sequence.append(str(e.get("interval_class_from_prev") or "unknown"))
            named_interval_sequence.append(name)
            directed_named_interval_sequence.append(directed_name)
            (bar1_interval_sequence if bar_index == 1 else bar2_interval_sequence).append(iv)
            (bar1_named_interval_sequence if bar_index == 1 else bar2_named_interval_sequence).append(name)

            interval_name_counts[name] += 1
            interval_name_to_note_indices.setdefault(name, []).append(idx)
            interval_name_to_note_indices.setdefault(directed_name, []).append(idx)

            if iv > 0:
                ascending_interval_count += 1
                ascending_interval_name_counts[name] += 1
                ascending_interval_name_counts[directed_name] += 1
            elif iv < 0:
                descending_interval_count += 1
                descending_interval_name_counts[name] += 1
                descending_interval_name_counts[directed_name] += 1
            else:
                repeated_pitch_count += 1

            if abs(iv) in {1, 2}:
                step_note_indices.append(idx)
            if abs(iv) >= 6:
                leap_note_indices.append(idx)

    bar1_chord_pc_set, bar2_chord_pc_set = set(bar1_chord_pcs), set(bar2_chord_pcs)
    bar1_chord_tone_note_indices = [int(e["note_index"]) for e in bar1_events if int(e["pitch"]) % 12 in bar1_chord_pc_set]
    bar2_chord_tone_note_indices = [int(e["note_index"]) for e in bar2_events if int(e["pitch"]) % 12 in bar2_chord_pc_set]
    bar1_non_chord_tone_note_indices = [int(e["note_index"]) for e in bar1_events if int(e["pitch"]) % 12 not in bar1_chord_pc_set]
    bar2_non_chord_tone_note_indices = [int(e["note_index"]) for e in bar2_events if int(e["pitch"]) % 12 not in bar2_chord_pc_set]

    bar1_chord_tone_ratio: Optional[float] = round(len(bar1_chord_tone_note_indices) / len(bar1_events), 6) if bar1_events else None
    bar2_chord_tone_ratio: Optional[float] = round(len(bar2_chord_tone_note_indices) / len(bar2_events), 6) if bar2_events else None

    return {
        "qa_focus_tags": qa_focus_tags,
        "bar1_start_sec": bar1_start_sec, "bar1_end_sec": bar1_end_sec,
        "bar2_start_sec": bar2_start_sec, "bar2_end_sec": bar2_end_sec,
        "bar1_note_indices": bar1_note_indices, "bar2_note_indices": bar2_note_indices,
        "highest_note_indices": highest_note_indices, "lowest_note_indices": lowest_note_indices,
        "downbeat_note_indices": downbeat_note_indices, "on_beat_note_indices": on_beat_note_indices,
        "offbeat_note_indices": offbeat_note_indices, "beat_note_indices": beat_note_indices,
        "highest_pitch": max_pitch, "lowest_pitch": min_pitch,
        "bar1_highest_pitch": max(bar1_pitches) if bar1_pitches else None,
        "bar2_highest_pitch": max(bar2_pitches) if bar2_pitches else None,
        "bar1_lowest_pitch": min(bar1_pitches) if bar1_pitches else None,
        "bar2_lowest_pitch": min(bar2_pitches) if bar2_pitches else None,
        "bar1_pitch_sequence": bar1_pitches, "bar2_pitch_sequence": bar2_pitches,
        "bar1_duration_sequence": [float(e["duration_beats"]) for e in bar1_events],
        "bar2_duration_sequence": [float(e["duration_beats"]) for e in bar2_events],
        "pitch_class_set": sorted({p % 12 for p in pitches}),
        "chord_progression": [bar1_chord_type, bar2_chord_type],
        "harmonic_relation_type": "same_chord" if bar1_chord_type == bar2_chord_type else "changed_chord",
        "bar1_chord_pitch_classes": bar1_chord_pcs, "bar2_chord_pitch_classes": bar2_chord_pcs,
        "bar1_chord_tone_note_indices": bar1_chord_tone_note_indices,
        "bar2_chord_tone_note_indices": bar2_chord_tone_note_indices,
        "bar1_non_chord_tone_note_indices": bar1_non_chord_tone_note_indices,
        "bar2_non_chord_tone_note_indices": bar2_non_chord_tone_note_indices,
        "bar1_chord_tone_ratio": bar1_chord_tone_ratio, "bar2_chord_tone_ratio": bar2_chord_tone_ratio,
        "interval_sequence": interval_sequence, "interval_class_sequence": interval_class_sequence,
        "named_interval_sequence": named_interval_sequence,
        "directed_named_interval_sequence": directed_named_interval_sequence,
        "bar1_interval_sequence": bar1_interval_sequence, "bar2_interval_sequence": bar2_interval_sequence,
        "bar1_named_interval_sequence": bar1_named_interval_sequence,
        "bar2_named_interval_sequence": bar2_named_interval_sequence,
        "ascending_interval_count": ascending_interval_count,
        "descending_interval_count": descending_interval_count,
        "repeated_pitch_count": repeated_pitch_count,
        "interval_name_counts": dict(interval_name_counts),
        "ascending_interval_name_counts": dict(ascending_interval_name_counts),
        "descending_interval_name_counts": dict(descending_interval_name_counts),
        "interval_name_to_note_indices": interval_name_to_note_indices,
        "leap_note_indices": leap_note_indices, "step_note_indices": step_note_indices,
    }
