"""Step 7: turn each meta_slim sample into two catalogs of *facts* - one per
task, in `<split>/grounding_facts/` and `<split>/understanding_facts/` - that
09_render_grounding_qa.py / 10_render_understanding_qa.py later turn into
actual question/answer text.

A fact is a single answerable thing about the clip:
  - a grounding fact says WHERE (a list of note_indices -> time spans) - e.g.
    "downbeat_notes" or "interval_A4" (all notes reached by an ascending
    perfect fourth).
  - an understanding fact says WHAT (a value) plus WHERE the evidence for
    that value lives (evidence_spans) - e.g. "tonal_mode" = "major".

Every fact also carries a qa_category (ABS/REL/ORD/PAT, mirroring
MGBench-3N's query families) and a confidence ("high" or "low"). Facts that
depend on the excerpt having audibly salient harmony (chord type, chord
progression, chord-tone ratio, harmonic pattern) are only "high" confidence
when structure_mode == "explicit_harmonic_pattern" - the one mode built to
make harmony unambiguous to the ear. Everywhere else they're still generated
(the label is still true, since it's a control variable of the generator)
but filed under low_confidence_facts with a `reason`, so a downstream
renderer can choose to skip them, or include them but flag the caveat.

Usage:
    python scripts/07_build_qa_facts.py <dataset_root> [--overwrite]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import src.*` works regardless of cwd

GROUNDING_HARMONY_LOW_CONF_REASON = "Harmony is present in labels, but not necessarily strongly perceptible in the normal subset."
UNDERSTANDING_HARMONY_LOW_CONF_REASON = "Harmony is labeled, but the normal subset may not make the chord identity strongly salient by ear."
HARMONIC_PATTERN_LOW_CONF_REASON = "Harmonic pattern label is present, but this subset is not optimized for strong harmonic salience."


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_meta_slim_files(dataset_root: Path) -> List[Path]:
    files: List[Path] = []
    for split in ["train", "val", "test"]:
        d = dataset_root / split / "meta_slim"
        if d.exists():
            files.extend(sorted(d.glob("*.json")))
    return files


def infer_split(path: Path) -> str:
    for split in ["train", "val", "test"]:
        if split in path.parts:
            return split
    return "unknown"


def unique_sorted(indices: List[int]) -> List[int]:
    return sorted(set(int(i) for i in indices))


def spans_from_note_indices(events: List[Dict[str, Any]], note_indices: List[int]) -> List[List[float]]:
    index_set = set(int(i) for i in note_indices)
    return [[round(float(e["onset_sec"]), 6), round(float(e["offset_sec"]), 6)]
            for e in events if int(e["note_index"]) in index_set]


# =============================================================================
# qa_category: bucket each fact into the same ABS/REL/ORD/PAT families used by
# MGBench-3N, so both subsets can be analyzed on a common query taxonomy.
# =============================================================================

def grounding_category_from_fact_id(fact_id: str, concept: str, target: str) -> str:
    if fact_id in {"first_note", "last_note", "bar_1", "bar_2"} or fact_id.startswith("note_"):
        return "ORD"
    if fact_id.startswith("beat_") or fact_id in {"downbeat_notes", "on_beat_notes", "offbeat_notes"}:
        return "ORD"
    if concept == "pitch_extreme":
        return "ABS"
    if concept in {"interval_target", "named_interval_target", "chord_tone", "non_chord_tone"}:
        return "REL"
    if concept in {"repetition_pattern", "harmonic_pattern"} or "pattern" in target or "repeat" in target:
        return "PAT"
    return "ABS"


_UNDERSTANDING_ABS_FACT_IDS = {
    "time_signature", "tempo_bpm", "tempo_class", "tonal_mode", "tonic_pc", "num_notes",
    "bar1_chord_type", "bar2_chord_type", "pitch_span", "unique_pitch_count", "highest_pitch",
    "lowest_pitch", "mean_duration_beats", "step_count", "skip_count", "leap_count",
    "direction_changes", "mean_velocity", "ascending_interval_count", "descending_interval_count",
    "repeated_pitch_count",
}
_UNDERSTANDING_ORD_FACT_IDS = {"has_downbeat_notes", "has_offbeat_notes"}
_UNDERSTANDING_REL_FACT_IDS = {
    "which_bar_has_more_notes", "which_bar_has_higher_peak", "bar_note_count_equal",
    "bar_pitch_sequence_equal", "bar_duration_sequence_equal", "harmonic_relation_type",
    "bar1_chord_tone_ratio", "bar2_chord_tone_ratio",
}
_UNDERSTANDING_PAT_FACT_IDS = {
    "repetition_type", "harmonic_pattern_type", "chord_progression",
    "named_interval_sequence", "directed_named_interval_sequence", "interval_name_counts",
}


def understanding_category_from_fact_id(fact_id: str, concept: str) -> str:
    if fact_id in _UNDERSTANDING_ABS_FACT_IDS:
        return "ABS"
    if fact_id in _UNDERSTANDING_REL_FACT_IDS:
        return "REL"
    if fact_id in _UNDERSTANDING_ORD_FACT_IDS:
        return "ORD"
    if fact_id in _UNDERSTANDING_PAT_FACT_IDS:
        return "PAT"
    if fact_id.startswith("beat_") and fact_id.endswith("_note_count"):
        return "ORD"
    if fact_id.startswith("contains_interval_") or fact_id.startswith("count_interval_"):
        return "REL"
    if concept in {"bar_comparison", "bar_sequence_relation", "harmonic_relation", "chord_tone_ratio"}:
        return "REL"
    if concept in {"repetition", "harmonic_pattern", "interval_sequence", "chord_progression"}:
        return "PAT"
    if concept in {"downbeat", "beat_count", "beat_alignment"}:
        return "ORD"
    return "ABS"


# =============================================================================
# Evidence selection for understanding facts: which notes justify this answer?
# =============================================================================

def choose_understanding_evidence(sample: Dict[str, Any], fact_id: str) -> Tuple[List[int], List[List[float]]]:
    """Map a fact_id to the note span(s) that best justify its answer - e.g.
    "tonic_pc" points at the highest/lowest notes (register anchors), interval
    counts point at every stepwise/leaping note, bar-level facts point at
    that bar's notes. This enumerates every fact_id build_understanding_facts
    actually produces; anything not listed here falls back to every note in
    the clip (the "else" branch), which in practice never triggers - it's
    just a safety net for a fact added later without updating this function."""
    qa = sample["qa_summary"]
    structure_mode = str(sample["control_attributes"]["structure_mode"])

    all_notes = unique_sorted(qa["bar1_note_indices"] + qa["bar2_note_indices"])
    interval_notes = unique_sorted(qa["step_note_indices"] + qa["leap_note_indices"]) or all_notes
    downbeat_or_on_beat = unique_sorted(qa["downbeat_note_indices"] + qa["on_beat_note_indices"]) or all_notes
    highest_or_lowest = unique_sorted(qa["highest_note_indices"] + qa["lowest_note_indices"]) or all_notes
    both_bars_chord_tones = (
        unique_sorted(qa["bar1_chord_tone_note_indices"] + qa["bar2_chord_tone_note_indices"]) or all_notes
    ) if structure_mode == "explicit_harmonic_pattern" else all_notes

    def bar_chord_tones(bar_note_key: str, bar_chord_tone_key: str) -> List[int]:
        if structure_mode == "explicit_harmonic_pattern":
            return unique_sorted(qa[bar_chord_tone_key]) or unique_sorted(qa[bar_note_key])
        return unique_sorted(qa[bar_note_key])

    if fact_id in {"tempo_bpm", "tempo_class", "mean_duration_beats", "tonal_mode", "num_notes",
                    "bar1_note_count", "bar2_note_count", "unique_pitch_count", "repetition_type",
                    "harmonic_pattern_type", "bar_pitch_sequence_equal", "bar_duration_sequence_equal",
                    "bar_note_count_equal", "which_bar_has_more_notes", "which_bar_has_higher_peak"}:
        evidence_note_indices = all_notes
    elif fact_id.startswith("beat_") and fact_id.endswith("_note_count"):
        evidence_note_indices = all_notes
    elif fact_id == "time_signature":
        evidence_note_indices = downbeat_or_on_beat
    elif fact_id == "tonic_pc" or fact_id in {"pitch_span", "highest_pitch", "lowest_pitch"}:
        evidence_note_indices = highest_or_lowest
    elif fact_id in {"step_count", "skip_count", "leap_count", "direction_changes", "ascending_interval_count",
                       "descending_interval_count", "repeated_pitch_count", "named_interval_sequence",
                       "directed_named_interval_sequence", "interval_name_counts"} \
            or fact_id.startswith("contains_interval_") or fact_id.startswith("count_interval_"):
        evidence_note_indices = interval_notes
    elif fact_id in {"bar1_chord_type", "bar1_chord_tone_ratio"}:
        evidence_note_indices = bar_chord_tones("bar1_note_indices", "bar1_chord_tone_note_indices")
    elif fact_id in {"bar2_chord_type", "bar2_chord_tone_ratio"}:
        evidence_note_indices = bar_chord_tones("bar2_note_indices", "bar2_chord_tone_note_indices")
    elif fact_id in {"chord_progression", "harmonic_relation_type"}:
        evidence_note_indices = both_bars_chord_tones
    elif fact_id == "has_downbeat_notes":
        evidence_note_indices = downbeat_or_on_beat
    elif fact_id == "has_offbeat_notes":
        evidence_note_indices = unique_sorted(qa["offbeat_note_indices"] + all_notes)
    else:
        evidence_note_indices = all_notes

    return evidence_note_indices, spans_from_note_indices(sample["events"], evidence_note_indices)


# =============================================================================
# Fact list builders
# =============================================================================

def add_grounding_fact(
    target_list: List[Dict[str, Any]], *, fact_id: str, concept: str, target: str,
    note_indices: List[int], events: List[Dict[str, Any]], source_fields: List[str],
    confidence: str, reason: Optional[str] = None,
) -> None:
    if not note_indices:  # nothing to point at -> not a valid grounding question for this clip
        return
    target_list.append({
        "fact_id": fact_id, "qa_category": grounding_category_from_fact_id(fact_id, concept, target),
        "concept": concept, "target": target, "answer_type": "time_span_list",
        "note_indices": note_indices, "time_spans": spans_from_note_indices(events, note_indices),
        "confidence": confidence, "reason": reason, "source_fields": source_fields,
    })


def add_understanding_fact(
    target_list: List[Dict[str, Any]], *, sample: Dict[str, Any], fact_id: str, concept: str,
    answer_type: str, answer: Any, source_fields: List[str], confidence: str,
    reason: Optional[str] = None, extra: Optional[Dict[str, Any]] = None,
) -> None:
    evidence_note_indices, evidence_spans = choose_understanding_evidence(sample, fact_id)
    fact = {
        "fact_id": fact_id, "qa_category": understanding_category_from_fact_id(fact_id, concept),
        "concept": concept, "answer_type": answer_type, "answer": answer, "confidence": confidence,
        "reason": reason, "source_fields": source_fields, "evidence_type": "note_span_list",
        "evidence_note_indices": evidence_note_indices, "evidence_spans": evidence_spans,
    }
    if extra:
        fact.update(extra)
    target_list.append(fact)


def build_grounding_facts(sample: Dict[str, Any]) -> Dict[str, Any]:
    basic, control, events, qa = sample["basic_info"], sample["control_attributes"], sample["events"], sample["qa_summary"]
    structure_mode = str(control["structure_mode"])
    high: List[Dict[str, Any]] = []
    low: List[Dict[str, Any]] = []

    add_grounding_fact(high, fact_id="bar_1", concept="bar", target="bar_1", note_indices=qa["bar1_note_indices"], events=events,
                       source_fields=["qa_summary.bar1_note_indices", "qa_summary.bar1_start_sec", "qa_summary.bar1_end_sec"], confidence="high")
    add_grounding_fact(high, fact_id="bar_2", concept="bar", target="bar_2", note_indices=qa["bar2_note_indices"], events=events,
                       source_fields=["qa_summary.bar2_note_indices", "qa_summary.bar2_start_sec", "qa_summary.bar2_end_sec"], confidence="high")

    add_grounding_fact(high, fact_id="highest_note", concept="pitch_extreme", target="highest_note", note_indices=qa["highest_note_indices"],
                       events=events, source_fields=["qa_summary.highest_note_indices", "qa_summary.highest_pitch"], confidence="high")
    add_grounding_fact(high, fact_id="lowest_note", concept="pitch_extreme", target="lowest_note", note_indices=qa["lowest_note_indices"],
                       events=events, source_fields=["qa_summary.lowest_note_indices", "qa_summary.lowest_pitch"], confidence="high")

    if events:
        add_grounding_fact(high, fact_id="first_note", concept="note_order", target="first_note", note_indices=[int(events[0]["note_index"])],
                           events=events, source_fields=["events.note_index", "events.onset_sec", "events.offset_sec"], confidence="high")
        add_grounding_fact(high, fact_id="last_note", concept="note_order", target="last_note", note_indices=[int(events[-1]["note_index"])],
                           events=events, source_fields=["events.note_index", "events.onset_sec", "events.offset_sec"], confidence="high")

    for e in events:
        idx = int(e["note_index"])
        add_grounding_fact(high, fact_id=f"note_{idx}", concept="note_identity", target=f"note_{idx}", note_indices=[idx], events=events,
                           source_fields=["events.note_index", "events.onset_sec", "events.offset_sec", "events.pitch"], confidence="high")

    add_grounding_fact(high, fact_id="downbeat_notes", concept="downbeat", target="downbeat_notes", note_indices=qa["downbeat_note_indices"],
                       events=events, source_fields=["qa_summary.downbeat_note_indices", "events.is_downbeat"], confidence="high")
    add_grounding_fact(high, fact_id="on_beat_notes", concept="beat_alignment", target="on_beat_notes", note_indices=qa["on_beat_note_indices"],
                       events=events, source_fields=["qa_summary.on_beat_note_indices", "events.is_on_beat"], confidence="high")
    add_grounding_fact(high, fact_id="offbeat_notes", concept="beat_alignment", target="offbeat_notes", note_indices=qa["offbeat_note_indices"],
                       events=events, source_fields=["qa_summary.offbeat_note_indices", "events.is_offbeat"], confidence="high")

    for beat_key in sorted(qa["beat_note_indices"], key=int):
        add_grounding_fact(high, fact_id=f"beat_{beat_key}_notes", concept="beat", target=f"beat_{beat_key}_notes",
                           note_indices=qa["beat_note_indices"][beat_key], events=events,
                           source_fields=["qa_summary.beat_note_indices", "events.beat_index_in_bar"], confidence="high")

    add_grounding_fact(high, fact_id="step_target_notes", concept="interval_target", target="step_target_notes", note_indices=qa["step_note_indices"],
                       events=events, source_fields=["qa_summary.step_note_indices", "events.interval_from_prev"], confidence="high")
    add_grounding_fact(high, fact_id="leap_target_notes", concept="interval_target", target="leap_target_notes", note_indices=qa["leap_note_indices"],
                       events=events, source_fields=["qa_summary.leap_note_indices", "events.interval_from_prev"], confidence="high")

    for interval_name, note_indices in sorted(qa["interval_name_to_note_indices"].items()):
        add_grounding_fact(high, fact_id=f"interval_{interval_name}", concept="named_interval_target", target=interval_name,
                           note_indices=note_indices, events=events,
                           source_fields=["qa_summary.interval_name_to_note_indices", "events.interval_from_prev"], confidence="high")

    # Chord-tone facts are only "high" confidence in the mode built to make harmony audible.
    chord_target_list = high if structure_mode == "explicit_harmonic_pattern" else low
    chord_confidence = "high" if structure_mode == "explicit_harmonic_pattern" else "low"
    chord_reason = None if structure_mode == "explicit_harmonic_pattern" else GROUNDING_HARMONY_LOW_CONF_REASON
    for bar, concept, note_key, chord_field in [
        (1, "chord_tone", "bar1_chord_tone_note_indices", "bar1_chord_type"),
        (2, "chord_tone", "bar2_chord_tone_note_indices", "bar2_chord_type"),
    ]:
        add_grounding_fact(chord_target_list, fact_id=f"bar{bar}_chord_tone_notes", concept=concept, target=f"bar{bar}_chord_tone_notes",
                           note_indices=qa[note_key], events=events,
                           source_fields=[f"qa_summary.{note_key}", f"qa_summary.bar{bar}_chord_pitch_classes", f"control_attributes.{chord_field}"],
                           confidence=chord_confidence, reason=chord_reason)
    if structure_mode == "explicit_harmonic_pattern":
        for bar, note_key, chord_field in [(1, "bar1_non_chord_tone_note_indices", "bar1_chord_type"), (2, "bar2_non_chord_tone_note_indices", "bar2_chord_type")]:
            add_grounding_fact(high, fact_id=f"bar{bar}_non_chord_tone_notes", concept="non_chord_tone", target=f"bar{bar}_non_chord_tone_notes",
                               note_indices=qa[note_key], events=events,
                               source_fields=[f"qa_summary.{note_key}", f"qa_summary.bar{bar}_chord_pitch_classes", f"control_attributes.{chord_field}"],
                               confidence="high")

    all_pattern_notes = unique_sorted(qa["bar1_note_indices"] + qa["bar2_note_indices"])
    if structure_mode == "explicit_repetition":
        add_grounding_fact(high, fact_id="repetition_pattern_notes", concept="repetition_pattern", target="repetition_pattern_notes",
                           note_indices=all_pattern_notes, events=events,
                           source_fields=["control_attributes.repetition_type", "qa_summary.bar1_note_indices", "qa_summary.bar2_note_indices"], confidence="high")
        repetition_type = str(control["repetition_type"])
        if repetition_type == "exact_bar_repeat":
            add_grounding_fact(high, fact_id="exact_repeat_pattern_notes", concept="repetition_pattern", target="exact_repeat_pattern_notes",
                               note_indices=all_pattern_notes, events=events,
                               source_fields=["control_attributes.repetition_type", "qa_summary.bar1_pitch_sequence", "qa_summary.bar2_pitch_sequence",
                                              "qa_summary.bar1_duration_sequence", "qa_summary.bar2_duration_sequence"], confidence="high")
        if repetition_type == "transposed_bar_repeat":
            add_grounding_fact(high, fact_id="transposed_repeat_pattern_notes", concept="repetition_pattern", target="transposed_repeat_pattern_notes",
                               note_indices=all_pattern_notes, events=events,
                               source_fields=["control_attributes.repetition_type", "qa_summary.bar1_pitch_sequence", "qa_summary.bar2_pitch_sequence"], confidence="high")

    if structure_mode == "explicit_harmonic_pattern":
        add_grounding_fact(high, fact_id="harmonic_pattern_notes", concept="harmonic_pattern", target="harmonic_pattern_notes",
                           note_indices=all_pattern_notes, events=events,
                           source_fields=["control_attributes.harmonic_pattern_type", "qa_summary.bar1_chord_tone_note_indices", "qa_summary.bar2_chord_tone_note_indices"], confidence="high")
        harmonic_pattern_type = str(control["harmonic_pattern_type"])
        if harmonic_pattern_type == "ascending_arpeggio":
            add_grounding_fact(high, fact_id="ascending_arpeggio_pattern_notes", concept="harmonic_pattern", target="ascending_arpeggio_pattern_notes",
                               note_indices=all_pattern_notes, events=events,
                               source_fields=["control_attributes.harmonic_pattern_type", "qa_summary.bar1_pitch_sequence", "qa_summary.bar2_pitch_sequence"], confidence="high")
        if harmonic_pattern_type == "broken_chord_repeat":
            add_grounding_fact(high, fact_id="broken_chord_repeat_pattern_notes", concept="harmonic_pattern", target="broken_chord_repeat_pattern_notes",
                               note_indices=all_pattern_notes, events=events,
                               source_fields=["control_attributes.harmonic_pattern_type", "qa_summary.bar1_pitch_sequence", "qa_summary.bar2_pitch_sequence",
                                              "qa_summary.bar1_duration_sequence", "qa_summary.bar2_duration_sequence"], confidence="high")

    return {
        "sample_id": sample.get("sample_id", ""), "split": sample.get("split", ""),
        "high_confidence_facts": high, "low_confidence_facts": low,
        "meta_summary": {"structure_mode": structure_mode, "qa_focus_tags": qa["qa_focus_tags"],
                         "beats_per_bar": int(basic["beats_per_bar"]), "num_events": len(events)},
    }


# Simple (fact_id, concept, answer_type, source_field, answer) facts that need
# no special-casing beyond "look the value up and file it as understanding_fact".
def _direct_understanding_facts(sample: Dict[str, Any]) -> List[Tuple[str, str, str, str, Any]]:
    basic, control, derived, qa = sample["basic_info"], sample["control_attributes"], sample["derived_attributes"], sample["qa_summary"]
    return [
        ("time_signature", "meter", "class_label", "basic_info.time_signature", basic["time_signature"]),
        ("tempo_bpm", "tempo", "integer", "basic_info.tempo_bpm", basic["tempo_bpm"]),
        ("tempo_class", "tempo", "class_label", "basic_info.tempo_class", basic["tempo_class"]),
        ("tonal_mode", "mode", "class_label", "control_attributes.tonal_mode", control["tonal_mode"]),
        ("tonic_pc", "tonic", "integer", "control_attributes.tonic_pc", control["tonic_pc"]),
        ("num_notes", "count", "integer", "basic_info.num_notes", basic["num_notes"]),
        ("bar1_note_count", "bar_count", "number", "derived_attributes.bar1_note_count", derived["bar1_note_count"]),
        ("bar2_note_count", "bar_count", "number", "derived_attributes.bar2_note_count", derived["bar2_note_count"]),
        ("pitch_span", "pitch_summary", "number", "derived_attributes.pitch_span", derived["pitch_span"]),
        ("unique_pitch_count", "pitch_summary", "number", "derived_attributes.unique_pitch_count", derived["unique_pitch_count"]),
        ("mean_duration_beats", "rhythm_summary", "number", "derived_attributes.mean_duration_beats", derived["mean_duration_beats"]),
        ("step_count", "interval_summary", "number", "derived_attributes.step_count", derived["step_count"]),
        ("skip_count", "interval_summary", "number", "derived_attributes.skip_count", derived["skip_count"]),
        ("leap_count", "interval_summary", "number", "derived_attributes.leap_count", derived["leap_count"]),
        ("direction_changes", "interval_summary", "number", "derived_attributes.direction_changes", derived["direction_changes"]),
        ("mean_velocity", "dynamics_summary", "number", "derived_attributes.mean_velocity", derived["mean_velocity"]),
        ("highest_pitch", "pitch_extreme", "number", "qa_summary.highest_pitch", qa["highest_pitch"]),
        ("lowest_pitch", "pitch_extreme", "number", "qa_summary.lowest_pitch", qa["lowest_pitch"]),
        ("ascending_interval_count", "interval_summary", "number", "qa_summary.ascending_interval_count", qa["ascending_interval_count"]),
        ("descending_interval_count", "interval_summary", "number", "qa_summary.descending_interval_count", qa["descending_interval_count"]),
        ("repeated_pitch_count", "interval_summary", "number", "qa_summary.repeated_pitch_count", qa["repeated_pitch_count"]),
    ]


def build_understanding_facts(sample: Dict[str, Any]) -> Dict[str, Any]:
    control, qa = sample["control_attributes"], sample["qa_summary"]
    structure_mode = str(control["structure_mode"])
    high: List[Dict[str, Any]] = []
    low: List[Dict[str, Any]] = []

    for fact_id, concept, answer_type, source_field, answer in _direct_understanding_facts(sample):
        add_understanding_fact(high, sample=sample, fact_id=fact_id, concept=concept, answer_type=answer_type,
                               answer=answer, source_fields=[source_field], confidence="high")

    # Harmony facts: "high" confidence only for explicit_harmonic_pattern, same rule as the grounding side.
    chord_conf = "high" if structure_mode == "explicit_harmonic_pattern" else "low"
    chord_reason = None if structure_mode == "explicit_harmonic_pattern" else UNDERSTANDING_HARMONY_LOW_CONF_REASON
    chord_target = high if chord_conf == "high" else low
    for fact_id, concept, answer_type, source_field, answer in [
        ("bar1_chord_type", "chord", "class_label", "control_attributes.bar1_chord_type", control["bar1_chord_type"]),
        ("bar2_chord_type", "chord", "class_label", "control_attributes.bar2_chord_type", control["bar2_chord_type"]),
        ("chord_progression", "chord_progression", "sequence", "qa_summary.chord_progression", qa["chord_progression"]),
        ("harmonic_relation_type", "harmonic_relation", "class_label", "qa_summary.harmonic_relation_type", qa["harmonic_relation_type"]),
        ("bar1_chord_tone_ratio", "chord_tone_ratio", "number", "qa_summary.bar1_chord_tone_ratio", qa["bar1_chord_tone_ratio"]),
        ("bar2_chord_tone_ratio", "chord_tone_ratio", "number", "qa_summary.bar2_chord_tone_ratio", qa["bar2_chord_tone_ratio"]),
    ]:
        add_understanding_fact(chord_target, sample=sample, fact_id=fact_id, concept=concept, answer_type=answer_type,
                               answer=answer, source_fields=[source_field], confidence=chord_conf, reason=chord_reason)

    add_understanding_fact(high, sample=sample, fact_id="repetition_type", concept="repetition", answer_type="class_label",
                           answer=control["repetition_type"], source_fields=["control_attributes.repetition_type"], confidence="high")

    if structure_mode == "explicit_harmonic_pattern":
        add_understanding_fact(high, sample=sample, fact_id="harmonic_pattern_type", concept="harmonic_pattern", answer_type="class_label",
                               answer=control["harmonic_pattern_type"], source_fields=["control_attributes.harmonic_pattern_type"], confidence="high")
    elif control["harmonic_pattern_type"] != "none":
        add_understanding_fact(low, sample=sample, fact_id="harmonic_pattern_type", concept="harmonic_pattern", answer_type="class_label",
                               answer=control["harmonic_pattern_type"], source_fields=["control_attributes.harmonic_pattern_type"],
                               confidence="low", reason=HARMONIC_PATTERN_LOW_CONF_REASON)

    derived = sample["derived_attributes"]
    add_understanding_fact(high, sample=sample, fact_id="bar_pitch_sequence_equal", concept="bar_sequence_relation", answer_type="boolean",
                           answer=qa["bar1_pitch_sequence"] == qa["bar2_pitch_sequence"],
                           source_fields=["qa_summary.bar1_pitch_sequence", "qa_summary.bar2_pitch_sequence"], confidence="high")
    add_understanding_fact(high, sample=sample, fact_id="bar_duration_sequence_equal", concept="bar_sequence_relation", answer_type="boolean",
                           answer=qa["bar1_duration_sequence"] == qa["bar2_duration_sequence"],
                           source_fields=["qa_summary.bar1_duration_sequence", "qa_summary.bar2_duration_sequence"], confidence="high")
    add_understanding_fact(high, sample=sample, fact_id="bar_note_count_equal", concept="bar_comparison", answer_type="boolean",
                           answer=derived["bar1_note_count"] == derived["bar2_note_count"],
                           source_fields=["derived_attributes.bar1_note_count", "derived_attributes.bar2_note_count"], confidence="high")
    add_understanding_fact(high, sample=sample, fact_id="which_bar_has_more_notes", concept="bar_comparison", answer_type="class_label",
                           answer=("bar_1" if derived["bar1_note_count"] > derived["bar2_note_count"]
                                   else "bar_2" if derived["bar2_note_count"] > derived["bar1_note_count"] else "equal"),
                           source_fields=["derived_attributes.bar1_note_count", "derived_attributes.bar2_note_count"], confidence="high")
    add_understanding_fact(high, sample=sample, fact_id="which_bar_has_higher_peak", concept="bar_comparison", answer_type="class_label",
                           answer=("bar_1" if qa["bar1_highest_pitch"] > qa["bar2_highest_pitch"]
                                   else "bar_2" if qa["bar2_highest_pitch"] > qa["bar1_highest_pitch"] else "equal"),
                           source_fields=["qa_summary.bar1_highest_pitch", "qa_summary.bar2_highest_pitch"], confidence="high")

    add_understanding_fact(high, sample=sample, fact_id="has_downbeat_notes", concept="downbeat", answer_type="boolean",
                           answer=len(qa["downbeat_note_indices"]) > 0, source_fields=["qa_summary.downbeat_note_indices"], confidence="high")
    add_understanding_fact(high, sample=sample, fact_id="has_offbeat_notes", concept="beat_alignment", answer_type="boolean",
                           answer=len(qa["offbeat_note_indices"]) > 0, source_fields=["qa_summary.offbeat_note_indices"], confidence="high")

    add_understanding_fact(high, sample=sample, fact_id="named_interval_sequence", concept="interval_sequence", answer_type="sequence",
                           answer=qa["named_interval_sequence"], source_fields=["qa_summary.named_interval_sequence"], confidence="high")
    add_understanding_fact(high, sample=sample, fact_id="directed_named_interval_sequence", concept="interval_sequence", answer_type="sequence",
                           answer=qa["directed_named_interval_sequence"], source_fields=["qa_summary.directed_named_interval_sequence"], confidence="high")
    add_understanding_fact(high, sample=sample, fact_id="interval_name_counts", concept="interval_summary", answer_type="mapping",
                           answer=qa["interval_name_counts"], source_fields=["qa_summary.interval_name_counts"], confidence="high")

    for interval_name, count in sorted(qa["interval_name_counts"].items()):
        add_understanding_fact(high, sample=sample, fact_id=f"contains_interval_{interval_name}", concept="interval_presence", answer_type="boolean",
                               answer=count > 0, source_fields=["qa_summary.interval_name_counts"], confidence="high", extra={"interval_name": interval_name})
        add_understanding_fact(high, sample=sample, fact_id=f"count_interval_{interval_name}", concept="interval_count", answer_type="integer",
                               answer=count, source_fields=["qa_summary.interval_name_counts"], confidence="high", extra={"interval_name": interval_name})

    for beat_key, note_indices in sorted(qa["beat_note_indices"].items(), key=lambda kv: int(kv[0])):
        add_understanding_fact(high, sample=sample, fact_id=f"beat_{beat_key}_note_count", concept="beat_count", answer_type="integer",
                               answer=len(note_indices), source_fields=["qa_summary.beat_note_indices"], confidence="high", extra={"beat": int(beat_key)})

    return {
        "sample_id": sample.get("sample_id", ""), "split": sample.get("split", ""),
        "high_confidence_facts": high, "low_confidence_facts": low,
        "meta_summary": {"structure_mode": structure_mode, "qa_focus_tags": qa["qa_focus_tags"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str, help="Path to dataset root, e.g. two_bar_dataset")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    meta_slim_files = collect_meta_slim_files(dataset_root)
    if not meta_slim_files:
        raise FileNotFoundError(f"No meta_slim json files found under {dataset_root}")

    written, skipped = 0, 0
    for meta_slim_path in meta_slim_files:
        split = infer_split(meta_slim_path)
        grounding_path = dataset_root / split / "grounding_facts" / f"{meta_slim_path.stem}.grounding.json"
        understanding_path = dataset_root / split / "understanding_facts" / f"{meta_slim_path.stem}.understanding.json"

        if not args.overwrite and grounding_path.exists() and understanding_path.exists():
            skipped += 1
            continue

        sample = load_json(meta_slim_path)
        save_json(grounding_path, build_grounding_facts(sample))
        save_json(understanding_path, build_understanding_facts(sample))
        written += 1

    print("=== build_qa_facts ===")
    print(f"meta_slim files found: {len(meta_slim_files)}")
    print(f"written: {written}  skipped (already existed): {skipped}")


if __name__ == "__main__":
    main()
