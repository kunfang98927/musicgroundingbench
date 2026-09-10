"""Step 9: turn each sample's high-confidence understanding facts into
rendered understanding QA pairs (question text + free-form text answer),
balanced across the ABS/REL/ORD/PAT query families.

Structurally this mirrors 08_render_grounding_qa.py, but note the category
processing order and quotas are genuinely different constants for this task
(not a typo) - kept exactly as in the original for that reason. As with
step 8, `sample_facts_for_sample` consumes the RNG in an exact sequence that
must not be reordered; see the note in 08_render_grounding_qa.py and
docs/REPRODUCING.md.

Verified 100% byte-identical to the official released dataset (all 90,072
entries plus all.json / query_registry.json / summary.json). The constants
and orderings here were reverse-engineered against that dataset rather than
copied from the private repo's current code, which had drifted - so treat
`concept_priority_within_category`, the quotas, and the fps-based quantizer
as load-bearing. In particular the ABS order puts `mode` directly after
`meter`, and ORD puts `downbeat` first: that is what stops the quota loop
before it ever shuffles the multi-fact `tempo` / `beat_count` buckets, and
those un-drawn shuffles are exactly what keeps the RNG stream aligned.

Usage:
    python scripts/09_render_understanding_qa.py <dataset_root> [--seed 20260413]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_TOTAL_PER_SAMPLE = 8
DEFAULT_CATEGORY_QUOTAS = {"PAT": 3, "REL": 2, "ABS": 2, "ORD": 1}

DEFAULT_PER_SAMPLE_CONCEPT_LIMITS = {
    "interval_presence": 1, "interval_count": 1, "interval_sequence": 0, "interval_summary": 0,
    "bar_comparison": 1, "bar_sequence_relation": 1, "chord": 1, "chord_progression": 1,
    "harmonic_relation": 1, "chord_tone_ratio": 1, "beat_count": 1, "beat_alignment": 1,
    "downbeat": 1, "tempo": 1, "meter": 1, "mode": 1, "tonic": 1, "count": 1,
    "pitch_extreme": 1, "pitch_summary": 1, "repetition": 1, "harmonic_pattern": 1,
}

# Concepts whose facts are built and shuffled like any other, but are never actually
# asked about. They stay in the pool on purpose: the shuffle still consumes RNG, so
# dropping them upstream would change every later draw. (The released dataset's own
# summary.json reports these at limit 1 while containing zero questions about them -
# i.e. the generator filtered them somewhere its config dump didn't see.)
EXCLUDED_CONCEPTS = frozenset({
    "bar_sequence_relation", "chord", "beat_count", "beat_alignment",
    "tempo", "count", "pitch_extreme", "pitch_summary",
})
DEFAULT_PER_SAMPLE_CATEGORY_MAX = {"PAT": 4, "REL": 3, "ABS": 3, "ORD": 1}


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= (n % 100) <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
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


def unique_by_fact_id(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for fact in facts:
        fact_id = str(fact.get("fact_id"))
        if fact_id in seen:
            continue
        seen.add(fact_id)
        out.append(fact)
    return out


def shuffled_copy(xs: List[Any], rng: random.Random) -> List[Any]:
    ys = xs[:]
    rng.shuffle(ys)
    return ys


def stable_int_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf8")).hexdigest()[:8], 16)


def build_audio_path(dataset_root: Path, split: str, stem: str, audio_subdir: str) -> str:
    return (Path(split) / audio_subdir / f"{stem}.wav").as_posix()


def build_notes_from_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"pitch": int(e["pitch"]), "velocity": int(e["velocity"]),
             "start": round(float(e["onset_sec"]), 6), "end": round(float(e["offset_sec"]), 6)} for e in events]


def quantize_span_to_tokens(start_sec: float, end_sec: float, fps: float, num_time_tokens: int) -> List[int]:
    max_idx = num_time_tokens - 1
    start_tok = max(0, min(max_idx, int(round(float(start_sec) * fps))))
    end_tok = max(start_tok, min(max_idx, int(round(float(end_sec) * fps))))
    return [start_tok, end_tok]


def convert_spans_sec_to_token_spans(spans_sec: List[List[float]], fps: float, num_time_tokens: int) -> List[List[int]]:
    return [quantize_span_to_tokens(s, e, fps, num_time_tokens) for s, e in spans_sec]


# =============================================================================
# Per-clip sampling - same algorithm as 08_render_grounding_qa.py, but its own
# quotas/priorities/pass order (PAT, REL, ABS, ORD - not PAT, REL, ORD, ABS).
# =============================================================================

def concept_priority_within_category(category: str) -> List[str]:
    if category == "PAT":
        return ["repetition", "harmonic_pattern", "chord_progression", "interval_summary", "interval_sequence"]
    if category == "REL":
        return ["harmonic_relation", "chord_tone_ratio", "interval_presence", "interval_count", "bar_comparison", "bar_sequence_relation"]
    if category == "ABS":
        return ["meter", "mode", "tempo", "tonic", "count", "pitch_summary", "pitch_extreme", "chord"]
    if category == "ORD":
        return ["downbeat", "beat_count", "beat_alignment"]
    return []


def bucket_facts_by_category_and_concept(facts: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for fact in unique_by_fact_id(facts):
        out[str(fact["qa_category"])][str(fact["concept"])].append(fact)
    return out


def add_fact_if_allowed(
    selected: List[Dict[str, Any]], selected_ids: set, concept_counts: Dict[str, int], category_counts: Dict[str, int],
    fact: Dict[str, Any], per_sample_concept_limits: Dict[str, int], per_sample_category_max: Dict[str, int],
) -> bool:
    fid, concept, category = str(fact["fact_id"]), str(fact["concept"]), str(fact["qa_category"])
    if fid in selected_ids or concept in EXCLUDED_CONCEPTS:
        return False
    concept_limit = per_sample_concept_limits.get(concept)
    if concept_limit is not None and concept_counts[concept] >= concept_limit:
        return False
    category_limit = per_sample_category_max.get(category)
    if category_limit is not None and category_counts[category] >= category_limit:
        return False

    selected.append(fact)
    selected_ids.add(fid)
    concept_counts[concept] += 1
    category_counts[category] += 1
    return True


def sample_facts_for_sample(
    high_conf_facts: List[Dict[str, Any]], total_per_sample: int, category_quotas: Dict[str, int],
    per_sample_concept_limits: Dict[str, int], per_sample_category_max: Dict[str, int], rng: random.Random,
) -> List[Dict[str, Any]]:
    buckets = bucket_facts_by_category_and_concept(high_conf_facts)

    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()
    concept_counts: Dict[str, int] = defaultdict(int)
    category_counts: Dict[str, int] = defaultdict(int)

    for category in ["PAT", "REL", "ABS", "ORD"]:
        need = category_quotas.get(category, 0)
        if need <= 0:
            continue

        concepts = list(buckets.get(category, {}).keys())
        priority = concept_priority_within_category(category)
        ordered_concepts = [c for c in priority if c in concepts]
        rest = [c for c in concepts if c not in ordered_concepts]
        rng.shuffle(rest)
        ordered_concepts.extend(rest)

        for concept in ordered_concepts:
            if len(selected) >= total_per_sample or need <= 0:
                break
            for fact in shuffled_copy(buckets[category][concept], rng):
                if add_fact_if_allowed(selected, selected_ids, concept_counts, category_counts, fact,
                                        per_sample_concept_limits, per_sample_category_max):
                    need -= 1
                    break

        if need > 0:
            refill_pool = []
            for concept in ordered_concepts:
                refill_pool.extend(shuffled_copy(buckets[category][concept], rng))
            for fact in refill_pool:
                if len(selected) >= total_per_sample or need <= 0:
                    break
                if add_fact_if_allowed(selected, selected_ids, concept_counts, category_counts, fact,
                                        per_sample_concept_limits, per_sample_category_max):
                    need -= 1

    # Pass 2 walks the categories in the same concept order pass 1 used.
    refill_priority = [(c, concept_priority_within_category(c)) for c in ["PAT", "REL", "ABS", "ORD"]]
    refill_pool: List[Dict[str, Any]] = []
    for category, concept_order in refill_priority:
        cat_map = buckets.get(category, {})
        for concept in concept_order:
            if concept in cat_map:
                refill_pool.extend(shuffled_copy(cat_map[concept], rng))
        others = [c for c in cat_map.keys() if c not in concept_order]
        rng.shuffle(others)
        for concept in others:
            refill_pool.extend(shuffled_copy(cat_map[concept], rng))

    for fact in refill_pool:
        if len(selected) >= total_per_sample:
            break
        add_fact_if_allowed(selected, selected_ids, concept_counts, category_counts, fact,
                             per_sample_concept_limits, per_sample_category_max)

    return selected[:total_per_sample]


# =============================================================================
# Answer-text rendering: fact -> a short phrase (style "0") and two full
# sentences (styles "1"/"2"), one branch per fact_id/concept.
# =============================================================================

def answer_value_to_phrase(fact: Dict[str, Any]) -> str:
    fact_id = str(fact["fact_id"])
    concept = str(fact["concept"])
    answer = fact["answer"]

    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if fact_id == "time_signature":
        return f"{answer} time"
    if fact_id == "tempo_class":
        return f"a {str(answer).replace('_', ' ')} tempo"
    if fact_id == "tempo_bpm":
        return f"{answer} BPM"
    if fact_id == "tonal_mode":
        return "natural minor mode" if str(answer) == "natural_minor" else f"{str(answer).replace('_', ' ')} mode"
    if fact_id == "tonic_pc":
        return f"pitch class {answer}"
    if fact_id == "repetition_type":
        mapping = {"none": "no repetition", "exact_bar_repeat": "an exact repetition", "transposed_bar_repeat": "a transposed repetition"}
        return mapping.get(str(answer), str(answer).replace("_", " "))
    if fact_id == "harmonic_pattern_type":
        mapping = {"none": "no harmonic pattern", "ascending_arpeggio": "an ascending arpeggio pattern",
                   "broken_chord_repeat": "a broken-chord repetition pattern", "descending_arpeggio": "a descending arpeggio pattern",
                   "oscillating_arpeggio": "an oscillating arpeggio pattern"}
        return mapping.get(str(answer), str(answer).replace("_", " "))
    if fact_id in {"bar1_chord_type", "bar2_chord_type"}:
        mapping = {"I": "the tonic chord", "IV": "the subdominant chord", "V": "the dominant chord", "vi": "the submediant chord",
                   "i": "the tonic chord", "iv": "the subdominant chord", "v": "the dominant chord", "VI": "the submediant chord"}
        return mapping.get(str(answer), f"the {answer} chord")
    if fact_id in {"which_bar_has_more_notes", "which_bar_has_higher_peak"}:
        mapping = {"bar_1": "the first bar", "bar_2": "the second bar", "equal": "both bars"}
        return mapping.get(str(answer), str(answer).replace("_", " "))
    if fact_id == "harmonic_relation_type":
        mapping = {"same_chord": "the same chord", "changed_chord": "a chord change"}
        return mapping.get(str(answer), str(answer).replace("_", " "))
    if concept in {"interval_presence", "bar_sequence_relation", "beat_alignment", "downbeat"}:
        return "yes" if bool(answer) else "no"
    if isinstance(answer, list):
        return ", ".join(str(x).replace("_", " ") for x in answer)
    if isinstance(answer, dict):
        return "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in answer.items())
    if isinstance(answer, str):
        return answer.replace("_", " ")
    return str(answer)


def render_understanding_answer_texts(fact: Dict[str, Any]) -> Dict[str, str]:
    fact_id = str(fact["fact_id"])
    phrase = answer_value_to_phrase(fact)
    answer = fact["answer"]

    if fact_id == "time_signature":
        return {"0": phrase, "1": f"The time signature is {phrase}.", "2": f"This clip is in {phrase}."}
    if fact_id == "tempo_bpm":
        return {"0": phrase, "1": f"The tempo is {phrase}.", "2": f"The clip is played at a tempo of {phrase}."}
    if fact_id == "tempo_class":
        return {"0": phrase, "1": f"The tempo is {phrase}.", "2": f"This clip has {phrase}."}
    if fact_id == "tonal_mode":
        return {"0": phrase, "1": f"The mode is {phrase}.", "2": f"This clip is in {phrase}."}
    if fact_id == "tonic_pc":
        return {"0": phrase, "1": f"The tonic is {phrase}.", "2": f"The tonal center corresponds to {phrase}."}
    if fact_id == "num_notes":
        return {"0": phrase, "1": f"There are {answer} notes.", "2": f"This clip contains {answer} notes in total."}
    if fact_id == "bar1_note_count":
        return {"0": f"{answer} notes", "1": f"The first bar contains {answer} notes.", "2": f"There are {answer} notes in the first bar."}
    if fact_id == "bar2_note_count":
        return {"0": f"{answer} notes", "1": f"The second bar contains {answer} notes.", "2": f"There are {answer} notes in the second bar."}
    if fact_id == "pitch_span":
        return {"0": f"{answer} semitones", "1": f"The pitch span is {answer} semitones.",
                "2": f"The melody spans {answer} semitones from its lowest to highest pitch."}
    if fact_id == "unique_pitch_count":
        return {"0": f"{answer} pitches", "1": f"There are {answer} unique pitches.", "2": f"This clip contains {answer} distinct pitch values."}
    if fact_id == "mean_duration_beats":
        return {"0": f"{float(answer):.2f} beats", "1": f"The mean note duration is {float(answer):.2f} beats.",
                "2": f"On average, each note lasts about {float(answer):.2f} beats."}
    if fact_id in {"step_count", "skip_count", "leap_count", "direction_changes", "ascending_interval_count",
                    "descending_interval_count", "repeated_pitch_count"}:
        label = {"step_count": "stepwise motions", "skip_count": "skips", "leap_count": "leaps",
                 "direction_changes": "direction changes", "ascending_interval_count": "ascending intervals",
                 "descending_interval_count": "descending intervals", "repeated_pitch_count": "repeated pitches"}[fact_id]
        return {"0": f"{answer} {label}", "1": f"There are {answer} {label}.", "2": f"This clip contains {answer} {label}."}
    if fact_id in {"highest_pitch", "lowest_pitch"}:
        which = "highest" if fact_id == "highest_pitch" else "lowest"
        return {"0": f"MIDI {answer}", "1": f"The {which} pitch is MIDI {answer}.", "2": f"The {which} note in this clip has MIDI pitch {answer}."}
    if fact_id in {"bar1_chord_type", "bar2_chord_type"}:
        which = "first" if fact_id == "bar1_chord_type" else "second"
        return {"0": phrase, "1": f"The {which} bar uses {phrase}.", "2": f"The harmonic content of the {which} bar corresponds to {phrase}."}
    if fact_id == "chord_progression":
        progression_text = " then ".join(answer_value_to_phrase({"fact_id": "bar1_chord_type", "answer": x, "concept": "chord"}) for x in answer)
        return {"0": progression_text, "1": f"The chord progression is {progression_text}.",
                "2": f"The clip follows the chord progression {progression_text}."}
    if fact_id == "harmonic_relation_type":
        return {"0": phrase, "1": f"The harmonic relation is {phrase}.", "2": f"Between the two bars, the harmony shows {phrase}."}
    if fact_id in {"bar1_chord_tone_ratio", "bar2_chord_tone_ratio"}:
        which = "first" if fact_id == "bar1_chord_tone_ratio" else "second"
        pct = 100.0 * float(answer)
        return {"0": f"{pct:.1f}%", "1": f"The chord-tone ratio in the {which} bar is {pct:.1f}%.",
                "2": f"In the {which} bar, about {pct:.1f}% of the notes are chord tones."}
    if fact_id == "repetition_type":
        return {"0": phrase, "1": f"The repetition type is {phrase}.", "2": f"This melody shows {phrase} between the two bars."}
    if fact_id == "harmonic_pattern_type":
        return {"0": phrase, "1": f"The harmonic pattern is {phrase}.", "2": f"This clip exhibits {phrase}."}
    if fact_id in {"bar_pitch_sequence_equal", "bar_duration_sequence_equal", "bar_note_count_equal", "has_downbeat_notes", "has_offbeat_notes"}:
        yes_no = "yes" if bool(answer) else "no"
        sentence_by_id = {
            "bar_pitch_sequence_equal": ("The two bars have the same pitch sequence.", "The two bars do not have the same pitch sequence."),
            "bar_duration_sequence_equal": ("The two bars have the same duration sequence.", "The two bars do not have the same duration sequence."),
            "bar_note_count_equal": ("The two bars contain the same number of notes.", "The two bars do not contain the same number of notes."),
            "has_downbeat_notes": ("The clip contains downbeat notes.", "The clip does not contain downbeat notes."),
            "has_offbeat_notes": ("The clip contains offbeat notes.", "The clip does not contain offbeat notes."),
        }
        true_text, false_text = sentence_by_id[fact_id]
        sentence = true_text if bool(answer) else false_text
        return {"0": yes_no, "1": sentence, "2": sentence}
    if fact_id == "which_bar_has_more_notes":
        if str(answer) == "equal":
            return {"0": "both bars", "1": "Both bars have the same number of notes.",
                    "2": "Neither bar has more notes; both bars contain the same number of notes."}
        return {"0": phrase, "1": f"{phrase.capitalize()} has more notes.", "2": f"The bar with more notes is {phrase}."}
    if fact_id == "which_bar_has_higher_peak":
        return {"0": phrase, "1": f"{phrase.capitalize()} reaches the higher pitch peak.", "2": f"The higher pitch peak occurs in {phrase}."}
    if fact_id == "named_interval_sequence":
        return {"0": phrase, "1": f"The interval sequence is {phrase}.", "2": f"The melody moves through the interval sequence {phrase}."}
    if fact_id == "directed_named_interval_sequence":
        return {"0": phrase, "1": f"The directed interval sequence is {phrase}.", "2": f"The melody follows the directed interval sequence {phrase}."}
    if fact_id == "interval_name_counts":
        return {"0": phrase, "1": f"The interval counts are: {phrase}.", "2": f"The distribution of interval types is as follows: {phrase}."}
    if fact_id.startswith("contains_interval_"):
        interval_name = fact.get("interval_name", "").replace("_", " ")
        if bool(answer):
            return {"0": "yes", "1": f"Yes, the clip contains a {interval_name}.", "2": f"Yes, this clip includes at least one {interval_name}."}
        return {"0": "no", "1": f"No, the clip does not contain a {interval_name}.", "2": f"No, there is no {interval_name} in this clip."}
    if fact_id.startswith("count_interval_"):
        interval_name = fact.get("interval_name", "").replace("_", " ")
        return {"0": f"{answer} {interval_name}", "1": f"There are {answer} {interval_name} intervals.",
                "2": f"This clip contains {answer} instances of {interval_name}."}
    if fact_id.startswith("beat_") and fact_id.endswith("_note_count"):
        beat_ord = ordinal(int(fact.get("beat")))
        return {"0": f"{answer} notes", "1": f"There are {answer} notes starting on the {beat_ord} beat.",
                "2": f"{answer} notes begin on the {beat_ord} beat in this clip."}

    return {"0": phrase, "1": f"The answer is {phrase}.", "2": f"The correct answer is {phrase}."}


def render_question_and_key(fact: Dict[str, Any]) -> Tuple[str, List[str], str]:
    fact_id = str(fact["fact_id"])
    concept = str(fact["concept"])

    if fact_id == "time_signature":
        qv = ["What is the time signature of this clip?", "Identify the time signature used in this clip.", "Which time signature does this clip use?"]
        return qv[0], qv, "Q/ABS/METER/time_signature"
    if fact_id == "tempo_bpm":
        qv = ["What is the tempo of this clip in BPM?", "Identify the tempo of this clip in beats per minute.", "How fast is this clip in BPM?"]
        return qv[0], qv, "Q/ABS/TEMPO/tempo_bpm"
    if fact_id == "tempo_class":
        qv = ["What is the tempo class of this clip?", "Identify the tempo class of this clip.", "Is the tempo slow, medium, or fast?"]
        return qv[0], qv, "Q/ABS/TEMPO/tempo_class"
    if fact_id == "tonal_mode":
        qv = ["What is the tonal mode of this clip?", "Identify the mode of this clip.", "Is this clip in major or natural minor mode?"]
        return qv[0], qv, "Q/ABS/MODE/tonal_mode"
    if fact_id == "tonic_pc":
        qv = ["What is the tonic pitch class of this clip?", "Identify the tonic pitch class.", "Which pitch class acts as the tonic in this clip?"]
        return qv[0], qv, "Q/ABS/TONIC/tonic_pc"
    if fact_id == "num_notes":
        qv = ["How many notes are in this clip?", "What is the total number of notes in this clip?", "Count the notes in this clip."]
        return qv[0], qv, "Q/ABS/COUNT/num_notes"
    if fact_id in {"bar1_note_count", "bar2_note_count"}:
        which = "first" if fact_id == "bar1_note_count" else "second"
        qv = [f"How many notes are in the {which} bar?", f"What is the number of notes in the {which} bar?", f"Count the notes in the {which} bar."]
        return qv[0], qv, f"Q/ABS/BAR_COUNT/bar={which}"
    if fact_id == "pitch_span":
        qv = ["What is the pitch span of this clip?", "How wide is the pitch span of this melody?", "How many semitones does the melody span?"]
        return qv[0], qv, "Q/ABS/PITCH/pitch_span"
    if fact_id == "unique_pitch_count":
        qv = ["How many unique pitches are in this clip?", "What is the number of distinct pitches in this clip?", "Count the distinct pitches used in this clip."]
        return qv[0], qv, "Q/ABS/PITCH/unique_pitch_count"
    if fact_id == "mean_duration_beats":
        qv = ["What is the mean note duration in beats?", "What is the average note duration in beats?", "How long is a note on average in beats?"]
        return qv[0], qv, "Q/ABS/RHYTHM/mean_duration_beats"
    if fact_id in {"step_count", "skip_count", "leap_count", "direction_changes", "ascending_interval_count",
                    "descending_interval_count", "repeated_pitch_count"}:
        q1 = {"step_count": "How many stepwise motions are in this clip?", "skip_count": "How many skips are in this clip?",
              "leap_count": "How many leaps are in this clip?", "direction_changes": "How many melodic direction changes are in this clip?",
              "ascending_interval_count": "How many ascending intervals are in this clip?",
              "descending_interval_count": "How many descending intervals are in this clip?",
              "repeated_pitch_count": "How many repeated pitches are in this clip?"}[fact_id]
        qv = [q1, q1.replace("How many", "What is the number of"), q1]
        return qv[0], qv, f"Q/ABS/INTERVAL_SUMMARY/{fact_id}"
    if fact_id in {"highest_pitch", "lowest_pitch"}:
        which = "highest" if fact_id == "highest_pitch" else "lowest"
        qv = [f"What is the {which} pitch in this clip?", f"Identify the {which} pitch.", f"Which MIDI pitch is the {which} one in this clip?"]
        return qv[0], qv, f"Q/ABS/PITCH_EXTREME/{which}_pitch"
    if fact_id in {"bar1_chord_type", "bar2_chord_type"}:
        which = "first" if fact_id == "bar1_chord_type" else "second"
        qv = [f"What chord is implied in the {which} bar?", f"Identify the chord type of the {which} bar.", f"Which chord best matches the {which} bar?"]
        return qv[0], qv, f"Q/ABS/CHORD/bar={which}"
    if fact_id == "chord_progression":
        qv = ["What is the chord progression of this clip?", "Identify the chord progression across the two bars.", "Which chord progression does this clip follow?"]
        return qv[0], qv, "Q/PAT/HARMONY/chord_progression"
    if fact_id == "harmonic_relation_type":
        qv = ["What is the harmonic relation between the two bars?", "How does the harmony relate between the first and second bars?", "Is there a chord change between the two bars?"]
        return qv[0], qv, "Q/REL/HARMONY/harmonic_relation"
    if fact_id in {"bar1_chord_tone_ratio", "bar2_chord_tone_ratio"}:
        which = "first" if fact_id == "bar1_chord_tone_ratio" else "second"
        qv = [f"What is the chord-tone ratio in the {which} bar?", f"How many notes in the {which} bar are chord tones, proportionally?",
              f"What proportion of the notes in the {which} bar are chord tones?"]
        return qv[0], qv, f"Q/REL/HARMONY/chord_tone_ratio_bar={which}"
    if fact_id == "repetition_type":
        qv = ["What is the repetition type of this clip?", "Identify the repetition pattern between the two bars.", "What kind of repetition does this melody show?"]
        return qv[0], qv, "Q/PAT/REPETITION/type"
    if fact_id == "harmonic_pattern_type":
        qv = ["What harmonic pattern does this clip contain?", "Identify the harmonic pattern in this clip.", "Which harmonic pattern best describes this clip?"]
        return qv[0], qv, "Q/PAT/HARMONIC/type"
    if fact_id == "bar_pitch_sequence_equal":
        qv = ["Do the two bars have the same pitch sequence?", "Is the pitch sequence the same in both bars?", "Are the pitch sequences of the two bars identical?"]
        return qv[0], qv, "Q/REL/BAR_REL/pitch_sequence_equal"
    if fact_id == "bar_duration_sequence_equal":
        qv = ["Do the two bars have the same duration sequence?", "Is the rhythm duration sequence the same in both bars?", "Are the duration sequences of the two bars identical?"]
        return qv[0], qv, "Q/REL/BAR_REL/duration_sequence_equal"
    if fact_id == "bar_note_count_equal":
        qv = ["Do the two bars contain the same number of notes?", "Is the note count equal across the two bars?", "Do bar 1 and bar 2 have the same note count?"]
        return qv[0], qv, "Q/REL/BAR_REL/note_count_equal"
    if fact_id == "which_bar_has_more_notes":
        qv = ["Which bar has more notes?", "Which bar contains more notes?", "Is the first or second bar denser in note count?"]
        return qv[0], qv, "Q/REL/BAR_REL/which_bar_more_notes"
    if fact_id == "which_bar_has_higher_peak":
        qv = ["Which bar reaches the higher pitch peak?", "Which bar contains the higher highest pitch?", "Does the first or second bar reach a higher pitch peak?"]
        return qv[0], qv, "Q/REL/BAR_REL/which_bar_higher_peak"
    if fact_id == "has_downbeat_notes":
        qv = ["Does this clip contain downbeat notes?", "Are there any downbeat notes in this clip?", "Does the melody place any notes on the downbeat?"]
        return qv[0], qv, "Q/ORD/BEAT/has_downbeat_notes"
    if fact_id == "has_offbeat_notes":
        qv = ["Does this clip contain offbeat notes?", "Are there any offbeat notes in this clip?", "Does the melody include notes that fall off the beat?"]
        return qv[0], qv, "Q/ORD/BEAT/has_offbeat_notes"
    if fact_id == "named_interval_sequence":
        qv = ["What is the interval sequence of this clip?", "Identify the sequence of interval names in this melody.", "Which interval sequence does this melody follow?"]
        return qv[0], qv, "Q/PAT/INTERVAL/named_interval_sequence"
    if fact_id == "directed_named_interval_sequence":
        qv = ["What is the directed interval sequence of this clip?", "Identify the directional interval sequence in this melody.", "Which directed interval sequence does this melody follow?"]
        return qv[0], qv, "Q/PAT/INTERVAL/directed_named_interval_sequence"
    if fact_id == "interval_name_counts":
        qv = ["What are the counts of each interval type in this clip?", "Summarize the interval counts in this clip.", "How often does each interval type occur in this clip?"]
        return qv[0], qv, "Q/PAT/INTERVAL/interval_name_counts"
    if fact_id.startswith("contains_interval_"):
        interval_name = str(fact.get("interval_name", "")).replace("_", " ")
        qv = [f"Does this clip contain a {interval_name}?", f"Is there a {interval_name} in this clip?", f"Does the melody include at least one {interval_name}?"]
        return qv[0], qv, f"Q/REL/INTERVAL/contains={fact.get('interval_name')}"
    if fact_id.startswith("count_interval_"):
        interval_name = str(fact.get("interval_name", "")).replace("_", " ")
        qv = [f"How many {interval_name} intervals are in this clip?", f"What is the count of {interval_name} intervals in this clip?", f"How often does the {interval_name} occur in this clip?"]
        return qv[0], qv, f"Q/REL/INTERVAL/count={fact.get('interval_name')}"
    if fact_id.startswith("beat_") and fact_id.endswith("_note_count"):
        beat = fact.get("beat")
        beat_ord = ordinal(int(beat))
        qv = [f"How many notes start on the {beat_ord} beat?", f"How many notes have onset on the {beat_ord} beat?", f"How many notes begin on the {beat_ord} beat in this clip?"]
        return qv[0], qv, f"Q/ORD/BEAT/beat={beat}_note_count"

    # Fallback for any fact_id not special-cased above (none currently reach this branch).
    qv = [f"What is the answer for {fact_id}?", f"Identify the value of {fact_id}.", f"Provide the answer for {fact_id}."]
    return qv[0], qv, f"Q/{fact.get('qa_category')}/{concept}/{fact_id}"


def render_one_entry(
    fact: Dict[str, Any], sample: Dict[str, Any], dataset_root: Path, understanding_path: Path,
    audio_subdir: str, num_time_tokens: int, fps: float,
) -> Dict[str, Any]:
    question, question_variants, query_key = render_question_and_key(fact)
    answer_texts = render_understanding_answer_texts(fact)

    stem = understanding_path.name.replace(".understanding.json", "")
    split = infer_split_from_path(understanding_path)
    answer_spans_sec = [[round(float(s), 6), round(float(e), 6)] for s, e in fact["evidence_spans"]]

    entry = {
        "question": question, "question_variants": question_variants,
        "answer_text": answer_texts["1"], "answer_texts": answer_texts, "default_language_style": 1,
        "gold_answer": fact["answer"], "gold_answer_type": fact["answer_type"],
        "answer_spans": convert_spans_sec_to_token_spans(answer_spans_sec, fps=fps, num_time_tokens=num_time_tokens),
        "answer_spans_sec": answer_spans_sec,
        "evidence_note_indices": fact["evidence_note_indices"], "evidence_type": fact["evidence_type"],
        "query_key": query_key, "query_id": None, "question_token": None,
        "qa_category": fact["qa_category"], "concept": fact["concept"], "confidence": fact["confidence"],
        "source_fact_id": fact["fact_id"], "source_fields": fact["source_fields"],
        "audio_path": build_audio_path(dataset_root, split, stem, audio_subdir),
        "notes": build_notes_from_events(sample["events"]),
        "sample_id": sample.get("sample_id", stem), "split": split,
    }
    if fact.get("reason") is not None:
        entry["reason"] = fact["reason"]
    if "interval_name" in fact:
        entry["interval_name"] = fact["interval_name"]
    if "beat" in fact:
        entry["beat"] = fact["beat"]
    return entry


def assign_query_ids(entries: List[Dict[str, Any]]) -> None:
    keys = sorted(set(str(x["query_key"]) for x in entries))
    mapping = {k: i + 1 for i, k in enumerate(keys)}
    for entry in entries:
        qid = mapping[entry["query_key"]]
        entry["query_id"] = qid
        entry["question_token"] = f"<|Q{qid}|>"


def render_split_entries(
    dataset_root: Path, understanding_files: List[Path], total_per_sample: int, category_quotas: Dict[str, int],
    per_sample_concept_limits: Dict[str, int], per_sample_category_max: Dict[str, int], audio_subdir: str,
    num_time_tokens: int, fps: float, seed: int,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for understanding_path in understanding_files:
        sample = load_json(corresponding_meta_slim_path(dataset_root, understanding_path))
        payload = load_json(understanding_path)

        local_rng = random.Random(seed + stable_int_hash(understanding_path.name))
        chosen_facts = sample_facts_for_sample(
            high_conf_facts=payload.get("high_confidence_facts", []), total_per_sample=total_per_sample,
            category_quotas=category_quotas, per_sample_concept_limits=per_sample_concept_limits,
            per_sample_category_max=per_sample_category_max, rng=local_rng,
        )
        for fact in chosen_facts:
            entries.append(render_one_entry(fact, sample, dataset_root, understanding_path, audio_subdir, num_time_tokens, fps))

    return entries


def summarize_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_category: Dict[str, int] = defaultdict(int)
    by_concept: Dict[str, int] = defaultdict(int)
    by_query_key: Dict[str, int] = defaultdict(int)
    for x in entries:
        by_category[str(x["qa_category"])] += 1
        by_concept[str(x["concept"])] += 1
        by_query_key[str(x["query_key"])] += 1
    return {"num_entries": len(entries), "qa_category_counts": dict(sorted(by_category.items())),
            "concept_counts": dict(sorted(by_concept.items())), "query_key_counts": dict(sorted(by_query_key.items()))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=str)
    ap.add_argument("--audio_subdir", type=str, default="wav")
    ap.add_argument("--num_time_tokens", type=int, default=750)
    ap.add_argument("--fps", type=float, default=75.0)
    ap.add_argument("--total_per_sample", type=int, default=DEFAULT_TOTAL_PER_SAMPLE)
    ap.add_argument("--seed", type=int, default=20260413)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    all_understanding_files = collect_understanding_fact_files(dataset_root)
    if not all_understanding_files:
        raise FileNotFoundError(f"No understanding facts found under {dataset_root}")

    by_split: Dict[str, List[Path]] = defaultdict(list)
    for p in all_understanding_files:
        by_split[infer_split_from_path(p)].append(p)

    output_root = dataset_root / "understanding_qa"
    output_root.mkdir(parents=True, exist_ok=True)

    category_quotas = dict(DEFAULT_CATEGORY_QUOTAS)
    if sum(category_quotas.values()) != args.total_per_sample:
        raise ValueError(f"Understanding quotas must sum to total_per_sample={args.total_per_sample}, but got {sum(category_quotas.values())}.")
    per_sample_concept_limits = dict(DEFAULT_PER_SAMPLE_CONCEPT_LIMITS)
    per_sample_category_max = dict(DEFAULT_PER_SAMPLE_CATEGORY_MAX)

    split_entries_dict: Dict[str, List[Dict[str, Any]]] = {}
    for split in ["train", "val", "test"]:
        out_path = output_root / f"{split}.json"
        if out_path.exists() and not args.overwrite:
            split_entries_dict[split] = load_json(out_path)
        else:
            split_entries_dict[split] = render_split_entries(
                dataset_root=dataset_root, understanding_files=by_split.get(split, []),
                total_per_sample=args.total_per_sample, category_quotas=category_quotas,
                per_sample_concept_limits=per_sample_concept_limits, per_sample_category_max=per_sample_category_max,
                audio_subdir=args.audio_subdir, num_time_tokens=args.num_time_tokens, fps=args.fps, seed=args.seed,
            )

    combined_entries = [e for split in ["train", "val", "test"] for e in split_entries_dict[split]]
    assign_query_ids(combined_entries)

    split_summaries = {}
    for split in ["train", "val", "test"]:
        save_json(output_root / f"{split}.json", split_entries_dict[split])
        split_summaries[split] = summarize_entries(split_entries_dict[split])

    save_json(output_root / "all.json", combined_entries)
    save_json(output_root / "query_registry.json", {
        "query_key_to_id": {x["query_key"]: x["query_id"] for x in sorted(combined_entries, key=lambda d: (d["query_id"], d["query_key"]))}
    })
    save_json(output_root / "summary.json", {
        "config": {
            "audio_subdir": args.audio_subdir, "num_time_tokens": args.num_time_tokens, "total_per_sample": args.total_per_sample,
            "category_quotas": category_quotas, "per_sample_concept_limits": per_sample_concept_limits,
            "per_sample_category_max": per_sample_category_max, "seed": args.seed,
            "high_confidence_only": True, "answer_text_style_default": 1,
        },
        "splits": split_summaries, "combined": summarize_entries(combined_entries),
    })

    print("=== render_understanding_qa ===")
    for split in ["train", "val", "test"]:
        print(f"{split}: {split_summaries[split]['num_entries']} entries")
    print(f"[DONE] wrote {output_root}/{{train,val,test,all}}.json, query_registry.json, summary.json")


if __name__ == "__main__":
    main()
