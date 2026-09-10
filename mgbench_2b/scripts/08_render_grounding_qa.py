"""Step 8: turn each sample's high-confidence grounding facts into rendered
grounding QA pairs (question text + time-span answer), balanced across the
ABS/REL/ORD/PAT query families.

This file is kept close to a 1:1 port of the original, deliberately NOT
restructured for style: `sample_facts_for_sample` below consumes the RNG in
an exact sequence (which category/concept gets shuffled, in what order, how
many times) - reordering *any* of that, even without changing the sampling
*rules*, would make the same --seed draw different questions.

Verified 100% byte-identical against a real copy of the official released
dataset (all splits) - see docs/REPRODUCING.md. The one thing that matters
enough to call out: the render seed (--seed, default 20260413) is a
*different* seed than generation's (--seed on 01_generate_midi.py, 1024) -
passing the generation seed here silently produces a wrong-but-plausible
dataset instead of an error.

Usage:
    python scripts/08_render_grounding_qa.py <dataset_root> [--seed 20260413]
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

# How many of the total_per_sample questions should come from each query
# family (must sum to total_per_sample).
DEFAULT_CATEGORY_QUOTAS = {"PAT": 3, "REL": 3, "ORD": 1, "ABS": 1}

# Cap on how many questions about the same *concept* (e.g. "chord_tone",
# "named_interval_target") can appear for one clip, so one prolific concept
# can't crowd out the rest.
DEFAULT_PER_SAMPLE_CONCEPT_LIMITS = {
    "named_interval_target": 1, "interval_target": 1, "chord_tone": 4, "non_chord_tone": 4,
    "repetition_pattern": 4, "harmonic_pattern": 4, "bar": 1, "beat": 1, "downbeat": 1,
    "beat_alignment": 1, "note_order": 1, "note_identity": 1, "pitch_extreme": 1,
}
DEFAULT_PER_SAMPLE_CATEGORY_MAX = {"PAT": 4, "REL": 5, "ABS": 2, "ORD": 2}


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


def collect_grounding_fact_files(dataset_root: Path) -> List[Path]:
    files: List[Path] = []
    for split in ["train", "val", "test"]:
        d = dataset_root / split / "grounding_facts"
        if d.exists():
            files.extend(sorted(d.glob("*.grounding.json")))
    return files


def corresponding_meta_slim_path(dataset_root: Path, grounding_path: Path) -> Path:
    split = infer_split_from_path(grounding_path)
    stem = grounding_path.name.replace(".grounding.json", "")
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


# =============================================================================
# Question / answer text templates - one branch per fact_id family, each
# giving 3 phrasing variants (question_variants) plus the canonical
# query_key used to group facts into families for the paper's ABS/REL/ORD/PAT
# analysis.
# =============================================================================

_INTERVAL_NAME_TEXT = {
    "unison": "unison", "repeated_unison": "repeated note", "minor_second": "minor second",
    "major_second": "major second", "minor_third": "minor third", "major_third": "major third",
    "perfect_fourth": "perfect fourth", "tritone": "tritone", "perfect_fifth": "perfect fifth",
    "minor_sixth": "minor sixth", "major_sixth": "major sixth", "minor_seventh": "minor seventh",
    "major_seventh": "major seventh", "octave": "octave",
}


def human_interval_name(name: str) -> str:
    if name.startswith("ascending_"):
        return f"ascending {human_interval_name(name[len('ascending_'):])}"
    if name.startswith("descending_"):
        return f"descending {human_interval_name(name[len('descending_'):])}"
    return _INTERVAL_NAME_TEXT.get(name, name.replace("_", " "))


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= (n % 100) <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_time_for_style0(x: float) -> str:
    return f"{x:.2f}s"


def format_time_for_text(x: float) -> str:
    return f"{x:.2f} seconds"


def style0_span_text(spans_sec: List[List[float]]) -> str:
    return "; ".join(f"[{format_time_for_style0(s)}, {format_time_for_style0(e)}]" for s, e in spans_sec)


def style1_span_text(spans_sec: List[List[float]]) -> str:
    if not spans_sec:
        return ""
    parts = [f"From {format_time_for_text(s)} to {format_time_for_text(e)}" for s, e in spans_sec]
    return "; ".join(parts) + "."


def style2_span_text(spans_sec: List[List[float]], subject: str) -> str:
    if not spans_sec:
        return ""
    if len(spans_sec) == 1:
        s, e = spans_sec[0]
        return f"{subject} is located from {format_time_for_text(s)} to {format_time_for_text(e)}."
    parts = [f"from {format_time_for_text(s)} to {format_time_for_text(e)}" for s, e in spans_sec]
    return f"{subject} are located in the following intervals: {'; '.join(parts)}."


def quantize_span_to_tokens(start_sec: float, end_sec: float, fps: float, num_time_tokens: int) -> List[int]:
    max_idx = num_time_tokens - 1
    start_tok = max(0, min(max_idx, int(round(float(start_sec) * fps))))
    end_tok = max(start_tok, min(max_idx, int(round(float(end_sec) * fps))))
    return [start_tok, end_tok]


def convert_spans_sec_to_token_spans(spans_sec: List[List[float]], fps: float, num_time_tokens: int) -> List[List[int]]:
    return [quantize_span_to_tokens(s, e, fps=fps, num_time_tokens=num_time_tokens) for s, e in spans_sec]


def render_question_and_key(fact: Dict[str, Any]) -> Tuple[str, List[str], str]:
    fact_id = str(fact["fact_id"])
    qa_category = str(fact["qa_category"])
    target = str(fact.get("target", ""))
    concept = str(fact["concept"])

    if fact_id == "highest_note":
        qv = ["Locate the highest note.", "Find the time span of the highest note.", "Which time interval contains the highest note?"]
        return qv[0], qv, "Q/ABS/PITCH_EXTREME/highest_note"

    if fact_id == "lowest_note":
        qv = ["Locate the lowest note.", "Find the time span of the lowest note.", "Which time interval contains the lowest note?"]
        return qv[0], qv, "Q/ABS/PITCH_EXTREME/lowest_note"

    if fact_id == "first_note":
        qv = ["Locate the first note.", "Find the time span of the first note.", "Which time interval contains the first note?"]
        return qv[0], qv, "Q/ORD/NOTE/first"

    if fact_id == "last_note":
        qv = ["Locate the last note.", "Find the time span of the last note.", "Which time interval contains the last note?"]
        return qv[0], qv, "Q/ORD/NOTE/last"

    if fact_id.startswith("note_"):
        idx1 = int(fact_id.split("_", 1)[1]) + 1
        qv = [f"Locate the {ordinal(idx1)} note in the sequence.", f"Find the time span of the {ordinal(idx1)} note.",
              f"Which time interval contains the {ordinal(idx1)} note in the sequence?"]
        return qv[0], qv, f"Q/ORD/NOTE/index={idx1}"

    if fact_id == "bar_1":
        qv = ["Locate the first bar.", "Find the time span of the first bar.", "Which time interval corresponds to the first bar?"]
        return qv[0], qv, "Q/ORD/BAR/bar=1"

    if fact_id == "bar_2":
        qv = ["Locate the second bar.", "Find the time span of the second bar.", "Which time interval corresponds to the second bar?"]
        return qv[0], qv, "Q/ORD/BAR/bar=2"

    if fact_id.startswith("beat_") and fact_id.endswith("_notes"):
        beat_ord = ordinal(int(fact_id.split("_")[1]))
        qv = [f"Locate the note(s) starting on the {beat_ord} beat.",
              f"Find the time span(s) of the note(s) whose onset is on the {beat_ord} beat.",
              f"Which time interval(s) contain the note(s) starting on the {beat_ord} beat?"]
        return qv[0], qv, f"Q/ORD/BEAT/beat={int(fact_id.split('_')[1])}"

    if fact_id == "downbeat_notes":
        qv = ["Locate the downbeat note(s).", "Find the time span(s) of the downbeat note(s).", "Which time interval(s) contain the downbeat note(s)?"]
        return qv[0], qv, "Q/ORD/BEAT/downbeat"

    if fact_id == "on_beat_notes":
        qv = ["Locate the on-beat note(s).", "Find the time span(s) of the on-beat note(s).", "Which time interval(s) contain the on-beat note(s)?"]
        return qv[0], qv, "Q/ORD/RHYTHM/on_beat"

    if fact_id == "offbeat_notes":
        qv = ["Locate the offbeat note(s).", "Find the time span(s) of the offbeat note(s).", "Which time interval(s) contain the offbeat note(s)?"]
        return qv[0], qv, "Q/ORD/RHYTHM/offbeat"

    if fact_id == "step_target_notes":
        qv = ["Locate the note(s) reached by stepwise motion.", "Find the time span(s) of the note(s) reached by a step.",
              "Which time interval(s) contain the target note(s) of stepwise motion?"]
        return qv[0], qv, "Q/REL/INTERVAL/type=step"

    if fact_id == "leap_target_notes":
        qv = ["Locate the note(s) reached by leap motion.", "Find the time span(s) of the note(s) reached by a leap.",
              "Which time interval(s) contain the target note(s) of leap motion?"]
        return qv[0], qv, "Q/REL/INTERVAL/type=leap"

    if fact_id.startswith("interval_"):
        interval_label = human_interval_name(target)
        qv = [f"Locate the note(s) reached by a {interval_label}.", f"Find the time span(s) of the note(s) reached by a {interval_label}.",
              f"Which time interval(s) contain the note(s) reached by a {interval_label}?"]
        return qv[0], qv, f"Q/REL/INTERVAL/name={target}"

    if fact_id == "bar1_chord_tone_notes":
        qv = ["Locate the chord-tone note(s) in the first bar.", "Find the time span(s) of the chord-tone note(s) in the first bar.",
              "Which time interval(s) contain the chord-tone note(s) in the first bar?"]
        return qv[0], qv, "Q/REL/HARMONY/chord_tone_bar=1"

    if fact_id == "bar2_chord_tone_notes":
        qv = ["Locate the chord-tone note(s) in the second bar.", "Find the time span(s) of the chord-tone note(s) in the second bar.",
              "Which time interval(s) contain the chord-tone note(s) in the second bar?"]
        return qv[0], qv, "Q/REL/HARMONY/chord_tone_bar=2"

    if fact_id == "bar1_non_chord_tone_notes":
        qv = ["Locate the non-chord-tone note(s) in the first bar.", "Find the time span(s) of the non-chord-tone note(s) in the first bar.",
              "Which time interval(s) contain the non-chord-tone note(s) in the first bar?"]
        return qv[0], qv, "Q/REL/HARMONY/non_chord_tone_bar=1"

    if fact_id == "bar2_non_chord_tone_notes":
        qv = ["Locate the non-chord-tone note(s) in the second bar.", "Find the time span(s) of the non-chord-tone note(s) in the second bar.",
              "Which time interval(s) contain the non-chord-tone note(s) in the second bar?"]
        return qv[0], qv, "Q/REL/HARMONY/non_chord_tone_bar=2"

    if fact_id == "repetition_pattern_notes":
        qv = ["Locate the repetition pattern.", "Find the time span(s) of the repetition pattern.", "Which time interval(s) contain the repetition pattern?"]
        return qv[0], qv, "Q/PAT/REPETITION/pattern"

    if fact_id == "exact_repeat_pattern_notes":
        qv = ["Locate the exact repetition pattern.", "Find the time span(s) of the exact repetition pattern.",
              "Which time interval(s) contain the exact repetition pattern?"]
        return qv[0], qv, "Q/PAT/REPETITION/type=exact"

    if fact_id == "transposed_repeat_pattern_notes":
        qv = ["Locate the transposed repetition pattern.", "Find the time span(s) of the transposed repetition pattern.",
              "Which time interval(s) contain the transposed repetition pattern?"]
        return qv[0], qv, "Q/PAT/REPETITION/type=transposed"

    if fact_id == "harmonic_pattern_notes":
        qv = ["Locate the harmonic pattern.", "Find the time span(s) of the harmonic pattern.", "Which time interval(s) contain the harmonic pattern?"]
        return qv[0], qv, "Q/PAT/HARMONIC/pattern"

    if fact_id == "ascending_arpeggio_pattern_notes":
        qv = ["Locate the ascending arpeggio pattern.", "Find the time span(s) of the ascending arpeggio pattern.",
              "Which time interval(s) contain the ascending arpeggio pattern?"]
        return qv[0], qv, "Q/PAT/HARMONIC/type=ascending_arpeggio"

    if fact_id == "broken_chord_repeat_pattern_notes":
        qv = ["Locate the broken-chord repetition pattern.", "Find the time span(s) of the broken-chord repetition pattern.",
              "Which time interval(s) contain the broken-chord repetition pattern?"]
        return qv[0], qv, "Q/PAT/HARMONIC/type=broken_chord_repeat"

    # Fallback for any fact_id not special-cased above (none currently reach this branch).
    qv = [f"Locate the target event for {fact_id}.", f"Find the time span(s) associated with {fact_id}.",
          f"Which time interval(s) correspond to {fact_id}?"]
    return qv[0], qv, f"Q/{qa_category}/{concept}/{fact_id}"


def render_grounding_answer_texts(fact: Dict[str, Any]) -> Dict[str, str]:
    spans_sec = fact["time_spans"]
    fact_id = str(fact["fact_id"])

    if fact_id == "highest_note":
        subject = "The highest note"
    elif fact_id == "lowest_note":
        subject = "The lowest note"
    elif fact_id == "first_note":
        subject = "The first note"
    elif fact_id == "last_note":
        subject = "The last note"
    elif fact_id.startswith("note_"):
        subject = f"The {ordinal(int(fact_id.split('_', 1)[1]) + 1)} note"
    elif fact_id == "bar_1":
        subject = "The first bar"
    elif fact_id == "bar_2":
        subject = "The second bar"
    elif fact_id.startswith("beat_") and fact_id.endswith("_notes"):
        subject = f"The note or notes starting on the {ordinal(int(fact_id.split('_')[1]))} beat"
    elif fact_id == "downbeat_notes":
        subject = "The downbeat note or notes"
    elif fact_id == "on_beat_notes":
        subject = "The on-beat note or notes"
    elif fact_id == "offbeat_notes":
        subject = "The offbeat note or notes"
    elif fact_id == "step_target_notes":
        subject = "The note or notes reached by stepwise motion"
    elif fact_id == "leap_target_notes":
        subject = "The note or notes reached by leap motion"
    elif fact_id.startswith("interval_"):
        subject = f"The note or notes reached by a {human_interval_name(str(fact.get('target', '')))}"
    elif fact_id.endswith("_pattern_notes"):
        subject = "The pattern"
    elif "chord_tone" in fact_id:
        subject = "The relevant note or notes"
    else:
        subject = "The target"

    return {"0": style0_span_text(spans_sec), "1": style1_span_text(spans_sec), "2": style2_span_text(spans_sec, subject)}


def build_notes_from_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"pitch": int(e["pitch"]), "velocity": int(e["velocity"]),
             "start": round(float(e["onset_sec"]), 6), "end": round(float(e["offset_sec"]), 6)} for e in events]


# =============================================================================
# Per-clip sampling: pick `total_per_sample` facts out of all of this clip's
# high-confidence facts, respecting category quotas and per-concept/category
# caps. The exact order of rng.shuffle() calls below is load-bearing for
# reproducibility - see the module docstring.
# =============================================================================

def bucket_facts_by_category_and_concept(facts: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for fact in unique_by_fact_id(facts):
        out[str(fact["qa_category"])][str(fact["concept"])].append(fact)
    return out


def shuffled_copy(xs: List[Any], rng: random.Random) -> List[Any]:
    ys = xs[:]
    rng.shuffle(ys)
    return ys


def stable_int_hash(s: str) -> int:
    """MD5-based hash -> int, so per-clip seeds are stable across machines/Python
    versions (unlike builtin hash(), which is randomized per-process by default)."""
    return int(hashlib.md5(s.encode("utf8")).hexdigest()[:8], 16)


def concept_priority_within_category(category: str) -> List[str]:
    if category == "PAT":
        return ["repetition_pattern", "harmonic_pattern"]
    if category == "REL":
        return ["chord_tone", "non_chord_tone", "interval_target", "named_interval_target"]
    if category == "ORD":
        return ["bar", "beat", "downbeat", "beat_alignment", "note_order", "note_identity"]
    if category == "ABS":
        return ["pitch_extreme"]
    return []


def add_fact_if_allowed(
    selected: List[Dict[str, Any]], selected_ids: set, concept_counts: Dict[str, int], category_counts: Dict[str, int],
    fact: Dict[str, Any], per_sample_concept_limits: Dict[str, int], per_sample_category_max: Dict[str, int],
) -> bool:
    fid, concept, category = str(fact["fact_id"]), str(fact["concept"]), str(fact["qa_category"])
    if fid in selected_ids:
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

    # Pass 1: satisfy each category's quota, spreading across concepts within it.
    for category in ["PAT", "REL", "ORD", "ABS"]:
        need = category_quotas.get(category, 0)
        if need <= 0:
            continue

        concepts = list(buckets.get(category, {}).keys())
        priority = concept_priority_within_category(category)
        ordered_concepts = [c for c in priority if c in concepts]
        rest = [c for c in concepts if c not in ordered_concepts]
        rng.shuffle(rest)
        ordered_concepts.extend(rest)

        # Round 1a: at most one fact per concept, in priority order.
        for concept in ordered_concepts:
            if len(selected) >= total_per_sample or need <= 0:
                break
            for fact in shuffled_copy(buckets[category][concept], rng):
                if add_fact_if_allowed(selected, selected_ids, concept_counts, category_counts, fact,
                                        per_sample_concept_limits, per_sample_category_max):
                    need -= 1
                    break

        # Round 1b: quota still not met - refill from anywhere in this category.
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

    # Pass 2: global refill up to total_per_sample. Fixed priority order below
    # (not the same order as Pass 1) keeps named_interval_target - which exists
    # for nearly every clip - from crowding out rarer concepts during refill.
    refill_priority = [
        ("PAT", ["repetition_pattern", "harmonic_pattern"]),
        ("REL", ["chord_tone", "non_chord_tone", "interval_target", "named_interval_target"]),
        ("ABS", ["pitch_extreme"]),
        ("ORD", ["bar", "beat", "downbeat", "beat_alignment", "note_order", "note_identity"]),
    ]
    refill_pool: List[Dict[str, Any]] = []
    for category, concept_order in refill_priority:
        cat_map = buckets.get(category, {})
        for concept in concept_order:
            if concept in cat_map:
                refill_pool.extend(shuffled_copy(cat_map[concept], rng))
        other_concepts = [c for c in cat_map.keys() if c not in concept_order]
        rng.shuffle(other_concepts)
        for concept in other_concepts:
            refill_pool.extend(shuffled_copy(cat_map[concept], rng))

    for fact in refill_pool:
        if len(selected) >= total_per_sample:
            break
        add_fact_if_allowed(selected, selected_ids, concept_counts, category_counts, fact,
                             per_sample_concept_limits, per_sample_category_max)

    return selected[:total_per_sample]


def build_audio_path(dataset_root: Path, split: str, stem: str, audio_subdir: str) -> str:
    return (Path(split) / audio_subdir / f"{stem}.wav").as_posix()


def render_one_entry(
    fact: Dict[str, Any], sample: Dict[str, Any], dataset_root: Path, grounding_path: Path,
    audio_subdir: str, num_time_tokens: int, fps: float,
) -> Dict[str, Any]:
    question, question_variants, query_key = render_question_and_key(fact)
    answer_texts = render_grounding_answer_texts(fact)

    stem = grounding_path.name.replace(".grounding.json", "")
    split = infer_split_from_path(grounding_path)
    answer_spans_sec = [[round(float(s), 6), round(float(e), 6)] for s, e in fact["time_spans"]]

    return {
        "question": question, "question_variants": question_variants,
        "answer_text": answer_texts["1"], "answer_texts": answer_texts, "default_language_style": 1,
        "answer_spans": convert_spans_sec_to_token_spans(answer_spans_sec, fps=fps, num_time_tokens=num_time_tokens),
        "answer_spans_sec": answer_spans_sec, "query_key": query_key, "query_id": None, "question_token": None,
        "qa_category": fact["qa_category"], "concept": fact["concept"], "confidence": fact["confidence"],
        "audio_path": build_audio_path(dataset_root, split, stem, audio_subdir),
        "notes": build_notes_from_events(sample["events"]),
        "sample_id": sample.get("sample_id", stem), "split": split, "source_fact_id": fact["fact_id"],
    }


def assign_query_ids(entries: List[Dict[str, Any]]) -> None:
    """Assign each distinct query_key a stable integer id (alphabetical order),
    used as the `<|QN|>` question token for token-based training."""
    keys = sorted(set(str(x["query_key"]) for x in entries))
    mapping = {k: i + 1 for i, k in enumerate(keys)}
    for entry in entries:
        qid = mapping[entry["query_key"]]
        entry["query_id"] = qid
        entry["question_token"] = f"<|Q{qid}|>"


def render_split_entries(
    dataset_root: Path, grounding_files: List[Path], total_per_sample: int, category_quotas: Dict[str, int],
    per_sample_concept_limits: Dict[str, int], audio_subdir: str, num_time_tokens: int, fps: float, seed: int,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for grounding_path in grounding_files:
        sample = load_json(corresponding_meta_slim_path(dataset_root, grounding_path))
        payload = load_json(grounding_path)

        # Per-clip seed = global seed + a stable hash of the filename, so
        # sampling is reproducible per-clip regardless of what order files
        # are processed in (and independent of any other clip's draws).
        local_rng = random.Random(seed + stable_int_hash(grounding_path.name))
        chosen_facts = sample_facts_for_sample(
            high_conf_facts=payload.get("high_confidence_facts", []), total_per_sample=total_per_sample,
            category_quotas=category_quotas, per_sample_concept_limits=per_sample_concept_limits,
            per_sample_category_max=DEFAULT_PER_SAMPLE_CATEGORY_MAX, rng=local_rng,
        )
        for fact in chosen_facts:
            entries.append(render_one_entry(fact, sample, dataset_root, grounding_path, audio_subdir, num_time_tokens, fps))

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
    all_grounding_files = collect_grounding_fact_files(dataset_root)
    if not all_grounding_files:
        raise FileNotFoundError(f"No grounding facts found under {dataset_root}")

    by_split: Dict[str, List[Path]] = defaultdict(list)
    for p in all_grounding_files:
        by_split[infer_split_from_path(p)].append(p)

    output_root = dataset_root / "grounding_qa"
    output_root.mkdir(parents=True, exist_ok=True)

    category_quotas = dict(DEFAULT_CATEGORY_QUOTAS)
    if sum(category_quotas.values()) != args.total_per_sample:
        raise ValueError(f"Grounding quotas must sum to total_per_sample={args.total_per_sample}, but got {sum(category_quotas.values())}.")
    per_sample_concept_limits = dict(DEFAULT_PER_SAMPLE_CONCEPT_LIMITS)

    split_entries_dict: Dict[str, List[Dict[str, Any]]] = {}
    for split in ["train", "val", "test"]:
        out_path = output_root / f"{split}.json"
        if out_path.exists() and not args.overwrite:
            split_entries_dict[split] = load_json(out_path)
        else:
            split_entries_dict[split] = render_split_entries(
                dataset_root=dataset_root, grounding_files=by_split.get(split, []),
                total_per_sample=args.total_per_sample, category_quotas=category_quotas,
                per_sample_concept_limits=per_sample_concept_limits, audio_subdir=args.audio_subdir,
                num_time_tokens=args.num_time_tokens, fps=args.fps, seed=args.seed,
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
            "per_sample_category_max": DEFAULT_PER_SAMPLE_CATEGORY_MAX, "seed": args.seed,
            "high_confidence_only": True, "answer_text_style_default": 1,
        },
        "splits": split_summaries, "combined": summarize_entries(combined_entries),
    })

    print("=== render_grounding_qa ===")
    for split in ["train", "val", "test"]:
        print(f"{split}: {split_summaries[split]['num_entries']} entries")
    print(f"[DONE] wrote {output_root}/{{train,val,test,all}}.json, query_registry.json, summary.json")


if __name__ == "__main__":
    main()
