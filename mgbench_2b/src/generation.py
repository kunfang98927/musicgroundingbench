"""Procedural generation of a single two-bar monophonic piano excerpt.

Generation happens in four stages, each consuming the same seeded
`random.Random` instance so a fixed top-level seed reproduces the whole
dataset:

  1. sample_controls    - pick the control variables (meter, tempo, tonal
                           mode, structure_mode, ...) that describe *what
                           kind* of excerpt this will be.
  2. generate_rhythm_skeleton - lay out note onsets/durations (in beats,
                           not yet pitched) consistent with those controls.
  3. realize_pitch_sequence  - assign a MIDI pitch to each onset.
  4. generate_velocities     - assign a MIDI velocity to each onset.

`generate_one_sample` runs all four stages and returns the raw sample;
timing humanization and validation happen one level up, in validation.py.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    BAR_RELATION_TYPES,
    INTERVAL_PROFILES,
    LEADING_SILENCE_MAX_SEC,
    LEADING_SILENCE_MIN_SEC,
    MAX_CLIP_DURATION_SEC,
    METER_CONFIGS,
    REGISTERS,
    REGISTER_BOUNDS,
    REPETITION_SHIFT_OPTIONS,
    STRONG_BEATS,
    SYNCOPATION_LEVELS,
    TEMPO_OPTIONS,
    TEMPO_TO_CLASS,
    TONAL_MODES,
    VELOCITY_PATTERN_TYPES,
    BasicInfo,
    ControlAttributes,
)
from src.music_theory import (
    allowed_pitches_in_range,
    chord_pitch_classes,
    clamp,
    duration_symbol,
    nearest_pitch_with_pc,
    scale_pitch_classes,
)

# Bar rhythm templates used only when structure_mode == "explicit_meter":
# fixed onset patterns (in beats) that make the meter unambiguous to hear.
EXPLICIT_METER_BAR_TEMPLATES = {
    "2/4": [[1.0, 1.0], [0.5, 0.5, 1.0], [1.0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
    "3/4": [[1.0, 1.0, 1.0], [1.5, 0.5, 1.0], [1.0, 0.5, 0.5, 1.0], [0.5, 0.5, 1.0, 1.0]],
    "4/4": [[1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 0.5, 0.5, 1.0], [0.5, 0.5, 1.0, 1.0, 1.0]],
    "6/8": [[3.0, 3.0], [1.0, 1.0, 1.0, 3.0], [3.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
}

# Bar rhythm templates used only when structure_mode == "explicit_harmonic_pattern":
# even eighth-note runs, so the harmonic-pattern QA has a clean beat grid to sit on.
HARMONIC_BAR_TEMPLATES = {
    "2/4": [[0.5, 0.5, 0.5, 0.5]],
    "3/4": [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
    "4/4": [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
    "6/8": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
}


# =============================================================================
# Stage 1: control variables
# =============================================================================

def sample_time_signature(rng: random.Random) -> str:
    return rng.choice(list(METER_CONFIGS.keys()))


def sample_leading_silence_sec(rng: random.Random) -> float:
    return rng.uniform(LEADING_SILENCE_MIN_SEC, LEADING_SILENCE_MAX_SEC)


def sample_tempo_and_silence_with_clip_limit(
    rng: random.Random,
    beats_per_bar: int,
) -> Tuple[int, str, float, float, float]:
    """Pick a tempo + leading silence such that the clip stays under MAX_CLIP_DURATION_SEC.

    Rejection-samples up to 1000 times (tempo/silence are cheap to redraw);
    falls back to the fastest tempo + minimum silence if that's still not enough,
    which only happens for pathologically slow tempo/meter combinations.
    """
    for _ in range(1000):
        tempo_bpm = rng.choice(TEMPO_OPTIONS)
        tempo_class = TEMPO_TO_CLASS[tempo_bpm]
        leading_silence_sec = sample_leading_silence_sec(rng)

        bar_duration_sec = beats_per_bar * (60.0 / tempo_bpm)
        clip_duration_sec = leading_silence_sec + 2 * bar_duration_sec

        if clip_duration_sec <= MAX_CLIP_DURATION_SEC:
            return tempo_bpm, tempo_class, leading_silence_sec, bar_duration_sec, clip_duration_sec

    tempo_bpm = max(TEMPO_OPTIONS)
    tempo_class = TEMPO_TO_CLASS[tempo_bpm]
    leading_silence_sec = LEADING_SILENCE_MIN_SEC
    bar_duration_sec = beats_per_bar * (60.0 / tempo_bpm)
    clip_duration_sec = leading_silence_sec + 2 * bar_duration_sec
    return tempo_bpm, tempo_class, leading_silence_sec, bar_duration_sec, min(clip_duration_sec, MAX_CLIP_DURATION_SEC)


def allowed_rhythm_families_for_meter(time_signature: str) -> List[str]:
    if time_signature == "6/8":
        return ["compound_flow", "mixed_short_long", "bar_repeat_rhythm",
                 "bar_contrast_rhythm", "pickup_like", "light_syncopation"]
    return ["uniform_quarters", "uniform_eighths", "mixed_short_long", "long_short_alternation",
             "light_syncopation", "pickup_like", "bar_repeat_rhythm", "bar_contrast_rhythm"]


def default_subdivision_for_family(time_signature: str, family: str) -> str:
    if time_signature == "6/8":
        return "compound_eighth"
    mapping = {
        "uniform_quarters": "quarter", "uniform_eighths": "eighth", "mixed_short_long": "eighth",
        "long_short_alternation": "eighth", "light_syncopation": "eighth", "pickup_like": "eighth",
        "bar_repeat_rhythm": "eighth", "bar_contrast_rhythm": "eighth",
    }
    return mapping.get(family, "eighth")


def sample_density_note_count(rng: random.Random, density: str, time_signature: str) -> int:
    if time_signature == "6/8":
        choices = {"sparse": [5, 6], "medium": [7, 8, 9], "dense": [9, 10, 11]}
    else:
        choices = {"sparse": [5, 6], "medium": [7, 8, 9], "dense": [10, 11, 12]}
    return rng.choice(choices[density])


def infer_density_level(num_notes: int, time_signature: str) -> str:
    """Used for the structured modes, where note count is set by a fixed template
    rather than sampled from `sample_density_note_count` - so density_level has to
    be inferred after the fact instead of being the thing that drove note count."""
    if num_notes <= 6:
        return "sparse"
    if num_notes <= 9:
        return "medium"
    return "dense"


def sample_controls(rng: random.Random) -> Tuple[BasicInfo, ControlAttributes]:
    """Sample every control variable for one excerpt.

    structure_mode is the main branch point: "normal" excerpts get every
    control sampled independently, while the three "explicit_*" modes each
    lock a few controls to fixed values so that concept is unambiguous
    (e.g. explicit_meter forces a strong, template-driven downbeat).
    """
    ts = sample_time_signature(rng)
    meter_cfg = METER_CONFIGS[ts]
    beats_per_bar = meter_cfg["beats_per_bar"]
    beat_unit = meter_cfg["beat_unit"]
    meter_class = meter_cfg["meter_class"]

    tempo_bpm, tempo_class, leading_silence_sec, bar_duration_sec, clip_duration_sec = (
        sample_tempo_and_silence_with_clip_limit(rng, beats_per_bar)
    )

    density_level = rng.choices(["sparse", "medium", "dense"], weights=[1, 5, 4], k=1)[0]
    structure_mode = rng.choices(
        ["normal", "explicit_meter", "explicit_repetition", "explicit_harmonic_pattern"],
        weights=[25, 25, 25, 25], k=1,
    )[0]

    meter_clarity = "normal"
    repetition_type = "none"
    repetition_transposition_semitones = 0
    accent_pattern_type = "default"
    harmonic_pattern_type = "none"

    if structure_mode == "explicit_meter":
        meter_clarity = "explicit"
        accent_pattern_type = "explicit_meter"
    elif structure_mode == "explicit_repetition":
        repetition_type = rng.choice(["exact_bar_repeat", "transposed_bar_repeat"])
        repetition_transposition_semitones = (
            0 if repetition_type == "exact_bar_repeat" else rng.choice(REPETITION_SHIFT_OPTIONS)
        )
    elif structure_mode == "explicit_harmonic_pattern":
        meter_clarity = "explicit"
        accent_pattern_type = "explicit_meter"
        harmonic_pattern_type = rng.choice(["ascending_arpeggio", "broken_chord_repeat"])
        repetition_type = rng.choice(["none", "exact_bar_repeat"])
        repetition_transposition_semitones = 0

    if structure_mode in {"explicit_repetition", "explicit_harmonic_pattern"} and density_level == "sparse":
        density_level = rng.choice(["medium", "dense"])

    rhythm_pattern_family = rng.choice(allowed_rhythm_families_for_meter(ts))
    syncopation_level = rng.choice(SYNCOPATION_LEVELS)
    register = rng.choice(REGISTERS)
    tonal_mode = rng.choice(TONAL_MODES)
    tonic_pc = rng.randint(0, 11)
    interval_profile = rng.choice(INTERVAL_PROFILES)
    interval_constraint_mode = rng.choice(["scale_constrained", "chord_tone_biased", "cadential_tonal"])

    if structure_mode in {"explicit_meter", "explicit_harmonic_pattern"}:
        syncopation_level = "none"
        if ts == "2/4":
            rhythm_pattern_family = rng.choice(["uniform_quarters", "uniform_eighths", "mixed_short_long"])
        elif ts == "3/4":
            rhythm_pattern_family = rng.choice(["uniform_quarters", "mixed_short_long", "pickup_like"])
        elif ts == "4/4":
            rhythm_pattern_family = rng.choice(["uniform_quarters", "uniform_eighths", "mixed_short_long", "bar_repeat_rhythm"])
        elif ts == "6/8":
            rhythm_pattern_family = rng.choice(["compound_flow", "mixed_short_long"])

    if tonal_mode == "major":
        bar1_chord_type = rng.choice(["I", "IV", "V", "vi"])
        bar2_chord_type = rng.choice(["I", "IV", "V", "vi"])
    else:
        bar1_chord_type = rng.choice(["i", "iv", "v", "VI"])
        bar2_chord_type = rng.choice(["i", "iv", "v", "VI"])

    velocity_pattern_type = rng.choice(VELOCITY_PATTERN_TYPES)
    if structure_mode in {"explicit_meter", "explicit_harmonic_pattern"}:
        velocity_pattern_type = "accent_downbeats"

    bar_relation_type = rng.choice(BAR_RELATION_TYPES)
    pitch_low_bound, pitch_high_bound = REGISTER_BOUNDS[register]

    generation_template_id = (
        f"{ts.replace('/', '_')}_{structure_mode}_{bar_relation_type}_{rhythm_pattern_family}_"
        f"{tonal_mode}_{interval_profile}_{syncopation_level}_{repetition_type}_{harmonic_pattern_type}_v12"
    )

    subdivision_type = default_subdivision_for_family(ts, rhythm_pattern_family)
    music_start_sec = leading_silence_sec
    music_end_sec = music_start_sec + 2 * bar_duration_sec
    clip_duration_sec = music_end_sec

    note_count = sample_density_note_count(rng, density_level, ts)
    basic_info = BasicInfo(
        num_bars=2, num_notes=note_count, time_signature=ts, beats_per_bar=beats_per_bar,
        beat_unit=beat_unit, meter_class=meter_class, tempo_bpm=tempo_bpm, tempo_class=tempo_class,
        bar_duration_sec=bar_duration_sec, clip_duration_sec=clip_duration_sec,
        leading_silence_sec=leading_silence_sec, music_start_sec=music_start_sec, music_end_sec=music_end_sec,
        subdivision_type=subdivision_type,
    )
    control = ControlAttributes(
        density_level=density_level, rhythm_pattern_family=rhythm_pattern_family,
        syncopation_level=syncopation_level, register=register, pitch_low_bound=pitch_low_bound,
        pitch_high_bound=pitch_high_bound, tonal_mode=tonal_mode, tonic_pc=tonic_pc,
        interval_profile=interval_profile, interval_constraint_mode=interval_constraint_mode,
        bar1_chord_type=bar1_chord_type, bar2_chord_type=bar2_chord_type, bar_relation_type=bar_relation_type,
        velocity_pattern_type=velocity_pattern_type, structure_mode=structure_mode, meter_clarity=meter_clarity,
        repetition_type=repetition_type, repetition_transposition_semitones=repetition_transposition_semitones,
        accent_pattern_type=accent_pattern_type, harmonic_pattern_type=harmonic_pattern_type,
        generation_template_id=generation_template_id,
    )
    return basic_info, control


# =============================================================================
# Stage 2: rhythm skeleton (onsets + durations, not yet pitched)
# =============================================================================

def allowed_duration_choices(ts: str, family: str) -> List[float]:
    if ts == "6/8":
        if family == "compound_flow":
            return [1.0, 1.0, 1.0, 2.0, 3.0]
        if family == "light_syncopation":
            return [1.0, 1.0, 2.0, 3.0]
        return [1.0, 1.0, 2.0, 2.0, 3.0]

    mapping = {
        "uniform_quarters": [1.0], "uniform_eighths": [0.5], "mixed_short_long": [0.5, 1.0, 1.5, 2.0],
        "long_short_alternation": [0.5, 1.5, 0.5, 1.5, 1.0], "light_syncopation": [0.5, 0.5, 1.0, 1.5],
        "pickup_like": [0.5, 0.5, 1.0, 1.0, 1.5], "bar_repeat_rhythm": [0.5, 1.0, 1.0, 1.5],
        "bar_contrast_rhythm": [0.5, 1.0, 1.5, 2.0],
    }
    return mapping.get(family, [0.5, 1.0, 1.5])


def allocate_note_counts_across_bars(rng: random.Random, total_notes: int) -> Tuple[int, int]:
    b1 = total_notes // 2
    b2 = total_notes - b1
    if total_notes >= 6 and rng.random() < 0.4:
        shift = rng.choice([-1, 1])
        if b1 + shift >= 2 and b2 - shift >= 2:
            b1 += shift
            b2 -= shift
    return b1, b2


def fit_bar_durations(rng: random.Random, note_count: int, bar_beats: int, choices: List[float], family: str) -> List[float]:
    """Greedily pick one duration per note so they sum to (at most) a full bar,
    always leaving enough room (>= 0.5 beats/note) for the notes still to place."""
    if note_count <= 0:
        return []
    result: List[float] = []
    remaining_beats = float(bar_beats)
    for i in range(note_count):
        remaining_notes = note_count - i
        min_needed_after = 0.5 * (remaining_notes - 1)
        feasible = [d for d in choices if d <= remaining_beats - min_needed_after + 1e-9]
        if not feasible:
            feasible = [0.5] if bar_beats >= 1 else [1.0]
        if family == "uniform_quarters":
            feasible = [d for d in feasible if math.isclose(d, 1.0)] or feasible
        elif family == "uniform_eighths":
            feasible = [d for d in feasible if math.isclose(d, 0.5)] or feasible
        elif family == "long_short_alternation":
            target = 1.5 if i % 2 == 0 else 0.5
            feasible = [d for d in feasible if math.isclose(d, target)] or feasible
        d = rng.choice(feasible)
        result.append(d)
        remaining_beats -= d
    return result


def build_onsets_from_durations(rng: random.Random, durs: List[float], bar_beats: int, family: str, syncopation_level: str) -> List[float]:
    """Distribute any leftover beat-time (bar_beats minus the sum of durations) as
    gaps before/between notes, so the bar is fully accounted for. "pickup_like"
    reserves the first gap; "strong" syncopation prefers pushing gaps onto weak
    positions instead of gap 0, so notes land off the beat."""
    if not durs:
        return []

    total_note_beats = sum(durs)
    remaining = max(0.0, bar_beats - total_note_beats)
    n_gaps = len(durs) + 1
    gaps = [0.0 for _ in range(n_gaps)]

    if family == "pickup_like" and remaining >= 0.5:
        gaps[0] = 0.5
        remaining -= 0.5

    if syncopation_level == "strong" and len(durs) >= 2 and remaining >= 0.5:
        num_pushes = min(2, int(remaining / 0.5))
        candidate_positions = list(range(1, len(durs)))
        rng.shuffle(candidate_positions)
        for pos in candidate_positions[:num_pushes]:
            gaps[pos] += 0.5
            remaining -= 0.5
            if remaining < 0.5:
                break

    max_gap = 0.5
    steps = int(round(remaining / 0.5))

    if syncopation_level == "strong":
        preferred = list(range(1, max(1, n_gaps - 1))) or list(range(n_gaps))
        for _ in range(steps):
            candidate_idxs = [i for i in preferred if gaps[i] < max_gap - 1e-9] \
                or [i for i in range(n_gaps) if gaps[i] < max_gap - 1e-9]
            if not candidate_idxs:
                break
            gaps[rng.choice(candidate_idxs)] += 0.5
    else:
        for _ in range(steps):
            candidate_idxs = [i for i in range(n_gaps) if gaps[i] < max_gap - 1e-9]
            if not candidate_idxs:
                break
            gaps[rng.choice(candidate_idxs)] += 0.5

    onsets = []
    t = gaps[0]
    for i, d in enumerate(durs):
        onsets.append(t)
        t += d + gaps[i + 1]
    return onsets


def generate_bar_rhythm(
    rng: random.Random, ts: str, family: str, syncopation_level: str, bar_beats: int,
    note_count: int, reference: Optional[Tuple[List[float], List[float]]] = None,
) -> Tuple[List[float], List[float]]:
    if reference is not None and family == "bar_repeat_rhythm":
        ref_onsets, ref_durs = reference
        if len(ref_durs) == note_count:
            return ref_onsets[:], ref_durs[:]
    choices = allowed_duration_choices(ts, family)
    durs = fit_bar_durations(rng, note_count, bar_beats, choices, family)
    onsets = build_onsets_from_durations(rng, durs, bar_beats, family, syncopation_level)
    return onsets, durs


def template_to_bar_notes(bar_index: int, durations: List[float]) -> List[Dict]:
    notes = []
    t = 0.0
    for d in durations:
        notes.append({"bar_index": bar_index, "within_bar_onset_beats": t, "duration_beats": float(d)})
        t += d
    return notes


def generate_normal_rhythm_skeleton(rng: random.Random, basic: BasicInfo, control: ControlAttributes) -> List[Dict]:
    b1_count, b2_count = allocate_note_counts_across_bars(rng, basic.num_notes)
    bar_beats = basic.beats_per_bar

    on1, dur1 = generate_bar_rhythm(rng, basic.time_signature, control.rhythm_pattern_family,
                                     control.syncopation_level, bar_beats, b1_count, reference=None)

    # bar_relation_type can override which rhythm family bar 2 uses, so the
    # rendered rhythm actually matches the bar-comparison label.
    second_family = control.rhythm_pattern_family
    if control.bar_relation_type == "contrast" and second_family == "bar_repeat_rhythm":
        second_family = "bar_contrast_rhythm"
    if control.bar_relation_type == "repeat_with_variation" and second_family == "bar_contrast_rhythm":
        second_family = "bar_repeat_rhythm"

    ref = (on1, dur1) if second_family == "bar_repeat_rhythm" else None
    on2, dur2 = generate_bar_rhythm(rng, basic.time_signature, second_family,
                                     control.syncopation_level, bar_beats, b2_count, reference=ref)

    notes = [{"bar_index": 1, "within_bar_onset_beats": on, "duration_beats": dur} for on, dur in zip(on1, dur1)]
    notes += [{"bar_index": 2, "within_bar_onset_beats": on, "duration_beats": dur} for on, dur in zip(on2, dur2)]
    notes.sort(key=lambda x: (x["bar_index"], x["within_bar_onset_beats"]))
    return notes


def generate_explicit_meter_rhythm_skeleton(rng: random.Random, basic: BasicInfo, control: ControlAttributes) -> List[Dict]:
    ts = basic.time_signature
    bar1_notes = template_to_bar_notes(1, rng.choice(EXPLICIT_METER_BAR_TEMPLATES[ts]))
    bar2_notes = template_to_bar_notes(2, rng.choice(EXPLICIT_METER_BAR_TEMPLATES[ts]))
    notes = bar1_notes + bar2_notes
    notes.sort(key=lambda x: (x["bar_index"], x["within_bar_onset_beats"]))
    return notes


def generate_explicit_repetition_rhythm_skeleton(rng: random.Random, basic: BasicInfo, control: ControlAttributes) -> List[Dict]:
    """Bar 2 always reuses bar 1's exact rhythm; only pitch (see realize_pitch_sequence)
    decides whether the repetition is exact or transposed."""
    bar_beats = basic.beats_per_bar
    b1_count, _ = allocate_note_counts_across_bars(rng, basic.num_notes)

    on1, dur1 = generate_bar_rhythm(rng, basic.time_signature, control.rhythm_pattern_family,
                                     "none", bar_beats, b1_count, reference=None)
    bar1_notes = [{"bar_index": 1, "within_bar_onset_beats": on, "duration_beats": dur} for on, dur in zip(on1, dur1)]
    bar2_notes = [{"bar_index": 2, "within_bar_onset_beats": float(n["within_bar_onset_beats"]),
                   "duration_beats": float(n["duration_beats"])} for n in bar1_notes]

    notes = bar1_notes + bar2_notes
    notes.sort(key=lambda x: (x["bar_index"], x["within_bar_onset_beats"]))
    return notes


def generate_explicit_harmonic_rhythm_skeleton(rng: random.Random, basic: BasicInfo, control: ControlAttributes) -> List[Dict]:
    ts = basic.time_signature
    bar1_template = rng.choice(HARMONIC_BAR_TEMPLATES[ts])
    bar2_template = bar1_template[:] if control.repetition_type == "exact_bar_repeat" else rng.choice(HARMONIC_BAR_TEMPLATES[ts])

    notes = template_to_bar_notes(1, bar1_template) + template_to_bar_notes(2, bar2_template)
    notes.sort(key=lambda x: (x["bar_index"], x["within_bar_onset_beats"]))
    return notes


def generate_rhythm_skeleton(rng: random.Random, basic: BasicInfo, control: ControlAttributes) -> List[Dict]:
    if control.structure_mode == "explicit_meter":
        return generate_explicit_meter_rhythm_skeleton(rng, basic, control)
    if control.structure_mode == "explicit_repetition":
        return generate_explicit_repetition_rhythm_skeleton(rng, basic, control)
    if control.structure_mode == "explicit_harmonic_pattern":
        return generate_explicit_harmonic_rhythm_skeleton(rng, basic, control)
    return generate_normal_rhythm_skeleton(rng, basic, control)


# =============================================================================
# Stage 3: pitch
# =============================================================================

def propose_interval(rng: random.Random, interval_profile: str) -> int:
    """Draw a *desired* next-note interval (in semitones) from a profile-specific
    distribution. This is only ever a soft target - realize_pitch_sequence scores
    candidate pitches by how close they land to it, it doesn't apply it directly."""
    if interval_profile == "conjunct":
        choices, weights = [-3, -2, -1, 0, 1, 2, 3], [1, 2, 4, 1, 4, 2, 1]
    elif interval_profile == "balanced":
        choices, weights = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], [1, 1, 2, 2, 3, 1, 3, 2, 2, 1, 1]
    else:  # disjunct
        choices, weights = [-8, -7, -5, -4, -3, -2, 0, 2, 3, 4, 5, 7, 8], [1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1]
    return rng.choices(choices, weights=weights, k=1)[0]


def build_chord_tone_pool(chord_pcs: List[int], low: int, high: int, center_pitch: int) -> List[int]:
    pool = []
    for pc in chord_pcs:
        p = nearest_pitch_with_pc(center_pitch, pc, low, high)
        if p is not None:
            pool.append(p)
    pool = sorted(set(pool))
    if not pool:
        pool = [p for p in range(low, high + 1) if p % 12 in set(chord_pcs)]
    return pool


def expand_arpeggio_pattern(pattern_type: str, tone_pool: List[int], note_count: int) -> List[int]:
    """Repeat a 3-4 note root/third/fifth cell to fill note_count notes."""
    if not tone_pool:
        return []
    tones = sorted(tone_pool)
    root, third, fifth = tones[0], tones[min(1, len(tones) - 1)], tones[min(2, len(tones) - 1)]
    if pattern_type == "broken_chord_repeat":
        base = [root, third, fifth, third]
    else:  # ascending_arpeggio and any other/default pattern
        base = [root, third, fifth]
    return [base[i % len(base)] for i in range(note_count)]


def realize_harmonic_pattern_pitches(rng: random.Random, basic: BasicInfo, control: ControlAttributes, notes: List[Dict]) -> List[int]:
    del rng  # deterministic given the chord/pattern controls; no randomness needed here
    low, high = control.pitch_low_bound, control.pitch_high_bound
    center_pitch = (low + high) // 2

    bar1_notes = [n for n in notes if n["bar_index"] == 1]
    bar2_notes = [n for n in notes if n["bar_index"] == 2]

    pool1 = build_chord_tone_pool(chord_pitch_classes(control.tonal_mode, control.tonic_pc, control.bar1_chord_type), low, high, center_pitch)
    pool2 = build_chord_tone_pool(chord_pitch_classes(control.tonal_mode, control.tonic_pc, control.bar2_chord_type), low, high, center_pitch)

    bar1_seq = expand_arpeggio_pattern(control.harmonic_pattern_type, pool1, len(bar1_notes))

    if control.repetition_type == "exact_bar_repeat":
        base2 = bar1_seq[:]
        if len(base2) < len(bar2_notes) and base2:
            base2 += [base2[i % len(base2)] for i in range(len(bar2_notes) - len(base2))]
        bar2_seq = base2[:len(bar2_notes)]
    else:
        bar2_seq = expand_arpeggio_pattern(control.harmonic_pattern_type, pool2, len(bar2_notes))

    return bar1_seq + bar2_seq


def realize_pitch_sequence(rng: random.Random, basic: BasicInfo, control: ControlAttributes, notes: List[Dict]) -> List[int]:
    """Assign a pitch to every onset in `notes`.

    For explicit_harmonic_pattern this just expands an arpeggio (see above).
    Otherwise each note is picked one at a time by scoring every in-scale
    candidate pitch on: how close it stays to a wandering melodic center,
    how close the resulting interval is to `propose_interval`'s target,
    whether it's a chord tone (weighted more on strong beats, for the
    chord-tone-biased/cadential constraint modes), and a penalty for very
    large leaps. The top ~6 candidates are then sampled from (not just the
    single best), which is what keeps two runs with different notes-but-same-
    seed structurally similar rather than identical.

    Bar 2 is then adjusted in place according to bar_relation_type /
    repetition_type, since those are properties of how bar 2 relates to
    bar 1, not something a fresh per-note score could capture on its own.
    """
    note_count = len(notes)
    if note_count <= 0:
        return []

    if control.structure_mode == "explicit_harmonic_pattern":
        return realize_harmonic_pattern_pitches(rng, basic, control, notes)

    low, high = control.pitch_low_bound, control.pitch_high_bound
    span = rng.randint(6, min(12, high - low))

    allowed_scale_pcs = scale_pitch_classes(control.tonal_mode, control.tonic_pc)
    allowed_scale_pitches = allowed_pitches_in_range(low, high, allowed_scale_pcs) or list(range(low, high + 1))

    center = rng.choice(allowed_scale_pitches)
    min_target = clamp(center - span // 2, low, high)
    max_target = clamp(center + span // 2, low, high)
    mid_target = 0.5 * (min_target + max_target)

    pitches: List[int] = []
    prev: Optional[int] = None
    prev_interval: Optional[int] = None

    for note in notes:
        chord_type = control.bar1_chord_type if note["bar_index"] == 1 else control.bar2_chord_type
        chord_pcs = chord_pitch_classes(control.tonal_mode, control.tonic_pc, chord_type)
        beat_idx = int(math.floor(note["within_bar_onset_beats"])) + 1
        is_strong = beat_idx in STRONG_BEATS[basic.time_signature]

        scored: List[Tuple[float, int]] = []
        for cand in allowed_scale_pitches:
            score = -abs(cand - mid_target) * 0.8

            if prev is not None:
                interval = cand - prev
                desired = propose_interval(rng, control.interval_profile)
                score -= abs(interval - desired) * 0.6
                if prev_interval is not None and abs(prev_interval) >= 5:
                    if interval != 0 and (interval > 0) != (prev_interval > 0):
                        score += 2.0  # reward bouncing back after a big leap
                    if abs(interval) <= 2:
                        score += 1.5
                if abs(interval) >= 12:
                    score -= 4.0
                elif abs(interval) >= 8:
                    score -= 2.0

            if cand % 12 in allowed_scale_pcs:
                score += 1.5
            else:
                score -= 8.0  # should be unreachable: allowed_scale_pitches is already scale-filtered

            if control.interval_constraint_mode in {"chord_tone_biased", "cadential_tonal"}:
                if cand % 12 in chord_pcs:
                    score += 1.2
                if is_strong:
                    score += 2.5 if cand % 12 in chord_pcs else -1.5

            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = rng.choice(scored[: min(6, len(scored))])[1]
        if prev is not None:
            prev_interval = chosen - prev
        pitches.append(chosen)
        prev = chosen

    _apply_bar_relation(rng, control, notes, pitches, allowed_scale_pitches, low, high)
    return pitches


def _apply_bar_relation(
    rng: random.Random, control: ControlAttributes, notes: List[Dict], pitches: List[int],
    allowed_scale_pitches: List[int], low: int, high: int,
) -> None:
    """Rewrite bar 2's pitches in place so they honor repetition_type /
    bar_relation_type, then return (mutates `pitches`)."""
    b1_idx = [i for i, n in enumerate(notes) if n["bar_index"] == 1 and i < len(pitches)]
    b2_idx = [i for i, n in enumerate(notes) if n["bar_index"] == 2 and i < len(pitches)]
    if not (b1_idx and b2_idx):
        return

    bar1 = [pitches[i] for i in b1_idx]
    allowed_pcs_set = set(allowed_scale_pitches[i] % 12 for i in range(len(allowed_scale_pitches)))

    if control.repetition_type == "exact_bar_repeat":
        for j, idx in enumerate(b2_idx):
            pitches[idx] = bar1[min(j, len(bar1) - 1)]
        return

    if control.repetition_type == "transposed_bar_repeat":
        shift = control.repetition_transposition_semitones
        shifted, valid = [], True
        for j in range(len(b2_idx)):
            ref = bar1[min(j, len(bar1) - 1)]
            p = ref + shift
            if not (low <= p <= high):
                valid = False
                break
            if control.interval_constraint_mode != "free_chromatic" and p % 12 not in allowed_pcs_set:
                valid = False
                break
            shifted.append(p)
        chosen = shifted if valid else [bar1[min(j, len(bar1) - 1)] for j in range(len(b2_idx))]
        for j, idx in enumerate(b2_idx):
            pitches[idx] = chosen[j]
        return

    if control.bar_relation_type == "repeat_with_variation":
        for j, idx in enumerate(b2_idx):
            ref = bar1[min(j, len(bar1) - 1)]
            cands = [c for c in (clamp(ref + d, low, high) for d in [-2, -1, 0, 1, 2]) if c % 12 in allowed_pcs_set]
            if cands:
                pitches[idx] = rng.choice(cands)
    elif control.bar_relation_type == "sequence_like":
        shift = rng.choice([-2, -1, 1, 2])
        for j, idx in enumerate(b2_idx):
            ref = bar1[min(j, len(bar1) - 1)]
            cand = clamp(ref + shift, low, high)
            if cand % 12 in allowed_pcs_set:
                pitches[idx] = cand
    elif control.bar_relation_type == "call_response":
        for idx in b2_idx:
            local = sorted(allowed_scale_pitches, key=lambda p: (abs((p - pitches[idx]) + 2), abs(p - pitches[idx])))
            if local:
                pitches[idx] = local[0]
    elif control.bar_relation_type == "cadential_second_bar":
        tonic_pc = control.tonic_pc % 12
        tonic_candidates = [p for p in allowed_scale_pitches if p % 12 == tonic_pc]
        end_target = tonic_candidates[len(tonic_candidates) // 2] if tonic_candidates else pitches[-1]
        for k, idx in enumerate(b2_idx):
            frac = (k + 1) / len(b2_idx)
            cand = int(round((1 - frac) * pitches[idx] + frac * end_target))
            nearby = sorted(allowed_scale_pitches, key=lambda p: abs(p - cand))[:4]
            if nearby:
                pitches[idx] = rng.choice(nearby)
    elif control.bar_relation_type == "contrast":
        for idx in b2_idx:
            nearby = sorted(allowed_scale_pitches, key=lambda p: -abs(p - pitches[idx]))
            if nearby:
                pitches[idx] = nearby[0]


# =============================================================================
# Stage 4: velocity
# =============================================================================

def generate_velocities(rng: random.Random, notes: List[Dict], control: ControlAttributes, basic: BasicInfo) -> List[int]:
    n = len(notes)
    if n == 0:
        return []
    base = rng.randint(72, 88)
    vals = []

    for i, note in enumerate(notes):
        if control.velocity_pattern_type == "flat":
            v = base
        elif control.velocity_pattern_type == "random_mild":
            v = base + rng.randint(-6, 6)
        elif control.velocity_pattern_type == "crescendo":
            v = base + int(round(18 * i / max(1, n - 1)))
        elif control.velocity_pattern_type == "decrescendo":
            v = base + int(round(18 * (n - 1 - i) / max(1, n - 1)))
        elif control.velocity_pattern_type == "accent_downbeats":
            beat_idx = int(math.floor(note["within_bar_onset_beats"])) + 1
            if control.accent_pattern_type == "explicit_meter":
                v = _explicit_meter_accent(basic.time_signature, beat_idx, base)
            else:
                strong = beat_idx in STRONG_BEATS[basic.time_signature]
                v = base + (10 if strong else -2)
        else:
            v = base
        vals.append(clamp(v, 40, 110))

    return vals


def _explicit_meter_accent(time_signature: str, beat_idx: int, base: int) -> int:
    if time_signature == "2/4":
        return base + (16 if beat_idx == 1 else -5)
    if time_signature == "3/4":
        return base + (16 if beat_idx == 1 else -4)
    if time_signature == "4/4":
        if beat_idx == 1:
            return base + 16
        if beat_idx == 3:
            return base + 9
        return base - 4
    if time_signature == "6/8":
        if beat_idx == 1:
            return base + 16
        if beat_idx == 4:
            return base + 9
        return base - 4
    return base


# =============================================================================
# Assembly
# =============================================================================

def assemble_events(notes: List[Dict], pitches: List[int], velocities: List[int], basic: BasicInfo) -> List[Dict]:
    """Convert beat-based (bar_index, within_bar_onset_beats, duration_beats) notes
    into second-based note events, and attach the beat/downbeat metadata that the
    QA-fact builder later reads (is_downbeat, is_strong_beat, ...)."""
    events: List[Dict] = []
    sec_per_beat = basic.bar_duration_sec / basic.beats_per_bar

    for i, (note, pitch, vel) in enumerate(zip(notes, pitches, velocities)):
        bar_index = note["bar_index"]
        onset_beats_global = (bar_index - 1) * basic.beats_per_bar + note["within_bar_onset_beats"]

        onset_sec = basic.leading_silence_sec + onset_beats_global * sec_per_beat
        offset_sec = basic.leading_silence_sec + (onset_beats_global + note["duration_beats"]) * sec_per_beat
        offset_sec = min(offset_sec, basic.clip_duration_sec)

        beat_index_in_bar = max(1, min(int(math.floor(note["within_bar_onset_beats"])) + 1, basic.beats_per_bar))
        is_downbeat = math.isclose(note["within_bar_onset_beats"], 0.0, abs_tol=1e-9)
        is_strong = beat_index_in_bar in STRONG_BEATS[basic.time_signature]

        events.append({
            "note_index": i,
            "pitch": int(pitch),
            "onset_sec": round(onset_sec, 6),
            "offset_sec": round(offset_sec, 6),
            "duration_beats": float(note["duration_beats"]),
            "duration_symbol": duration_symbol(float(note["duration_beats"])),
            "velocity": int(vel),
            "bar_index": int(bar_index),
            "beat_index_in_bar": int(beat_index_in_bar),
            "within_bar_onset_beats": float(note["within_bar_onset_beats"]),
            "is_downbeat": bool(is_downbeat),
            "is_strong_beat": bool(is_strong),
            "interval_from_prev": None,
            "interval_class_from_prev": None,
            "direction_from_prev": None,
        })

    return events


def build_concepts(basic: BasicInfo, control: ControlAttributes) -> List[str]:
    """Flatten basic/control into `"field:value"` tags. These tags are what
    render_grounding_qa / render_understanding_qa group and sample by, and
    what dataset_stats.py counts to report the label distribution."""
    concepts = [
        f"meter:{basic.time_signature}", f"meter_class:{basic.meter_class}",
        f"tempo_bpm:{basic.tempo_bpm}", f"tempo_class:{basic.tempo_class}",
        f"density:{control.density_level}", f"rhythm:{control.rhythm_pattern_family}",
        f"syncopation:{control.syncopation_level}", f"register:{control.register}",
        f"tonal_mode:{control.tonal_mode}", f"tonic_pc:{control.tonic_pc}",
        f"interval_profile:{control.interval_profile}", f"interval_constraint_mode:{control.interval_constraint_mode}",
        f"bar1_chord:{control.bar1_chord_type}", f"bar2_chord:{control.bar2_chord_type}",
        f"bar_relation:{control.bar_relation_type}", f"velocity:{control.velocity_pattern_type}",
        f"structure_mode:{control.structure_mode}", f"meter_clarity:{control.meter_clarity}",
        f"repetition_type:{control.repetition_type}", f"accent_pattern_type:{control.accent_pattern_type}",
        f"harmonic_pattern_type:{control.harmonic_pattern_type}",
    ]
    if control.repetition_type == "transposed_bar_repeat":
        concepts.append(f"repetition_shift:{control.repetition_transposition_semitones}")
    return concepts


def generate_one_sample(rng: random.Random) -> Dict[str, Any]:
    """Run all four stages and return one raw (not yet humanized/validated) sample."""
    basic, control = sample_controls(rng)
    notes = generate_rhythm_skeleton(rng, basic, control)
    basic.num_notes = len(notes)

    if control.structure_mode in {"explicit_meter", "explicit_repetition", "explicit_harmonic_pattern"}:
        control.density_level = infer_density_level(len(notes), basic.time_signature)

    pitches = realize_pitch_sequence(rng, basic, control, notes)
    velocities = generate_velocities(rng, notes, control, basic)
    events = assemble_events(notes, pitches, velocities, basic)
    return {"basic_info": basic.__dict__, "control_attributes": control.__dict__, "events": events}
