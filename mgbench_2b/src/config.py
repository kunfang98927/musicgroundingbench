"""Shared constants and data schemas for the two-bar generation pipeline.

Every control-variable option list, humanization range, and music-theory
table used by the generator lives here so the rest of the pipeline has a
single source of truth for "what values can this field take".
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

# Master seed for MIDI generation and the train/val/test split (Section 1
# of the pipeline). Override with --seed on 01_generate_midi.py.
RNG_SEED = 1024

PITCH_MIN = 21
PITCH_MAX = 108
MIN_NOTE_DUR_SEC = 0.06
VELOCITY_MIN = 40
VELOCITY_MAX = 110
PROGRAM = 0  # General MIDI program number (0 = acoustic grand piano)
SANITY_MAX_RETRIES_PER_SAMPLE = 200
EPS = 1e-9

# --- Humanization (timing/velocity jitter applied after the rhythm is laid out) ---
TIMING_HUMANIZATION_LEVELS = ["light", "medium", "strong"]
TIMING_HUMANIZATION_WEIGHTS = [3, 4, 3]

ONSET_JITTER_MAX_BY_LEVEL = {"light": 0.015, "medium": 0.030, "strong": 0.045}
OFFSET_JITTER_MAX_BY_LEVEL = {"light": 0.020, "medium": 0.040, "strong": 0.060}
BAR_DRIFT_MAX_BY_LEVEL = {"light": 0.008, "medium": 0.015, "strong": 0.025}
GLOBAL_DRIFT_MAX_BY_LEVEL = {"light": 0.006, "medium": 0.012, "strong": 0.020}
VELOCITY_JITTER_MAX_BY_LEVEL = {"light": 3, "medium": 6, "strong": 9}

# Samples with an explicit structural label (meter/repetition/harmonic
# pattern) get less humanization, so the labeled structure stays audible.
STRUCTURED_HUMANIZATION_SCALE = 0.65

LEADING_SILENCE_MIN_SEC = 0.15
LEADING_SILENCE_MAX_SEC = 0.90
MAX_CLIP_DURATION_SEC = 8.0

# --- Meter ---
METER_CONFIGS = {
    "2/4": {"beats_per_bar": 2, "beat_unit": 4, "meter_class": "simple_duple",
             "allowed_subdivisions": ["quarter", "eighth", "sixteenth"]},
    "3/4": {"beats_per_bar": 3, "beat_unit": 4, "meter_class": "simple_triple",
             "allowed_subdivisions": ["quarter", "eighth", "sixteenth"]},
    "4/4": {"beats_per_bar": 4, "beat_unit": 4, "meter_class": "simple_quadruple",
             "allowed_subdivisions": ["quarter", "eighth", "sixteenth"]},
    "6/8": {"beats_per_bar": 6, "beat_unit": 8, "meter_class": "compound_duple",
             "allowed_subdivisions": ["compound_eighth"]},
}

# --- Tempo ---
TEMPO_TO_CLASS = {72: "slow", 90: "medium", 108: "medium", 120: "medium", 144: "fast"}
TEMPO_OPTIONS = list(TEMPO_TO_CLASS.keys())

# --- Rhythm / density / register / tonality option lists ---
RHYTHM_PATTERN_FAMILIES = [
    "uniform_quarters", "uniform_eighths", "mixed_short_long", "long_short_alternation",
    "light_syncopation", "pickup_like", "compound_flow", "bar_repeat_rhythm", "bar_contrast_rhythm",
]
VELOCITY_PATTERN_TYPES = ["flat", "random_mild", "crescendo", "decrescendo", "accent_downbeats"]
DENSITY_LEVELS = ["sparse", "medium", "dense"]
SYNCOPATION_LEVELS = ["none", "strong"]
REGISTERS = ["low", "mid", "high"]
TONAL_MODES = ["major", "natural_minor"]
INTERVAL_PROFILES = ["conjunct", "balanced", "disjunct"]
INTERVAL_CONSTRAINT_MODES = ["free_chromatic", "scale_constrained", "chord_tone_biased", "cadential_tonal"]

# --- Two-bar structure (the "concepts" the benchmark queries about) ---
BAR_RELATION_TYPES = ["repeat_with_variation", "contrast", "sequence_like", "call_response", "cadential_second_bar"]
STRUCTURE_MODES = ["normal", "explicit_meter", "explicit_repetition", "explicit_harmonic_pattern"]
METER_CLARITY_LEVELS = ["normal", "explicit"]
REPETITION_TYPES = ["none", "exact_bar_repeat", "transposed_bar_repeat"]
REPETITION_SHIFT_OPTIONS = [-7, -5, -2, 2, 5, 7]
ACCENT_PATTERN_TYPES = ["default", "explicit_meter"]
HARMONIC_PATTERN_TYPES = ["none", "ascending_arpeggio", "descending_arpeggio", "oscillating_arpeggio", "broken_chord_repeat"]

REGISTER_BOUNDS = {"low": (40, 57), "mid": (55, 72), "high": (67, 84)}
STRONG_BEATS = {"2/4": {1}, "3/4": {1}, "4/4": {1, 3}, "6/8": {1, 4}}

MAJOR_SCALE_PCS = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR_SCALE_PCS = [0, 2, 3, 5, 7, 8, 10]

# Roman-numeral chord -> scale degrees (0-indexed) used to build each chord's pitch classes.
CHORD_DEGREES = {
    "major": {"I": [0, 2, 4], "IV": [3, 5, 0], "V": [4, 6, 1], "vi": [5, 0, 2]},
    "natural_minor": {"i": [0, 2, 4], "iv": [3, 5, 0], "v": [4, 6, 1], "VI": [5, 0, 2]},
}


@dataclass
class BasicInfo:
    num_bars: int
    num_notes: int
    time_signature: str
    beats_per_bar: int
    beat_unit: int
    meter_class: str
    tempo_bpm: int
    tempo_class: str
    bar_duration_sec: float
    clip_duration_sec: float
    leading_silence_sec: float
    music_start_sec: float
    music_end_sec: float
    subdivision_type: str


@dataclass
class ControlAttributes:
    """The control variables sampled up front; everything else is derived from these."""
    density_level: str
    rhythm_pattern_family: str
    syncopation_level: str
    register: str
    pitch_low_bound: int
    pitch_high_bound: int
    tonal_mode: str
    tonic_pc: int
    interval_profile: str
    interval_constraint_mode: str
    bar1_chord_type: str
    bar2_chord_type: str
    bar_relation_type: str
    velocity_pattern_type: str
    structure_mode: str
    meter_clarity: str
    repetition_type: str
    repetition_transposition_semitones: int
    accent_pattern_type: str
    harmonic_pattern_type: str
    generation_template_id: str


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SanityStats:
    """Accumulates accept/reject counts and distributions across a whole generation run."""
    attempts: int = 0
    accepted: int = 0
    rejected: int = 0
    reject_reasons: Counter = field(default_factory=Counter)
    warning_reasons: Counter = field(default_factory=Counter)
    accepted_control: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    accepted_derived_values: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
