"""Small, self-contained music-theory helpers (scales, chords, interval naming)."""

from typing import List, Optional

from src.config import CHORD_DEGREES, MAJOR_SCALE_PCS, NATURAL_MINOR_SCALE_PCS


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def interval_class(interval: Optional[int]) -> Optional[str]:
    """Bucket a signed semitone interval into the coarse category used in QA text."""
    if interval is None:
        return None
    a = abs(interval)
    if a == 0:
        return "repeat"
    if a in {1, 2}:
        return "step"
    if a in {3, 4, 5}:
        return "skip"
    return "leap"


def interval_direction(interval: Optional[int]) -> Optional[str]:
    if interval is None:
        return None
    if interval > 0:
        return "up"
    if interval < 0:
        return "down"
    return "same"


def duration_symbol(duration_beats: float) -> str:
    rounded = round(duration_beats, 3)
    if abs(rounded - 0.5) < 1e-9:
        return "eighth"
    if abs(rounded - 1.0) < 1e-9:
        return "quarter"
    if abs(rounded - 1.5) < 1e-9:
        return "dotted_quarter"
    if abs(rounded - 2.0) < 1e-9:
        return "half"
    if rounded < 1.0:
        return "short"
    if rounded < 2.0:
        return "medium"
    return "long"


def scale_pitch_classes(tonal_mode: str, tonic_pc: int) -> List[int]:
    tonic_pc %= 12
    if tonal_mode == "major":
        return sorted({(tonic_pc + x) % 12 for x in MAJOR_SCALE_PCS})
    if tonal_mode == "natural_minor":
        return sorted({(tonic_pc + x) % 12 for x in NATURAL_MINOR_SCALE_PCS})
    if tonal_mode == "chromatic":
        return list(range(12))
    raise ValueError(f"Unknown tonal_mode: {tonal_mode}")


def chord_pitch_classes(tonal_mode: str, tonic_pc: int, chord_type: str) -> List[int]:
    tonic_pc %= 12
    if tonal_mode == "major":
        degrees = CHORD_DEGREES["major"].get(chord_type, CHORD_DEGREES["major"]["I"])
        return sorted({(tonic_pc + MAJOR_SCALE_PCS[d]) % 12 for d in degrees})
    if tonal_mode == "natural_minor":
        degrees = CHORD_DEGREES["natural_minor"].get(chord_type, CHORD_DEGREES["natural_minor"]["i"])
        return sorted({(tonic_pc + NATURAL_MINOR_SCALE_PCS[d]) % 12 for d in degrees})
    return list(range(12))


def allowed_pitches_in_range(low: int, high: int, allowed_pcs: List[int]) -> List[int]:
    pcs = set(pc % 12 for pc in allowed_pcs)
    return [p for p in range(low, high + 1) if p % 12 in pcs]


def nearest_pitch_with_pc(center_pitch: int, pitch_class: int, low: int, high: int) -> Optional[int]:
    candidates = [p for p in range(low, high + 1) if p % 12 == (pitch_class % 12)]
    if not candidates:
        return None
    return min(candidates, key=lambda p: abs(p - center_pitch))
