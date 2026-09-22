"""MGBench-v2 oracle for the 3-note (3N) subset: solver on the realised notes of a clip (start, end, pitch, velocity).

Returns None when a question type is not answerable / not unambiguous for the clip (the builder then skips it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

VEL_MARGIN = 24        # loudest / softest / second-loudest need a velocity margin of at least this to every other note (audit: 16 is only ~90 % consistent with measured loudness)
GAP_MARGIN_SEC = 0.15  # shortest silent gap must beat the other gap by this much
INT_MARGIN = 2         # semitone margin for "widen" / "narrow" of successive pitch intervals


@dataclass(frozen=True)
class N:
    start: float
    end: float
    pitch: int
    vel: int


class Clip:
    def __init__(self, notes: Sequence[N]):
        self.notes: List[N] = sorted(notes, key=lambda x: (x.start, x.end))
        self.n = len(self.notes)
        self.p = [x.pitch for x in self.notes]
        self.v = [x.vel for x in self.notes]

    def spans(self, idx):
        return [(self.notes[i].start, self.notes[i].end) for i in sorted(idx)]

    # ---- absolute pitch
    def with_pitches(self, vals: Sequence[int]) -> List[int]:
        return [i for i in range(self.n) if self.p[i] in set(vals)]

    # ---- order
    def ordinal(self, k: int) -> List[int]:
        return [k - 1] if 1 <= k <= self.n else []

    def last(self) -> List[int]:
        return [self.n - 1]

    def prefix(self, m: int) -> List[int]:
        return list(range(m)) if self.n >= m else []

    def suffix(self, m: int) -> List[int]:
        return list(range(self.n - m, self.n)) if self.n >= m else []

    # ---- relative pitch (ties included; rank over distinct pitches)
    def pitch_rank(self, rank: int, highest: bool) -> List[int]:
        ps = sorted(set(self.p), reverse=highest)
        return [i for i in range(self.n) if self.p[i] == ps[rank - 1]] if rank <= len(ps) else []

    # ---- dynamics (margin rule)
    def loudest(self) -> Optional[List[int]]:
        return self._extreme(max)

    def softest(self) -> Optional[List[int]]:
        return self._extreme(min)

    def _extreme(self, f) -> Optional[List[int]]:
        if self.n < 2:
            return None
        t = f(self.v)
        i = self.v.index(t)
        others = [x for j, x in enumerate(self.v) if j != i]
        ok = all(abs(t - x) >= VEL_MARGIN for x in others)
        return [i] if ok and self.v.count(t) == 1 else None

    def second_loudest(self) -> Optional[List[int]]:
        if self.n != 3:
            return None
        order = sorted(range(3), key=lambda i: -self.v[i])
        a, b, c = (self.v[i] for i in order)
        return [order[1]] if a - b >= VEL_MARGIN and b - c >= VEL_MARGIN else None

    # ---- pairs (3-note clips)
    def closest_pitch_pair(self) -> Optional[List[int]]:
        if self.n != 3:
            return None
        pairs = [(abs(self.p[i] - self.p[j]), (i, j)) for i in range(3) for j in range(i + 1, 3)]
        pairs.sort()
        if pairs[0][0] == pairs[1][0]:
            return None
        return list(pairs[0][1])

    def closest_time_pair(self) -> Optional[List[int]]:
        if self.n != 3:
            return None
        g = [self.notes[i + 1].start - self.notes[i].end for i in range(2)]
        if abs(g[0] - g[1]) < GAP_MARGIN_SEC:
            return None
        return [0, 1] if g[0] < g[1] else [1, 2]

    def pairs_with_delta(self, d: int) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(self.n) for j in range(i + 1, self.n) if abs(self.p[i] - self.p[j]) == d]

    # ---- patterns (3-note clips)
    def contour(self) -> Optional[str]:
        if self.n != 3:
            return None
        d = [self.p[1] - self.p[0], self.p[2] - self.p[1]]
        if 0 in d:
            return None
        return "-".join("up" if x > 0 else "down" for x in d)

    def interval_trend(self) -> Optional[str]:
        if self.n != 3:
            return None
        a, b = abs(self.p[1] - self.p[0]), abs(self.p[2] - self.p[1])
        if b - a >= INT_MARGIN:
            return "widen"
        if a - b >= INT_MARGIN:
            return "narrow"
        return "same" if a == b else None
