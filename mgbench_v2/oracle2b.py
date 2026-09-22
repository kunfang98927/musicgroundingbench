"""MGBench-v2 oracle for the two-bar subset.

Independent solver: it reads the *realised* notes (from the MIDI file) plus a small timing frame
(meter, tempo, music start, bar length) and answers every v2 question type from those alone.
It never imports the generator, never reads a control variable other than the frame, and returns
None whenever a question type is not answerable / not unambiguous for an excerpt (the caller then
skips that type for that excerpt instead of guessing).

Conventions (see SPEC_v2_draft.md):
  * bar membership is decided by onset time; the audible bar frame is the first onset of each bar
  * beat positions are measured in beats from the first onset of the bar and snapped to a 0.5-beat grid
  * one beat = the note value of the denominator (quarter for x/4, eighth for 6/8)
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pretty_midi

BEATS_PER_BAR = {"2/4": 2, "3/4": 3, "4/4": 4, "6/8": 6}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
INTERVAL_NAMES = {0: "unison", 1: "minor second", 2: "major second", 3: "minor third", 4: "major third", 5: "perfect fourth",
                  6: "tritone", 7: "perfect fifth", 8: "minor sixth", 9: "major sixth", 10: "minor seventh",
                  11: "major seventh", 12: "octave"}
NOTE_VALUE_NAMES = {
    False: {0.5: "eighth note", 1.0: "quarter note", 1.5: "dotted quarter note", 2.0: "half note", 3.0: "dotted half note"},
    True: {0.5: "sixteenth note", 1.0: "eighth note", 1.5: "dotted eighth note", 2.0: "quarter note", 3.0: "dotted quarter note"},  # 6/8
}
FRAME_TOL_SEC = 0.10          # how far a bar's first onset may lie from the nominal bar start
MIN_NOTE_SEC = 0.10           # shorter notes are ornament artefacts of the v1 generator
GRID_TOL_BEATS = 0.18         # |measured - nearest 0.5-beat grid point| for all notes of an excerpt (0.18 < 0.20 leaves a band no label depends on)
REST_TOL_SEC = 0.06           # a silence longer than this between a note's end and the next onset is a rest; length questions are only asked on rest-free excerpts
END_TOL_SEC = 0.06            # the last note must end within this distance of the end of bar 2 (its length is then "up to the end of the piece")
SYNC_CROSS_LO, SYNC_CROSS_HI = 0.10, 0.35   # a note sounding <= LO beats past a position is not held through it, >= HI is; in between the excerpt is ambiguous
SYNC_METERS = ("2/4", "3/4", "4/4")
_METRIC_WEIGHT = {"4/4": {0: 4, 4: 3, 2: 2, 6: 2}, "3/4": {0: 3, 2: 2, 4: 2}, "2/4": {0: 3, 2: 2}}      # half-beat index in the bar -> strength; all other (off-beat) positions: 1


def metric_weight(meter: str, h: int) -> int:
    """metrical strength of half-beat position h of a bar: beat 1 > (4/4: beat 3 >) other beats > off-beats"""
    return _METRIC_WEIGHT[meter].get(h, 1)
TEMPO_TOL_BPM = 3.0           # tempo answers are accepted within +-3 bpm; the measured tempo must be that close to the nominal one


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    pitch: int
    velocity: int

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Frame:
    meter: str
    tempo_bpm: int
    music_start: float
    bar_dur: float

    @property
    def beats_per_bar(self) -> int:
        return BEATS_PER_BAR[self.meter]

    @property
    def sec_per_beat(self) -> float:
        return 60.0 / self.tempo_bpm


def load_notes(midi_path: str) -> List[Note]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = [Note(n.start, n.end, n.pitch, n.velocity) for inst in pm.instruments for n in inst.notes]
    return sorted(notes, key=lambda n: (n.start, n.end))


def frame_from_basic_info(basic: Dict) -> Frame:
    return Frame(basic["time_signature"], int(basic["tempo_bpm"]), float(basic["music_start_sec"]), float(basic["bar_duration_sec"]))


class Excerpt:
    """Realised excerpt + everything derived from it. All flags are measurements."""

    def __init__(self, notes: Sequence[Note], frame: Frame):
        self.notes: List[Note] = sorted(notes, key=lambda n: (n.start, n.end))
        self.frame = frame
        n = len(self.notes)
        boundary = frame.music_start + frame.bar_dur - FRAME_TOL_SEC
        self.bar_of: List[int] = [1 if x.start < boundary else 2 for x in self.notes]
        self.bars: Dict[int, List[int]] = {1: [i for i in range(n) if self.bar_of[i] == 1], 2: [i for i in range(n) if self.bar_of[i] == 2]}
        # --- integrity flags
        self.ultra_short = any(x.dur < MIN_NOTE_SEC for x in self.notes)
        self.overlap = any(self.notes[i].end > self.notes[i + 1].start + 1e-4 for i in range(n - 1))
        # a note that starts 40-160 ms before the bar line is a pick-up / anticipation: its bar membership is a convention, so such clips are treated as
        # having no clear bar frame (every bar- or beat-based question skips them)
        b2 = frame.music_start + frame.bar_dur
        self.bar_ambiguous = any(b2 - 0.16 < x.start < b2 - 0.04 for x in self.notes)
        self.frame_clear = False
        self.bar_len_measured: Optional[float] = None
        if self.bars[1] and self.bars[2]:
            s1, s2 = self.notes[self.bars[1][0]].start, self.notes[self.bars[2][0]].start
            ok1 = abs(s1 - frame.music_start) <= FRAME_TOL_SEC
            ok2 = abs(s2 - (frame.music_start + frame.bar_dur)) <= FRAME_TOL_SEC
            self.frame_clear = ok1 and ok2 and not self.bar_ambiguous
            self.bar_len_measured = s2 - s1
        self.bar_start: Dict[int, Optional[float]] = {b: (self.notes[self.bars[b][0]].start if self.bars[b] else None) for b in (1, 2)}
        # --- grid positions (beats from the bar's first onset, snapped to 0.5)
        self.pos_beats: List[Optional[float]] = []
        self.grid: List[Optional[float]] = []
        for i, x in enumerate(self.notes):
            bs = self.bar_start[self.bar_of[i]]
            p = (x.start - bs) / frame.sec_per_beat if bs is not None else None
            self.pos_beats.append(p)
            self.grid.append(None if p is None else round(p * 2) / 2)
        self.grid_ok = all(p is not None and abs(p - g) <= GRID_TOL_BEATS for p, g in zip(self.pos_beats, self.grid)) and n > 0
        self.dur_beats = [x.dur / frame.sec_per_beat for x in self.notes]      # MIDI durations (not used for labels any more)
        # v2 (rc3) note length := distance from a note's start to the start of the next note; the last note lasts up to the end of bar 2.
        # This is what one hears only when no rest follows a note and the last note fills the rest of the piece, so every length-based question needs `rest_free`.
        self.piece_end = frame.music_start + 2 * frame.bar_dur
        self.rest_free = n > 1 and all(self.notes[i + 1].start - self.notes[i].end <= REST_TOL_SEC for i in range(n - 1)) and abs(self.piece_end - self.notes[-1].end) <= END_TOL_SEC
        # the bar frame can be stated as "two equal bars from the first note's start to the last note's end" only if the excerpt really is like that
        self.end_ok = n > 0 and abs(self.piece_end - self.notes[-1].end) <= END_TOL_SEC
        self.frame_full = self.frame_clear and self.end_ok
        nxt = [x.start for x in self.notes[1:]] + [self.piece_end]
        self.ioi_beats: List[Optional[float]] = [(nxt[i] - self.notes[i].start) / frame.sec_per_beat for i in range(n)]
        self.value: List[Optional[float]] = [self._nearest_value(d) for d in self.ioi_beats]
        self.values_ok = self.rest_free and all(v is not None for v in self.value)


    # ------------------------------------------------------------------ helpers
    def _nearest_value(self, d: float) -> Optional[float]:
        cands = (0.5, 1.0, 1.5, 2.0, 3.0)
        v = min(cands, key=lambda c: abs(c - d))
        return v if abs(v - d) <= GRID_TOL_BEATS else None

    @property
    def n(self) -> int:
        return len(self.notes)

    def span(self, i: int) -> Tuple[float, float]:
        return (self.notes[i].start, self.notes[i].end)

    def spans(self, idx: Sequence[int]) -> List[Tuple[float, float]]:
        return [self.span(i) for i in sorted(idx)]

    @property
    def usable(self) -> bool:
        """excerpt has no ultra-short notes / overlaps and two non-empty bars (D3 filter)"""
        return (not self.ultra_short) and (not self.overlap) and len(self.bars[1]) >= 1 and len(self.bars[2]) >= 1

    # ================================================================== RHYTHM
    def bar_initial_notes(self) -> Optional[List[int]]:
        return [self.bars[1][0], self.bars[2][0]] if self.usable and self.frame_clear else None

    def notes_on_beat(self, k: int) -> Optional[List[int]]:
        """notes starting exactly on beat k (1-based) of a bar (grid position k-1)"""
        if not (self.usable and self.frame_clear and self.grid_ok) or not 1 <= k <= self.frame.beats_per_bar:
            return None
        return [i for i in range(self.n) if self.grid[i] == k - 1]

    def notes_at_bar_middle(self) -> Optional[List[int]]:
        if not (self.usable and self.frame_clear and self.grid_ok):
            return None
        mid = self.frame.beats_per_bar / 2
        return [i for i in range(self.n) if self.grid[i] == mid]

    def notes_in_second_half(self, bar: Optional[int] = None) -> Optional[List[int]]:
        """notes starting at or after the middle of a bar (middle included); optionally one bar only"""
        if not (self.usable and self.frame_clear and self.grid_ok):
            return None
        mid = self.frame.beats_per_bar / 2
        return [i for i in range(self.n) if self.grid[i] >= mid and (bar is None or self.bar_of[i] == bar)]

    def notes_with_value(self, beats: float) -> Optional[List[int]]:
        if not (self.usable and self.frame_clear and self.grid_ok and self.values_ok):
            return None
        return [i for i in range(self.n) if self.value[i] == beats]

    def value_name(self, i: int) -> Optional[str]:
        v = self.value[i]
        return None if v is None else NOTE_VALUE_NAMES[self.frame.meter == "6/8"].get(v)

    def syncopated_notes(self) -> Optional[List[int]]:
        """rc5: syncopation in the sense of Longuet-Higgins & Lee: a note that starts at a metrically WEAKER position and is held through a metrically STRONGER
        position (which therefore gets no new onset).  Positions are the half-beats of the bar (metric_weight); only 2/4, 3/4, 4/4 (in 6/8 one beat is an
        eighth note, the strong pulses are the two dotted quarters and 'off the beat' has no meaning at this grid).
        A note 'holds through' a stronger position when it still sounds >= SYNC_CROSS_HI beats after it; <= SYNC_CROSS_LO beats counts as not held; an excerpt with a
        note in between is ambiguous -> None (no question)."""
        if self.frame.meter not in SYNC_METERS or not (self.usable and self.frame_clear and self.grid_ok and self.values_ok):
            return None
        H = 2 * self.frame.beats_per_bar
        out = []
        for i in range(self.n):
            h0 = int(round(self.grid[i] * 2))
            a = (self.bar_of[i] - 1) * H + h0                   # onset, in half-beats from the start of bar 1
            q = a + 2 * self.ioi_beats[i]                       # end of the note (next onset / end of the piece), in half-beats
            w0 = metric_weight(self.frame.meter, h0)
            held = False
            for s in range(a + 1, int(math.ceil(q)) + 2):
                if metric_weight(self.frame.meter, s % H) <= w0:
                    continue
                cross = (q - s) / 2.0                           # beats the note keeps sounding after position s
                if SYNC_CROSS_LO < cross < SYNC_CROSS_HI:
                    return None
                held = held or cross >= SYNC_CROSS_HI
            if held:
                out.append(i)
        return out

    def ioi_sec(self, i: int) -> float:
        return (self.notes[i + 1].start if i + 1 < self.n else self.piece_end) - self.notes[i].start

    def longest_note_of_bar(self, bar: int, min_ratio: float = 1.3) -> Optional[int]:
        """note with the longest length (start -> next start, the last note up to the end of bar 2) in the bar; rest-free excerpts only; needs a clear margin to the runner-up"""
        if not self.usable or self.bar_ambiguous or not self.rest_free:
            return None
        idx = list(self.bars[bar])
        if len(idx) < 2:
            return None
        order = sorted(idx, key=lambda i: -self.ioi_sec(i))
        return order[0] if self.ioi_sec(order[0]) >= min_ratio * self.ioi_sec(order[1]) else None

    def shortest_notes_per_bar(self) -> Optional[int]:
        """how many of the shortest note fill one bar (integer only)"""
        if not (self.usable and self.frame_clear and self.grid_ok and self.values_ok):
            return None
        q = self.frame.beats_per_bar / min(self.value)
        return int(q) if abs(q - round(q)) < 1e-9 else None

    def measured_tempo(self) -> Optional[float]:
        """tempo (bpm) from a least-squares fit of onset time against bar-relative grid position over both bars"""
        if not (self.usable and self.frame_clear and self.grid_ok) or self.n < 4:
            return None
        bpb = self.frame.beats_per_bar
        xs = [(self.bar_of[i] - 1) * bpb + self.grid[i] for i in range(self.n)]
        ts = [x.start for x in self.notes]
        mx, mt = sum(xs) / self.n, sum(ts) / self.n
        var = sum((x - mx) ** 2 for x in xs)
        if var == 0:
            return None
        slope = sum((x - mx) * (t - mt) for x, t in zip(xs, ts)) / var       # seconds per beat
        return 60.0 / slope if slope > 0 else None

    def _tempo_consistent(self) -> bool:
        m = self.measured_tempo()
        return m is not None and abs(m - self.frame.tempo_bpm) <= TEMPO_TOL_BPM

    def tempo_answer(self) -> Optional[int]:
        """nominal tempo (accepted +-3 bpm), only when the audio really has that tempo"""
        return self.frame.tempo_bpm if self._tempo_consistent() else None

    def beats_per_bar_answer(self) -> Optional[int]:
        """beats per bar given the stated tempo; needs the same tempo consistency"""
        return self.frame.beats_per_bar if self._tempo_consistent() else None

    def beat_of_note(self, i: int) -> Optional[str]:
        """'beat 2' or 'the second half of beat 3' (off-beat = half a beat after beat k)"""
        if not (self.usable and self.frame_clear and self.grid_ok):
            return None
        g = self.grid[i]
        k = int(g) + 1
        return f"beat {k}" if g % 1 == 0 else f"the second half of beat {k}"

    # ================================================================== TONALITY
    def pitch_classes(self) -> List[int]:
        return [x.pitch % 12 for x in self.notes]

    @staticmethod
    def scale_pcs(root_pc: int) -> List[int]:
        return sorted((root_pc + s) % 12 for s in MAJOR_STEPS)

    def notes_outside_scale(self, root_pc: int) -> List[int]:
        sc = set(self.scale_pcs(root_pc))
        return [i for i, p in enumerate(self.pitch_classes()) if p not in sc]

    def scale_notes_never_played(self, root_pc: int) -> List[int]:
        played = set(self.pitch_classes())
        return [p for p in self.scale_pcs(root_pc) if p not in played]

    def distinct_pitch_classes(self) -> List[int]:
        return sorted(set(self.pitch_classes()))

    # ================================================================== INTERVALS
    def moves(self) -> List[Optional[int]]:
        return [None] + [self.notes[i].pitch - self.notes[i - 1].pitch for i in range(1, self.n)]

    def notes_reached_by(self, lo: int, hi: Optional[int]) -> List[int]:
        """notes reached from the previous note by an absolute move of lo..hi semitones (hi=None: no upper bound)"""
        m = self.moves()
        return [i for i in range(1, self.n) if abs(m[i]) >= lo and (hi is None or abs(m[i]) <= hi)]

    def notes_reached_by_named(self, semitones: int, direction: Optional[str] = None) -> List[int]:
        m = self.moves()
        out = []
        for i in range(1, self.n):
            d = m[i]
            if abs(d) != semitones:
                continue
            if direction == "ascending" and d <= 0:
                continue
            if direction == "descending" and d >= 0:
                continue
            out.append(i)
        return out

    def repeated_notes(self) -> List[int]:
        return [i for i in range(1, self.n) if self.notes[i].pitch == self.notes[i - 1].pitch]

    def largest_jump(self) -> int:
        return max(abs(d) for d in self.moves()[1:])

    def count_moves_of(self, semitones: int) -> int:
        return sum(1 for d in self.moves()[1:] if abs(d) == semitones)

    def downward_notes(self) -> List[int]:
        return [i for i in range(1, self.n) if self.notes[i].pitch < self.notes[i - 1].pitch]

    def pitch_rank_notes(self, rank: int, highest: bool = True) -> List[int]:
        """notes at the rank-th highest (or lowest) distinct pitch"""
        ps = sorted({x.pitch for x in self.notes}, reverse=highest)
        if rank > len(ps):
            return []
        return [i for i, x in enumerate(self.notes) if x.pitch == ps[rank - 1]]

    # ================================================================== BAR STRUCTURE
    def bar_note_count(self, bar: int) -> int:
        return len(self.bars[bar])

    def highest_pitch(self, bar: int) -> int:
        return max(self.notes[i].pitch for i in self.bars[bar])

    def last_notes_of_bars(self) -> List[int]:
        return [self.bars[1][-1], self.bars[2][-1]]

    # ================================================================== HARMONY (arpeggiated triads)
    def triad_of_bar(self, bar: int, cover: float = 0.9) -> Optional[Tuple[int, str]]:
        """(root pc, 'major'|'minor') iff exactly one triad contains >= `cover` of the bar's notes and all three
        triad notes are played; else None (bar is not an unambiguous triad)"""
        pcs = [self.notes[i].pitch % 12 for i in self.bars[bar]]
        if len(pcs) < 3:
            return None
        found = []
        for root in range(12):
            for q, iv in (("major", (0, 4, 7)), ("minor", (0, 3, 7))):
                tri = {(root + x) % 12 for x in iv}
                if sum(p in tri for p in pcs) >= cover * len(pcs) and tri <= set(pcs):
                    found.append((root, q))
        return found[0] if len(found) == 1 else None

    @staticmethod
    def triad_name(t: Tuple[int, str]) -> str:
        return f"{NOTE_NAMES[t[0]]} {t[1]}"

    def notes_not_in_other_triad(self, bar: int, other_bar: int) -> Optional[List[int]]:
        t = self.triad_of_bar(other_bar)
        if t is None or self.triad_of_bar(bar) is None:
            return None
        tri = {(t[0] + x) % 12 for x in ((0, 4, 7) if t[1] == "major" else (0, 3, 7))}
        return [i for i in self.bars[bar] if self.notes[i].pitch % 12 not in tri]

    def triad_member_notes(self, bar: int, member: str) -> Optional[List[int]]:
        t = self.triad_of_bar(bar)
        if t is None:
            return None
        off = {"root": 0, "third": 4 if t[1] == "major" else 3, "fifth": 7}[member]
        return [i for i in self.bars[bar] if self.notes[i].pitch % 12 == (t[0] + off) % 12]

    def root_motion(self) -> Optional[int]:
        t1, t2 = self.triad_of_bar(1), self.triad_of_bar(2)
        return None if t1 is None or t2 is None else (t2[0] - t1[0]) % 12

    # ================================================================== REPETITION
    def bar_shift(self) -> Optional[Tuple[Optional[int], int]]:
        """(constant shift or None, number of positions whose pitch differs) when both bars have the same note count"""
        b1, b2 = self.bars[1], self.bars[2]
        if len(b1) != len(b2) or not b1:
            return None
        d = [self.notes[j].pitch - self.notes[i].pitch for i, j in zip(b1, b2)]
        shift = d[0] if all(x == d[0] for x in d) else None
        return shift, sum(1 for x in d if x != 0)

    def bar2_notes_differing(self) -> Optional[List[int]]:
        b1, b2 = self.bars[1], self.bars[2]
        if len(b1) != len(b2) or not b1:
            return None
        return [j for i, j in zip(b1, b2) if self.notes[j].pitch != self.notes[i].pitch]

    # ================================================================== ARPEGGIO SHAPE
    def drops_to_lowest(self, bar: int) -> int:
        return len(self.drop_back_notes(bar))

    def drop_back_notes(self, bar: int) -> List[int]:
        """notes of the bar that step down onto the bar's lowest pitch from a higher note (the arpeggio starts over)"""
        idx = self.bars[bar]
        if len(idx) < 2:
            return []
        lo = min(self.notes[i].pitch for i in idx)
        return [b for a, b in zip(idx, idx[1:]) if self.notes[b].pitch < self.notes[a].pitch and self.notes[b].pitch == lo]

    # ------------------------------------------------------------------ HARMONIC PATTERN (arpeggio shape)
    def arpeggio_kind(self, bar: int) -> Optional[str]:
        """shape of the bar's arpeggio, from the pitch ranks alone: the bar has exactly three distinct pitches (low, mid, high) and follows
        'ascending' (low mid high | low mid high ...: rises, then jumps back to the lowest note) or 'updown' (low mid high mid | low mid high mid ...: rises, then falls
        back); a bar with fewer than 4 notes cannot tell the two apart and a bar that follows neither gets no label"""
        idx = self.bars[bar]
        pit = sorted({self.notes[i].pitch for i in idx})
        if len(idx) < 4 or len(pit) != 3:
            return None
        r = [pit.index(self.notes[i].pitch) for i in idx]
        asc = [(0, 1, 2)[k % 3] for k in range(len(r))]
        ud = [(0, 1, 2, 1)[k % 4] for k in range(len(r))]
        if r == asc:
            return "ascending"
        return "updown" if r == ud else None

    def arpeggio_first_cycle(self, bar: int) -> Optional[List[int]]:
        """the notes of the first complete repetition of the bar's arpeggio (3 notes ascending, 4 notes up-and-down)"""
        k = self.arpeggio_kind(bar)
        return None if k is None else list(self.bars[bar][:3 if k == "ascending" else 4])
