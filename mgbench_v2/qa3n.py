"""MGBench-v2 (3N): question families on top of oracle3n. Same candidate/selection machinery as the 2B builder."""
from __future__ import annotations

import random
from typing import Dict, List

from mgbench_v2 import phrasing as P
from mgbench_v2.oracle3n import Clip, GAP_MARGIN_SEC, INT_MARGIN, VEL_MARGIN
from mgbench_v2.qa2b import Cand

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(p: int) -> str:
    return f"{NAMES[p % 12]}{p // 12 - 1}"


def _vals(vs) -> str:
    vs = sorted(vs)
    if len(vs) == 1:
        return str(vs[0])
    if len(vs) == 2:
        return f"{vs[0]} or {vs[1]}"
    return ", ".join(str(x) for x in vs[:-1]) + f", or {vs[-1]}"


def _names(vs) -> str:
    vs = sorted(vs)
    t = [f"{note_name(v)} ({v})" for v in vs]
    if len(t) == 1:
        return t[0]
    if len(t) == 2:
        return f"{t[0]} or {t[1]}"
    return ", ".join(t[:-1]) + f", or {t[-1]}"


def _G(fam, sub, cat, name, akey, idx, whats, key, plural=True, **kw):
    return Cand(fam, sub, "grounding", name, cat, akey, list(idx), whats=whats, plural=plural, empty=(len(idx) == 0), prov={"key": key, **kw.pop("prov", {})}, **kw)


# ---------------------------------------------------------------------------------------- ABS / PITCH
def g_pitch(c: Clip, rng):
    have = sorted(set(c.p))
    absent = [p for p in range(21, 109) if p not in have]
    cands = []
    def add(vals):
        idx = c.with_pitches(vals)
        cands.append(_G("A-PITCH", "vals", "ABS", "pitch", f"k{len(vals)}_h{len(idx)}", idx,
                        [f"the note(s) whose pitch is {_vals(vals)} (MIDI)", f"the note(s) played at MIDI pitch {_vals(vals)}",
                         f"the note(s) whose pitch is {_names(vals)}", f"the note(s) with the pitch {_names(vals)}"],
                        f"pitch:{sorted(vals)}", prov={"vals": sorted(vals)}))
    for p in have:
        add([p])
    if len(have) >= 2:
        add(rng.sample(have, 2))
    if len(have) >= 3:
        add(have)
    for p in rng.sample(absent, 3):
        add([p])
    add([rng.choice(have)] + rng.sample(absent, 1))
    add(rng.sample(absent, 2))
    add(rng.sample(have, min(2, len(have))) + rng.sample(absent, 1))
    return cands


# ---------------------------------------------------------------------------------------- ORD / NOTE
ORD_WHATS = {
    "first": ["the first note", "the note that comes first", "the opening note"],
    "second": ["the second note", "the note that comes second", "the 2nd note"],
    "third": ["the third note", "the note that comes third", "the 3rd note"],
    "last": ["the last note", "the final note", "the note that comes last"],
}


def g_ord(c: Clip, rng):
    out = []
    for name, k in (("first", 1), ("second", 2), ("third", 3)):
        idx = c.ordinal(k)
        out.append(_G("O-NOTE", f"idx={name}", "ORD", "note", name + ("_empty" if not idx else ""), idx, ORD_WHATS[name], f"ord:{name}", plural=False))
    out.append(_G("O-NOTE", "idx=last", "ORD", "note", "last", c.last(), ORD_WHATS["last"], "ord:last", plural=False))
    return out


def g_ord_range(c: Clip, rng):
    if c.n < 2:
        return []
    out = []
    for name, idx, whats in (("prefix", c.prefix(2), ["the first two notes", "the two opening notes", "the first pair of notes"]),
                             ("suffix", c.suffix(2), ["the last two notes", "the two final notes", "the last pair of notes"])):
        out.append(_G("O-RANGE", name, "ORD", "note", name, idx, whats, f"ord:{name}"))
    return out


# ---------------------------------------------------------------------------------------- REL / PITCH, REL / VEL
def g_relpitch(c: Clip, rng):
    if c.n < 2:
        return []
    out = []
    for name, rank, high, whats in (
            ("highest", 1, True, ["the highest-pitch note(s)", "the note(s) with the highest pitch", "the highest note(s)"]),
            ("lowest", 1, False, ["the lowest-pitch note(s)", "the note(s) with the lowest pitch", "the lowest note(s)"]),
            ("second_highest", 2, True, ["the second-highest note(s)", "the note(s) with the second-highest pitch"]),
            ("second_lowest", 2, False, ["the second-lowest note(s)", "the note(s) with the second-lowest pitch"])):
        idx = c.pitch_rank(rank, high)
        out.append(_G("R-PITCH", f"rank={name}", "REL", "pitch", name + ("_empty" if not idx else ""), idx, whats, f"rp:{name}"))
    return out


def g_relvel(c: Clip, rng):
    out = []
    for name, idx, whats, plural in (
            ("loudest", c.loudest(), ["the loudest note", "the note played most loudly", "the note with the highest loudness"], False),
            ("softest", c.softest(), ["the softest note", "the quietest note", "the note played most softly"], False),
            ("second_loudest", c.second_loudest(), ["the second-loudest note", "the note with the second-highest loudness"], False)):
        if idx is not None:
            out.append(_G("R-VEL", f"rank={name}", "REL", "velocity", name, idx, whats, f"rv:{name}", plural=plural, prov={"margin": VEL_MARGIN}))
    return out


# ---------------------------------------------------------------------------------------- PAT
def g_pair(c: Clip, rng):
    out = []
    idx = c.closest_pitch_pair()
    if idx is not None:
        out.append(_G("P-PAIR", "pitch_closest", "PAT", "pair", "pitch_closest", idx,
                      ["the pair of notes whose pitches are closest to each other", "the two notes with the smallest pitch difference", "the pair of notes with the smallest pitch interval"],
                      "pair:pitch"))
    idx = c.closest_time_pair()
    if idx is not None:
        out.append(_G("P-PAIR", "time_closest", "PAT", "pair", "time_closest", idx,
                      ["the pair of consecutive notes with the shortest silent gap between them (from the end of one note to the start of the next)",
                       "the two consecutive notes that are closest together in time, measured from the end of the first to the start of the second",
                       "the consecutive pair of notes separated by the shortest pause"], "pair:time", prov={"gap_margin_sec": GAP_MARGIN_SEC}))
    return out


def g_pdiff(c: Clip, rng):
    if c.n < 2:
        return []
    out = []
    diffs = {}
    for i in range(c.n):
        for j in range(i + 1, c.n):
            diffs.setdefault(abs(c.p[i] - c.p[j]), []).append((i, j))
    def add(D, pairs):
        idx = sorted({k for pr in pairs for k in pr})
        b = str(D) if D <= 12 else "13-24"
        out.append(_G("P-PDIFF", "delta", "PAT", "pitch_diff", f"d{b}" + ("" if idx else "_empty"), idx,
                      [f"the pair of notes whose pitches differ by exactly {D} semitone{'s' if D != 1 else ''}", f"the two notes that are {D} semitone{'s' if D != 1 else ''} apart in pitch",
                       f"the pair of notes separated by {D} semitone{'s' if D != 1 else ''}"], f"pd:{D}", prov={"delta": D}))
    for D, pairs in diffs.items():
        if 1 <= D <= 24 and len(pairs) == 1:
            add(D, pairs)
    absent = [D for D in range(1, 25) if D not in diffs]
    for D in rng.sample(absent, min(3, len(absent))):
        add(D, [])
    return out


def g_contour(c: Clip, rng):
    t = c.contour()
    if t is None:
        return []
    out = []
    for pat in ("up-up", "up-down", "down-up", "down-down"):
        idx = [0, 1, 2] if pat == t else []
        a, b = pat.split("-")
        out.append(_G("P-CONTOUR", f"pitch={pat}", "PAT", "contour", pat + ("" if idx else "_empty"), idx,
                      [f"the segment where the pitch contour goes {a} then {b}", (f"the notes whose pitch first goes {a} and then goes {b}", True), f"the part of the clip where the pitch moves {a} and then {b}"],
                      f"ct:{pat}", plural=False))
    return out


def g_int(c: Clip, rng):
    t = c.interval_trend()
    if t is None:
        return []
    desc = {"widen": ["the segment where the pitch intervals between successive notes widen (the second interval is larger than the first)",
                      ("the notes whose successive pitch intervals get wider", True)],
            "narrow": ["the segment where the pitch intervals between successive notes narrow (the second interval is smaller than the first)",
                       ("the notes whose successive pitch intervals get narrower", True)],
            "same": ["the segment where the two pitch intervals between successive notes are equal in size",
                     ("the notes whose successive pitch intervals stay the same size", True)]}
    out = []
    for pat in ("widen", "narrow", "same"):
        idx = [0, 1, 2] if pat == t else []
        out.append(_G("P-INT", f"order={pat}", "PAT", "int", pat + ("" if idx else "_empty"), idx, desc[pat], f"in:{pat}", plural=False, prov={"margin_semitones": INT_MARGIN}))
    return out


SLOTS = {"A-PITCH": 4, "O-NOTE": 3, "O-RANGE": 1, "R-PITCH": 2, "R-VEL": 2, "P-PAIR": 2, "P-PDIFF": 2, "P-CONTOUR": 1, "P-INT": 1}
GEN3 = [g_pitch, g_ord, g_ord_range, g_relpitch, g_relvel, g_pair, g_pdiff, g_contour, g_int]


def candidates3(c: Clip, uid: str) -> Dict[str, List[Cand]]:
    from collections import defaultdict
    rng = random.Random(f"3n|{uid}")
    by = defaultdict(list)
    for g in GEN3:
        for cand in g(c, rng):
            by[cand.fam].append(cand)
    return by
