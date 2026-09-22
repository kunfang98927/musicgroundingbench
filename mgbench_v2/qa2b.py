"""MGBench-v2 (2B): question families, balanced selection and QA record construction.

Every answer comes from `oracle2b.Excerpt` (measurements on the realised MIDI). A family returns *candidates* per excerpt
(a candidate = one concrete question with its answer); `select` then draws at most one candidate per excerpt and family so that
answer classes are balanced (G3a) and empty answers stay rare. Nothing here reads a generator control variable.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from mgbench_v2 import phrasing as P
from mgbench_v2.oracle2b import NOTE_NAMES, NOTE_VALUE_NAMES, SYNC_METERS, Excerpt

FPS = 75
ORACLE_VERSION = "oracle2b-9"
EMPTY_CAP = 0.20            # max share of empty grounding answers within a family
YESNO_CAP = 0.10            # max share of yes/no items within a concept's understanding items


# ------------------------------------------------------------------------------------------------ candidate
@dataclass
class Cand:
    fam: str                          # e.g. "R-G2"
    sub: str                          # e.g. "beat=3"  (query_key suffix)
    kind: str                         # "grounding" | "understanding"
    concept: str
    cat: str                          # ABS | REL | ORD | PAT (kept for the existing metric grouping)
    akey: str                         # answer class used for balancing
    idx: List[int]                    # target notes (grounding) / evidence notes (understanding)
    whats: List = field(default_factory=list)   # grounding: noun phrases (one is drawn); a phrase is a str (uses `plural`) or (str, plural)
    plural: bool = True
    spans: Optional[List[Tuple[float, float]]] = None  # explicit spans (bar spans)
    ctx: Tuple[str, ...] = ()         # mandatory context sentences: meter | unit | tempo | metric | frame | text:<sentence>
    opt: Tuple[str, ...] = ()         # optional context (each used with prob. 0.5)
    empty: bool = False
    # understanding
    asks: List[str] = field(default_factory=list)
    fmt: Dict[str, object] = field(default_factory=dict)
    gold: object = None
    gtype: str = ""
    ans: Optional[Dict[str, str]] = None
    prov: Dict[str, object] = field(default_factory=dict)
    stratum: str = ""                 # question parameter / context variable that must be independent of the answer


def _names(root: int) -> str:
    return ", ".join(NOTE_NAMES[p] for p in Excerpt.scale_pcs(root))


def _art(name: str) -> str:
    """indefinite article by pronunciation of the first letter (note names such as 'E minor', 'F major', 'A#' take 'an')"""
    return "an" if (name[0] in "aeiou" or name[0] in "AEF") else "a"


def _G(fam, sub, concept, cat, akey, idx, whats, plural=True, **kw):
    return Cand(fam, sub, "grounding", concept, cat, akey, list(idx), whats=whats, plural=plural, empty=(len(idx) == 0 and kw.get("spans") is None), **kw)


def _U(fam, sub, concept, cat, akey, idx, asks, gold, gtype, ans, fmt=None, **kw):
    return Cand(fam, sub, "understanding", concept, cat, akey, list(idx), asks=asks, gold=gold, gtype=gtype, ans=ans, fmt=fmt or {}, **kw)


def _bucket(n: int, top: int = 3) -> str:
    return "0" if n == 0 else (str(n) if n < top else f"{top}+")


def _tempo_ok(e: Excerpt) -> bool:
    return e.tempo_answer() is not None


# ------------------------------------------------------------------------------------------------ RHYTHM
def g_R_G1(e):
    idx = e.bar_initial_notes()
    return [] if idx is None else [_G("R-G1", "bar_start", "rhythm", "ORD", "bar_start", idx,
                                      ["the notes that begin each bar", ("the first note of each bar", False), "the notes that start at the beginning of a bar"])]


def g_R_G2(e):
    if not _tempo_ok(e):            # rc3: the frame (meter, tempo, first note = beat 1) is always stated, so it has to be true
        return []
    out = []
    for k in range(1, e.frame.beats_per_bar + 1):
        idx = e.notes_on_beat(k)
        if idx is None:
            continue
        ord_ = P.ORD.get(k, str(k))
        out.append(_G("R-G2", f"beat={k}", "rhythm", "ORD", f"k{k}" + ("_empty" if not idx else ""), idx,
                      [f"the notes that start on beat {k} of a bar", f"the notes that begin on the {ord_} beat of a bar", f"the notes that land on beat {k}"],
                      ctx=("meter", "unit", "tempo"), prov={"k": k}, stratum=f"meter{e.frame.meter}|k{k}"))
    return out


def g_R_G3(e):
    idx = e.notes_at_bar_middle()
    return [] if idx is None else [_G("R-G3", "bar_middle", "rhythm", "ORD", "mid" + ("_empty" if not idx else ""), idx,
                                      ["the notes that start exactly halfway through a bar", "the notes that begin at the middle of a bar"])]


def g_R_G4(e):
    if not _tempo_ok(e):
        return []
    out = []
    names = NOTE_VALUE_NAMES[e.frame.meter == "6/8"]
    for v in (0.5, 1.0, 1.5, 2.0):
        idx = e.notes_with_value(v)
        if idx is None:
            continue
        nm = names[v]
        out.append(_G("R-G4", f"value={nm.replace(' ', '_')}", "rhythm", "ABS", nm + ("_empty" if not idx else ""), idx,
                      [f"all the {nm}s", f"the {nm}s of the clip", f"the notes that are {nm}s"],
                      ctx=("meter", "unit", "tempo"), prov={"value_beats": v}, stratum=f"meter{e.frame.meter}|v{v}"))
    return out


def g_R_G5(e):
    if e.frame.meter not in SYNC_METERS or not _tempo_ok(e):
        return []
    idx = e.syncopated_notes()
    if idx is None:
        return []
    return [_G("R-G5", "syncopated", "rhythm", "PAT", "sync" + ("_empty" if not idx else ""), idx,
               ["the syncopated notes, i.e. the notes that start on a weaker position of the bar and are held through a stronger position, which gets no new note",
                "the notes that begin on a weak position and keep sounding through a stronger one, so that the stronger position has no note of its own",
                "the syncopated notes: those that start on a weaker position and are held across a stronger beat or a bar line"],
               ctx=("meter", "unit", "tempo", "metric"))]


def g_R_G6(e):
    out = []
    if e.bar_ambiguous:
        return out
    for b in (1, 2):
        i = e.longest_note_of_bar(b, 1.3)
        if i is not None:
            out.append(_G("R-G6", f"longest_bar={b}", "rhythm", "ABS", f"bar{b}", [i],
                          [f"the longest note of the {P.ORD[b]} bar", f"the note that lasts longest in the {P.ORD[b]} bar"], plural=False,
                          prov={"min_ratio": 1.3}))
    return out


def _time_refs(e, k=5):
    """time-referenced notes (unambiguous: onsets differ by >= 0.10 s); a fixed random subset per excerpt"""
    rng = random.Random(f"refs|{getattr(e, 'uid', '')}")
    idx = list(range(e.n))
    rng.shuffle(idx)
    out = []
    for i in idx:
        s6 = round(e.notes[i].start, 6)
        if abs((s6 * 100) % 1 - 0.5) < 0.05:          # too close to a rounding boundary of the 2-decimal label
            continue
        out.append((f"{s6:.2f}", i))
        if len(out) == k:
            break
    return out


REF_FORMS = ["the note that starts at {t} seconds", "the note whose onset is at {t} seconds", "the note beginning at {t} s"]


def u_R_U3(e):
    if not _tempo_ok(e) or e.notes_with_value(0.5) is None:   # requires every duration to sit on the value grid
        return []
    out = []
    for t, i in _time_refs(e):
        nm = e.value_name(i)
        if nm is None:
            continue
        ref = REF_FORMS[i % 3].format(t=t)
        out.append(_U("R-U3", "note_value", "rhythm", "ABS", nm, [i],
                      ["What note value does {ref} have?", "What is the note value of {ref}?", "Which note value best describes {ref}?",
                       "How long is {ref}, as a note value?", "Name the note value of {ref}.", "Is {ref} a half, quarter, eighth or another kind of note? Name its value."],
                      nm, "class_label", {"0": nm, "1": f"{_art(nm).capitalize()} {nm}.", "2": f"{ref[0].upper() + ref[1:]} is {_art(nm)} {nm}."},
                      fmt={"ref": ref}, ctx=("meter", "unit", "tempo"), prov={"note_index": i, "onset": t}, stratum=f"meter{e.frame.meter}|t{int(float(t) / 0.5)}"))
    return out


def u_R_U4(e):
    """beat position of a note referred to by a *property* (highest / lowest / longest note of a bar), so that the wording carries no
    positional cue (a time reference does: early clip time <=> early beat position)"""
    if e.notes_on_beat(1) is None or e.bar_ambiguous or not _tempo_ok(e):
        return []
    out = []
    for b in (1, 2):
        idx = e.bars[b]
        refs = []
        for kind, key in (("highest", lambda i: e.notes[i].pitch), ("lowest", lambda i: -e.notes[i].pitch)):
            best = max(key(i) for i in idx)
            top = [i for i in idx if abs(key(i) - best) < 1e-9]
            if len(top) == 1:
                refs.append((kind, top[0]))
        li = e.longest_note_of_bar(b, 1.3)
        if li is not None:
            refs.append(("longest", li))
        for kind, i in refs:
            ref = f"the {kind} note of the {P.ORD[b]} bar"
            pos = e.grid[i] + 1.0
            pos_s = f"{pos:g}"
            out.append(_U("R-U4", "beat_position", "rhythm", "ORD", pos_s, [i],
                          ["Counting the start of the bar as position 1, at which beat position does {ref} start? Answer in steps of half a beat (for example 2 or 2.5).",
                           "Within its bar, at what beat position does {ref} start? Position 1 is the start of the bar; use half-beat steps such as 3.5.",
                           "{ref_c} is at which beat position of its bar (bar start = 1, half-beat steps allowed)?",
                           "Give the beat position of {ref} inside its bar, with the first beat of the bar as 1 and half beats as .5.",
                           "Where in its bar does {ref} begin? Answer with a beat position (bar start = 1, half-beat steps).",
                           "At what position, counted in beats from the bar start (= 1) in half-beat steps, does {ref} begin?"],
                          float(pos), "number", {"0": pos_s, "1": f"Beat position {pos_s}", "2": f"It starts at beat position {pos_s}."},
                          fmt={"ref": ref, "ref_c": ref[0].upper() + ref[1:]}, ctx=("meter", "unit", "tempo"),
                          prov={"note_index": i, "grid_beats": e.grid[i], "ref_kind": kind, "bar": b}, stratum=f"meter{e.frame.meter}|{kind}{b}"))
    return out


def u_R_U6(e):
    idx = e.notes_in_second_half(1)
    if idx is None or e.bar_ambiguous:
        return []
    n = len(idx)
    return [_U("R-U6", "second_half_bar1", "rhythm", "ORD", str(n), idx, [],
               n, "integer", P.numeric_answer_texts(n, "notes"),
               fmt={"np": "notes that start in the second half of the first bar (from the middle of the bar up to its end, the middle included)"}, ctx=(),
               stratum=f"bar{e.frame.bar_dur:.2f}")]                 # rc8: the bar length (= the clip length) must not predict the count


# ------------------------------------------------------------------------------------------------ TONALITY
def _tonal_cands(e, fn):
    return [fn(e, r) for r in range(12)]


def g_T_G1(e):
    if not e.usable:
        return []
    out = []
    for r in range(12):
        idx = e.notes_outside_scale(r)
        nm = _names(r)
        out.append(_G("T-G1", f"outside_set", "tonality", "ABS", _bucket(len(idx)), idx,
                      [f"the notes whose note name is not among {nm} (octave ignored)", f"the notes that do not belong to the set {nm}, ignoring octave",
                       f"the notes that are outside the note set {nm}"], prov={"set": nm}))
    return out


def u_T(e):
    if not e.usable:
        return []
    out = []
    for r in range(12):
        nm = _names(r)
        outside = e.notes_outside_scale(r)
        n = len(outside)
        out.append(_U("T-U1", "count_outside", "tonality", "ABS", _bucket(n, 4), outside,
                      ["How many notes of this clip do not belong to the set {nm} (octave ignored)?", "Count the notes that are not among {nm}, ignoring octave.",
                       "How many notes fall outside the note set {nm} (ignoring octave)?", "How many notes in this clip use a note name that is not in {nm}?",
                       "What is the number of notes whose name is outside {nm} (octave does not matter)?", "Among the notes of this clip, how many are not part of {nm} (octave ignored)?"],
                      n, "integer", P.numeric_answer_texts(n, "notes"), fmt={"nm": nm}, prov={"set": nm}))
        never = e.scale_notes_never_played(r)
        names = [NOTE_NAMES[p] for p in never]
        txt = ", ".join(names) if names else "none"
        out.append(_U("T-U2", "never_played", "tonality", "ABS", _bucket(len(never)), list(range(e.n)),
                      ["Which of the notes {nm} does this clip never play (octave ignored)?", "Which notes from {nm} are missing from the clip (ignoring octave)?",
                       "List every note among {nm} that does not occur in the clip, ignoring octave. Answer none if all of them occur.",
                       "Of {nm}, which note names are never played here (octave ignored)? Say none if all are played.",
                       "Name the notes of {nm} that are absent from this clip (ignoring octave); answer none if there are none.",
                       "Which members of {nm} never sound in the clip? Ignore octave and answer none if every member sounds."],
                      names, "sequence", {"0": txt, "1": ("None of them is missing." if not names else f"Missing: {txt}."), "2": ("All of these notes are played." if not names else f"The clip never plays {txt}.")},
                      fmt={"nm": nm}, prov={"set": nm}))
        yes = n == 0
        # rc6: "yes" needs every note as evidence (proving universality); "no" only needs the counterexample(s) that fall outside the set
        out.append(_U("T-U5", "all_in_set", "tonality", "ABS", "yes" if yes else "no", (list(range(e.n)) if yes else outside), ["Are all the notes of this clip taken from {nm} (octave ignored)?",
                      "Does every note of this clip have a name in the set {nm}, ignoring octave?", "Do all notes belong to {nm} (octave does not matter)?",
                      "Is every note in this clip one of {nm} (octave ignored)?"],
                      yes, "boolean", {"0": "yes" if yes else "no", "1": "Yes." if yes else "No.", "2": "Yes, every note belongs to that set." if yes else "No, at least one note is outside that set."},
                      fmt={"nm": nm}, prov={"set": nm}))
    pcs = e.distinct_pitch_classes()
    k = len(pcs)
    out.append(_U("T-U3", "distinct_names", "tonality", "ABS", str(k), list(range(e.n)),
                  ["How many different note names (ignoring octave) does this clip use?", "How many distinct pitch classes occur in the clip?",
                   "Ignoring octave, how many different notes are played?", "Count the different note names in this clip, octave ignored.",
                   "How many unique note names appear in the clip (octave does not matter)?", "What is the number of distinct note names used here, ignoring octaves?"],
                  k, "integer", P.numeric_answer_texts(k, "note names")))
    names = [NOTE_NAMES[p] for p in pcs]
    out.append(_U("T-U4", "list_names", "tonality", "ABS", str(k), list(range(e.n)),
                  ["Name every note (ignoring octave) that occurs in this clip.", "List all note names used in the clip, octave ignored, in ascending order from C.",
                   "Which note names occur in this clip (octave does not matter)? List them from C upward.", "Give the set of note names that are played, ignoring octave.",
                   "Ignoring octave, which notes does the clip contain? List them in order from C to B.",
                   "Write down, from C upward and without octave information, every note name that sounds in this clip."],
                  names, "sequence", {"0": ", ".join(names), "1": ", ".join(names) + ".", "2": f"The clip uses {', '.join(names)}."}))
    return out


# ------------------------------------------------------------------------------------------------ INTERVAL
def g_I(e):
    if not e.usable or e.n < 2:
        return []
    out = []
    for fam, lo, hi, txt in (("I-G1", 1, 2, "a move of 1 or 2 semitones"), ("I-G2", 3, 5, "a move of 3 to 5 semitones"), ("I-G3", 6, None, "a move of 6 semitones or more")):
        idx = e.notes_reached_by(lo, hi)
        out.append(_G(fam, f"move={lo}-{hi or 'up'}", "interval", "REL", "empty" if not idx else _bucket(len(idx)), idx,
                      [f"the notes reached from the previous note by {txt}", f"the notes that follow the previous note by {txt} (up or down)",
                       f"the notes whose distance from the previous note is {txt.replace('a move of ', '')}"], prov={"lo": lo, "hi": hi}))
    for s in range(1, 13):
        for d in ("ascending", "descending"):
            idx = e.notes_reached_by_named(s, d)
            nm = {1: "minor second", 2: "major second", 3: "minor third", 4: "major third", 5: "perfect fourth", 6: "tritone", 7: "perfect fifth",
                  8: "minor sixth", 9: "major sixth", 10: "minor seventh", 11: "major seventh", 12: "octave"}[s]
            dart = "an" if d == "ascending" else "a"
            art = _art(nm)
            out.append(_G("I-G4", f"named={d}_{s}", "interval", "REL", f"s{s}" + ("_empty" if not idx else ""), idx,
                          [f"the notes reached by {dart} {d} {nm} ({s} semitone{'s' if s > 1 else ''})", f"the notes that come {art} {nm} ({s} semitone{'s' if s > 1 else ''}) {'above' if d == 'ascending' else 'below'} the previous note",
                           f"the notes reached by {'rising' if d == 'ascending' else 'falling'} {s} semitone{'s' if s > 1 else ''} from the previous note"],
                          prov={"semitones": s, "direction": d}))
    idx = e.repeated_notes()
    out.append(_G("I-G5", "repeated", "interval", "REL", "empty" if not idx else "rep", idx,
                  ["the notes that repeat the pitch of the previous note", "the notes played at the same pitch as the note before them"]))
    for rank in (1, 2):
        for hl in ("highest", "lowest"):
            idx = e.pitch_rank_notes(rank, hl == "highest")
            adj = "" if rank == 1 else "second-"
            out.append(_G("I-G6", f"pitch_rank={hl}{rank}", "interval", "ABS", f"{hl}{rank}" + ("_empty" if not idx else ""), idx,
                          [f"the note(s) with the {adj}{hl} pitch", f"the {adj}{hl}-pitched note(s) of the clip"], prov={"rank": rank, "which": hl}))
    for d, fn in (("upward", lambda: [i for i in range(1, e.n) if e.notes[i].pitch > e.notes[i - 1].pitch]), ("downward", e.downward_notes)):
        idx = fn()
        out.append(_G("I-G7", f"move={d}", "interval", "REL", f"{d}" + ("_empty" if not idx else ""), idx,
                      [f"the notes reached by a {d} move from the previous note", f"the notes that are {'higher' if d == 'upward' else 'lower'} than the note before them"], prov={"direction": d}))
    return out


def u_I(e):
    if not e.usable or e.n < 2:
        return []
    out = []
    for s in range(1, 9):
        n = e.count_moves_of(s)
        out.append(_U("I-U1", "count_semitones", "interval", "REL", _bucket(n, 3), e.notes_reached_by_named(s), [
            "How many times does the melody move by exactly {sem} (up or down) between consecutive notes?", "Count the moves of exactly {sem} between neighbouring notes.",
            "How often does the pitch change by exactly {sem} from one note to the next?",
            "Between consecutive notes, how many moves span exactly {sem} (either direction)?", "How many note-to-note moves of exactly {sem} are there?",
            "What is the number of consecutive-note moves that are exactly {sem} wide?"], n, "integer", P.numeric_answer_texts(n, "moves"), fmt={"s": s, "sem": f"{s} semitone" + ("" if s == 1 else "s")}, prov={"semitones": s}, stratum=f"N{s}"))
    j = e.largest_jump()
    tied = [i for i in range(1, e.n) if abs(e.notes[i].pitch - e.notes[i - 1].pitch) == j]      # the evidence is every note pair that makes the largest move
    ii = tied[0]
    out.append(_U("I-U2", "largest_jump", "interval", "REL", str(j), sorted({k for i in tied for k in (i - 1, i)}), [
        "What is the largest jump between two consecutive notes, in semitones?", "By how many semitones does the widest note-to-note move span?",
        "Between consecutive notes, what is the biggest pitch distance in semitones?", "How many semitones is the largest melodic leap in this clip?",
        "Give the size, in semitones, of the widest move between neighbouring notes.",
        "Looking at every pair of consecutive notes, what is the biggest pitch change in semitones?"], j, "integer", P.numeric_answer_texts(j, "semitones"), prov={"first_note_of_jump": ii - 1}))
    for d, idx in (("downward", e.downward_notes()), ("upward", [i for i in range(1, e.n) if e.notes[i].pitch > e.notes[i - 1].pitch])):
        n = len(idx)
        out.append(_U("I-U3", f"count_{d}", "interval", "REL", f"{d}_{_bucket(n, 4)}", idx, [],
                      n, "integer", P.numeric_answer_texts(n, "notes"), fmt={"np": f"notes reached by {'an' if d == 'upward' else 'a'} {d} move from the previous note"}, prov={"direction": d}, stratum=d))
    return out


# ------------------------------------------------------------------------------------------------ BAR STRUCTURE
def g_B(e):
    if not e.usable or e.bar_ambiguous:
        return []
    out = []
    for b in (1, 2):
        idx = e.bars[b]
        sp = [(e.notes[idx[0]].start, e.notes[idx[-1]].end)]
        out.append(_G("B-G1", f"bar_span={b}", "bar_comparison", "ORD", f"bar{b}", idx,
                      [f"the {P.ORD[b]} bar, from the start of its first note to the end of its last note", f"the {P.ORD[b]} bar (everything from its first note's start to its last note's end)"],
                      plural=False, spans=sp))
    for which, pick in (("last", lambda b: e.bars[b][-1]),):           # "first note of each bar" is identical to R-G1 and was dropped
        idx = [pick(1), pick(2)]
        out.append(_G("B-G2", f"{which}_note_of_bars", "bar_comparison", "ORD", which, idx,
                      [(f"the {which} note of each bar", False), (f"the {which} note of both bars", False)]))
    return out


def u_B(e):
    if not e.usable or e.bar_ambiguous:
        return []
    out = []
    for b in (1, 2):
        n = e.bar_note_count(b)
        out.append(_U("B-U1", f"notes_in_bar={b}", "bar_comparison", "ORD", str(n), e.bars[b], [
            "How many notes does the {o} bar contain?", "Count the notes in the {o} bar.", "How many notes are played in the {o} bar?", "What is the number of notes in the {o} bar?",
            "In the {o} bar, how many notes can you hear?", "Give the note count of the {o} bar."], n, "integer", P.numeric_answer_texts(n, "notes"), fmt={"o": P.ORD[b]},
                      stratum=f"bar{e.frame.bar_dur:.2f}"))           # rc8: the bar length (= the clip length) must not predict the count
    d = e.bar_note_count(2) - e.bar_note_count(1)
    out.append(_U("B-U2", "note_count_difference", "bar_comparison", "ORD", str(d), list(range(e.n)), [
        "How many more notes does the second bar have than the first? Answer with a negative number if it has fewer.",
        "Subtract the number of notes in the first bar from the number in the second bar. What do you get?",
        "By how many notes does the second bar exceed the first bar (negative if it has fewer notes)?",
        "Compare the two bars: the second bar has how many notes more than the first (use a negative number if fewer)?",
        "What is the difference in note count between the bars (second bar minus first bar)? It is negative if the second bar has fewer notes.",
        "Count the notes in each bar and subtract the first count from the second. What is the result?"], d, "integer", P.numeric_answer_texts(d, "notes")))
    for which, fn in (("highest", max), ("lowest", min)):
        p1 = fn(e.notes[i].pitch for i in e.bars[1])
        p2 = fn(e.notes[i].pitch for i in e.bars[2])
        i1 = next(i for i in e.bars[1] if e.notes[i].pitch == p1)
        i2 = next(i for i in e.bars[2] if e.notes[i].pitch == p2)
        dd = p2 - p1
        out.append(_U("B-U3", f"{which}_difference", "bar_comparison", "ABS", str(dd), [i1, i2], [
            "By how many semitones is the {w} note of the second bar above the {w} note of the first bar? Use a negative number if it is lower.",
            "Compare the {w} notes of the two bars: how many semitones higher (negative if lower) is the second bar's {w} note?",
            "What is the pitch difference in semitones between the {w} note of bar two and the {w} note of bar one (bar two minus bar one)?",
            "Take the {w} note of each bar. How many semitones does the second one lie above the first (negative if below)?",
            "Measured in semitones, how far above (negative: below) the {w} note of the first bar is the {w} note of the second bar?",
            "Find the {w} pitch in each bar. What is the signed difference in semitones, second bar minus first bar?"], dd, "integer", P.numeric_answer_texts(dd, "semitones"),
                      fmt={"w": which}))
    yes = e.bar_note_count(2) > e.bar_note_count(1)
    out.append(_U("B-U4", "second_bar_more_notes", "bar_comparison", "ORD", "yes" if yes else "no", list(range(e.n)),
                  ["Does the second bar contain more notes than the first bar?", "Is the number of notes in the second bar larger than in the first bar?", "Does bar two have more notes than bar one?",
                   "Are there more notes in the second bar than in the first?"],
                  yes, "boolean", {"0": "yes" if yes else "no", "1": "Yes." if yes else "No.", "2": "Yes, the second bar has more notes." if yes else "No, the second bar does not have more notes."}))
    return out


# ------------------------------------------------------------------------------------------------ HARMONY
HARM_CTX = "text:Each bar is an arpeggio that spells out one major or minor triad."
# rc6: H-U2 measures root motion mod 12 (a pitch class distance); without this, a listener may instead judge the audible
# interval between two specific sounding notes, which differs whenever the root occurs in a different octave in each bar.
ROOT_PC_CTX = "text:Treat the root as a pitch class: ignore the octave it actually sounds in."


def _both_triads(e):
    return e.usable and not e.bar_ambiguous and e.triad_of_bar(1) is not None and e.triad_of_bar(2) is not None


def g_H(e):
    if not _both_triads(e):
        return []
    out = []
    idx = e.notes_not_in_other_triad(2, 1)
    out.append(_G("H-G1", "bar2_not_in_bar1_triad", "harmony", "REL", "empty" if not idx else _bucket(len(idx)), idx,
                  ["the notes of the second bar that do not belong to the triad of the first bar", "the notes in bar two that are not part of bar one's triad"], ctx=(HARM_CTX,)))
    for b in (1, 2):
        for member in ("root", "third", "fifth"):
            idx = e.triad_member_notes(b, member)
            out.append(_G("H-G2", f"triad_{member}_bar={b}", "harmony", "REL", member + ("_empty" if not idx else ""), idx,
                          [f"the {member} notes of the triad in the {P.ORD[b]} bar", f"the notes in the {P.ORD[b]} bar that play the {member} of its triad"], ctx=(HARM_CTX,)))
    return out


def u_H(e):
    if not _both_triads(e):
        return []
    out = []
    for b in (1, 2):
        t = e.triad_name(e.triad_of_bar(b))
        out.append(_U("H-U1", f"triad_of_bar={b}", "harmony", "REL", t, e.bars[b], [
            "Which triad do the notes of the {o} bar outline? Give its root and whether it is major or minor.", "Name the triad spelled out by the {o} bar (root and major/minor).",
            "What triad is arpeggiated in the {o} bar?", "The {o} bar outlines which major or minor triad?", "Identify the triad of the {o} bar, for example 'A minor'.",
            "Which major or minor triad do the notes of the {o} bar spell out?"],
                      t, "class_label", {"0": t, "1": f"{t}.", "2": f"The {P.ORD[b]} bar outlines {_art(t)} {t} triad."}, fmt={"o": P.ORD[b]}, ctx=(HARM_CTX,)))
    m = e.root_motion()
    out.append(_U("H-U2", "root_motion_up", "harmony", "REL", str(m), list(range(e.n)), [
        "By how many semitones does the root of the triad move upward from the first bar to the second bar? Answer 0 if the root stays the same (count upward, 0 to 11).",
        "Going from the first bar's triad to the second bar's triad, how many semitones up does the root move (0 to 11)?",
        "How many semitones separate the root of bar one from the root of bar two, counted upward from bar one (0 to 11)?",
        "Counting upward from the first bar's root, how many semitones is the root of the second bar's triad (0 to 11)?",
        "How large is the upward root motion, in semitones, from bar one's triad to bar two's triad (0 if the root does not change)?",
        "Starting from the root of the first bar, how many semitones up do you have to go to reach the root of the second bar (0 to 11)?"], m, "integer", P.numeric_answer_texts(m, "semitones"),
                      ctx=(HARM_CTX, ROOT_PC_CTX)))
    same = e.triad_of_bar(1) == e.triad_of_bar(2)
    out.append(_U("H-U3", "same_triad", "harmony", "REL", "yes" if same else "no", list(range(e.n)), ["Do the two bars outline the same triad?", "Is the triad of the second bar the same as the triad of the first bar?", "Are both bars built on the same triad?",
                   "Does the second bar spell out the same triad as the first bar?"],
                  same, "boolean", {"0": "yes" if same else "no", "1": "Yes." if same else "No.", "2": "Yes, both bars outline the same triad." if same else "No, the two bars outline different triads."}, ctx=(HARM_CTX,)))
    for b in (1, 2):
        n = e.drops_to_lowest(b)
        out.append(_U("A-U1", f"drops_to_lowest_bar={b}", "harmonic_pattern", "PAT", str(n), [i for a, i in zip(e.bars[b], e.bars[b][1:]) if e.notes[i].pitch < e.notes[a].pitch and e.notes[i].pitch == min(e.notes[j].pitch for j in e.bars[b])], [
            "How many times does the melody in the {o} bar move down to the lowest note of that bar (from a higher note)?", "In the {o} bar, count how often the melody falls back to that bar's lowest pitch from a higher note.",
            "How many downward moves in the {o} bar end on the lowest note of the bar?",
            "Count the times the melody of the {o} bar steps down onto that bar's lowest pitch.", "In the {o} bar, how many times does the line return to its lowest note from above?",
            "How often does the {o} bar drop back to its lowest note?"], n, "integer", P.numeric_answer_texts(n, "times"), fmt={"o": P.ORD[b]}, ctx=(HARM_CTX,), stratum=f"meter{e.frame.meter}"))
    return out


# ------------------------------------------------------------------------------------------------ HARMONIC PATTERN (arpeggio shape)
# rc10: the paper's 7th concept ("harmonic pattern") was covered by a single understanding family; these add the grounding side and a second understanding family.
# They are about the SHAPE of the arpeggio (where it starts over, how long one repetition is, whether it falls back), not about chord tones (that is harmony / H-G2).
def g_A(e):
    if not _both_triads(e):
        return []
    out = []
    for b in (1, 2):
        idx = e.drop_back_notes(b)
        out.append(_G("A-G1", f"drop_back_bar={b}", "harmonic_pattern", "PAT", "empty" if not idx else _bucket(len(idx)), idx,
                      [f"the notes where the melody in the {P.ORD[b]} bar falls back to that bar's lowest note", f"the notes of the {P.ORD[b]} bar that step down onto the bar's lowest pitch",
                       f"the notes at which the arpeggio of the {P.ORD[b]} bar starts over from its lowest note"],
                      ctx=(HARM_CTX,), prov={"bar": b}, stratum=f"meter{e.frame.meter}"))
        cyc = e.arpeggio_first_cycle(b)
        if cyc is not None:
            out.append(_G("A-G2", f"first_cycle_bar={b}", "harmonic_pattern", "PAT", e.arpeggio_kind(b), cyc,
                          [f"the first complete repetition of the arpeggio pattern in the {P.ORD[b]} bar (from its first note to the note before the pattern starts over)",
                           f"one complete run of the arpeggio pattern at the start of the {P.ORD[b]} bar, before it repeats"],
                          plural=False, spans=[(e.notes[cyc[0]].start, e.notes[cyc[-1]].end)], ctx=(HARM_CTX,), prov={"bar": b}))
    return out


def u_A(e):
    if not _both_triads(e):
        return []
    out = []
    for b in (1, 2):
        k = e.arpeggio_kind(b)
        if k is None:
            continue
        lab = "ascending" if k == "ascending" else "up-and-down"
        out.append(_U("A-U2", f"pattern_bar={b}", "harmonic_pattern", "PAT", k, e.bars[b], [
            "Is the arpeggio in the {o} bar ascending (it rises and then jumps back to the lowest note) or up-and-down (it rises and then falls back)? Answer 'ascending' or 'up-and-down'.",
            "Does the {o} bar follow an ascending arpeggio (rise, then jump back down) or an up-and-down arpeggio (rise, then fall back)? Answer 'ascending' or 'up-and-down'.",
            "What shape does the arpeggio of the {o} bar have: 'ascending' (rise, then jump back to the lowest note) or 'up-and-down' (rise, then fall back)?",
            "In the {o} bar, does the arpeggio rise and jump back ('ascending') or rise and fall back ('up-and-down')?",
            "Name the arpeggio shape of the {o} bar: 'ascending' (jumps back to the bottom after the top) or 'up-and-down' (comes back down step by step).",
            "Which shape does the {o} bar arpeggio have, 'ascending' or 'up-and-down'? Ascending rises and jumps back, up-and-down rises and falls back."],
                      lab, "class_label", {"0": lab, "1": f"{lab.capitalize()}.", "2": f"The arpeggio of the {P.ORD[b]} bar is {lab}."}, fmt={"o": P.ORD[b]}, ctx=(HARM_CTX,)))
    return out


# ------------------------------------------------------------------------------------------------ REPETITION
def g_Rp(e):
    if not e.usable or e.bar_ambiguous:
        return []
    diff = e.bar2_notes_differing()
    if diff is None:
        return []
    same = [j for j in e.bars[2] if j not in set(diff)]
    return [
        _G("Rp-G1", "bar2_pitch_differs", "repetition", "PAT", "empty" if not diff else _bucket(len(diff)), diff,
           ["the notes of the second bar whose pitch differs from the note at the same position in the first bar", "the notes in bar two that do not have the same pitch as the corresponding note of bar one"],
           ctx=("text:Compare the notes of the two bars position by position (first with first, second with second, and so on).",)),
        _G("Rp-G2", "bar2_pitch_same", "repetition", "PAT", "empty" if not same else _bucket(len(same)), same,
           ["the notes of the second bar that have the same pitch as the note at the same position in the first bar", "the notes in bar two that repeat the pitch of the corresponding note in bar one"],
           ctx=("text:Compare the notes of the two bars position by position (first with first, second with second, and so on).",)),
    ]


def u_Rp(e):
    if not e.usable or e.bar_ambiguous:
        return []
    bs = e.bar_shift()
    if bs is None:
        return []
    out = []
    shift, ndiff = bs
    ctx = ("text:The two bars have the same number of notes; pair them up in order (first note with first note, second with second, and so on).",)
    out.append(_U("Rp-U2", "n_pitch_differences", "repetition", "PAT", str(ndiff), e.bar2_notes_differing() or [], [
        "How many notes of the second bar have a different pitch from the note they are paired with in the first bar?", "Comparing the bars note by note, how many notes of the second bar differ in pitch from their partner?",
        "Count the notes of the second bar whose pitch differs from that of the note they are paired with.", "How many notes in bar two do not match the pitch of their partner in bar one?",
        "How many of the second bar's notes differ in pitch from their partner in the first bar?", "How many notes of the second bar are at a different pitch than their partners in the first bar?"],
                  ndiff, "integer", P.numeric_answer_texts(ndiff, "notes"), ctx=ctx))
    if shift is not None:
        out.append(_U("Rp-U1", "transposition", "repetition", "PAT", str(shift), list(range(e.n)), [
            "By how many semitones is the second bar shifted relative to the first bar? Answer 0 if the pitches are identical; use a negative number if it is lower.",
            "The second bar repeats the first bar at a different pitch level. How many semitones higher (negative if lower) is it? Answer 0 if it is at the same level.",
            "What is the pitch shift, in semitones, from the first bar to the second bar (0 if there is none, negative for downward)?",
            "Compared with the first bar, how many semitones higher (negative: lower) is the second bar? Answer 0 if it is the same.",
            "How many semitones is the second bar transposed relative to the first (0 = identical pitches, negative = downward)?",
            "If the second bar is a shifted copy of the first, by how many semitones is it shifted (0 if not shifted, negative if lowered)?"],
                      shift, "integer", P.numeric_answer_texts(shift, "semitones"), ctx=ctx))
    return out


GEN: List[Callable] = [g_R_G1, g_R_G2, g_R_G3, g_R_G4, g_R_G5, g_R_G6, g_T_G1, g_I, g_B, g_H, g_A, g_Rp,
                       u_R_U3, u_R_U4, u_R_U6, u_T, u_I, u_B, u_H, u_A, u_Rp]


# families whose question mentions a bar without a meter sentence: the listener has to be told where the bars are (rc4, from the listening check).
# The statement is RELATIVE ("two equal bars from the first note's start to the last note's end") in every one of these families, grounding and
# understanding alike, and never an absolute time: a stated start time would give away the first answer span (a fixed-length guess from the start
# time hits IoU >= 0.5 on 77 % of the R-G1 items), and even for understanding families the author decided (rc7) that the boundary must be found by
# the solver, not handed over. (rc6 briefly stated the bar-2 boundary time for the understanding families; it rested on a misread of the ratings
# and was reverted, see SPEC sec.11.) The boundary is derivable: the midpoint of first-note start and last-note end is within 13 ms (median) /
# 81 ms (max) of the first onset of bar 2.
BAR_FAMS = {"R-G1", "R-G3", "R-G6", "B-G1", "B-G2", "H-G1", "H-G2", "A-G1", "A-G2", "Rp-G1", "Rp-G2",
            "R-U6", "B-U1", "B-U2", "B-U3", "B-U4", "H-U1", "H-U2", "H-U3", "A-U1", "A-U2", "Rp-U1", "Rp-U2"}


def candidates(e: Excerpt) -> Dict[str, List[Cand]]:
    """all candidates of one excerpt grouped by family key"""
    by = defaultdict(list)
    for g in GEN:
        for c in g(e):
            if c.fam in BAR_FAMS:
                if not e.frame_full:              # the frame sentence would not be true
                    continue
                c.ctx = ("frame",) + tuple(c.ctx)
            by[c.fam].append(c)
    return by
