"""Question / answer wording for MGBench-v2 (2B).

Every question is composed from independent pools: a context sentence (optional), a stem, and a noun phrase for what is asked.
Stem choice is independent of the answer (G4). Answer formats stay the v1 formats so the existing span parsers keep working:
style 1 = "From 1.04 seconds to 1.64 seconds; ..." (default), style 0 = "[1.04s, 1.64s]; ...", style 2 = a sentence.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
NUM_WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            11: "eleven", 12: "twelve"}

# ---------------------------------------------------------------- grounding stems ({do}/{occur}/{is} agree with the number of {what})
G_STEMS = [
    "Locate {what}.",
    "Find the time span(s) of {what}.",
    "Which time interval(s) contain {what}?",
    "When {do} {what} {occur}?",
    "Mark the time span(s) covered by {what}.",
    "Give the start and end times of {what}.",
    "Where in the audio {is} {what}?",
    "Identify {what} and give the time span(s).",
]


def agree(plural: bool) -> Dict[str, str]:
    return {"do": "do" if plural else "does", "occur": "occur", "is": "are" if plural else "is"}


def render_grounding_question(rng, what: str, plural: bool, ctx_sentences: Sequence[str]) -> Tuple[str, List[str]]:
    """(question, three paraphrase variants that share the chosen context)"""
    ag = agree(plural)
    stems = rng.sample(G_STEMS, 3)
    qs = [s.format(what=what, **ag) for s in stems]
    return _with_context(rng, qs, ctx_sentences)


def _with_context(rng, qs: List[str], ctx_sentences: Sequence[str]):
    ctx = " ".join(ctx_sentences).strip()
    if not ctx:
        return qs[0], qs
    mode = rng.random()
    out = [(f"{ctx} {q}" if mode < 0.6 else f"{q} {ctx}") for q in qs]
    return out[0], out


# ---------------------------------------------------------------- context sentences
def meter_sentence(rng, ts: str, with_unit: bool) -> str:
    """meter + the bar frame: the excerpt is exactly two bars and its first note is beat 1 of bar 1 (true for every excerpt that gets a meter sentence)"""
    unit = "an eighth note" if ts == "6/8" else "a quarter note"
    core = rng.choice([f"This is a two-bar piece in {ts} that starts on beat 1 of bar 1", f"The music is in {ts}; its first note is beat 1 of bar 1",
                       f"Assume the time signature is {ts} and that the first note falls on beat 1 of the first bar", f"This two-bar excerpt is in {ts} and begins on the first beat of bar 1"])
    if with_unit:
        return f"{core} (one beat is {unit})."
    return core + "."


_METRIC = {"4/4": ["beat 1, beat 3, beats 2 and 4, and the halves between beats", "beat 1 (strongest), then beat 3, then beats 2 and 4, then the halves between beats (weakest)"],
           "3/4": ["beat 1, beats 2 and 3, and the halves between beats", "beat 1 (strongest), then beats 2 and 3, then the halves between beats (weakest)"],
           "2/4": ["beat 1, beat 2, and the halves between beats", "beat 1 (strongest), then beat 2, then the halves between beats (weakest)"]}


def metric_sentence(rng, ts: str) -> str:
    """strength of the positions of the bar (needed to say what a syncopated note is)"""
    if rng.random() < 0.5:
        return f"In {ts} the positions of a bar rank from strongest to weakest as {_METRIC[ts][0]}."
    return f"In {ts} the strength of the positions is: {_METRIC[ts][1]}."


def frame_sentence(rng) -> str:
    """where the bars are: two equally long bars filling the excerpt from the first note's start to the last note's end (true whenever `frame_full`)"""
    return rng.choice(["The excerpt consists of two equally long bars: the first begins with the first note and the second ends where the last note ends.",
                       "This is a two-bar excerpt with two equally long bars; bar 1 starts at the first note and bar 2 ends with the last note.",
                       "The clip holds exactly two equally long bars, from the start of the first note to the end of the last note.",
                       "Two equally long bars fill the excerpt, from the first note's start to the last note's end."])


def tempo_sentence(rng, bpm: int) -> str:
    exact = rng.random() < 0.5
    if exact:
        return rng.choice([f"The tempo is {bpm} beats per minute.", f"The tempo is {bpm} bpm.", f"The music plays at {bpm} bpm."])
    return rng.choice([f"The tempo is around {bpm} bpm.", f"The tempo is roughly {bpm} beats per minute.", f"The music plays at about {bpm} bpm."])


# ---------------------------------------------------------------- answers
def _t(x: float) -> str:
    return f"{x:.2f}"


def grounding_answer_texts(spans_sec: Sequence[Tuple[float, float]], what: str, plural: bool) -> Dict[str, str]:
    if not spans_sec:
        return {"0": "none", "1": "There is no such note in this clip.", "2": "No matching note was found in this clip."}
    s0 = "; ".join(f"[{_t(a)}s, {_t(b)}s]" for a, b in spans_sec)
    s1 = "; ".join(f"From {_t(a)} seconds to {_t(b)} seconds" for a, b in spans_sec) + "."
    s2 = f"{what[0].upper() + what[1:]} {'are' if plural else 'is'} located in the following intervals: " + "; ".join(
        f"from {_t(a)} seconds to {_t(b)} seconds" for a, b in spans_sec) + "."
    return {"0": s0, "1": s1, "2": s2}


def numeric_answer_texts(n, unit: str = "") -> Dict[str, str]:
    """three phrasings of a numeric answer (digits alone / short phrase / sentence)"""
    if unit and abs(n) == 1 and unit.endswith("s") and unit != "bpm":
        unit = unit[:-1]                                   # 1 note, 1 beat, 1 semitone, 1 time ...
    u = f" {unit}" if unit else ""
    return {"0": f"{n}", "1": f"{n}{u}".strip(), "2": f"The answer is {n}{u}."}


COUNT_ASKS = [
    "How many {np} are there?",
    "Count the {np}.",
    "What is the number of {np}?",
    "Determine how many {np} the clip contains.",
    "Give the number of {np}.",
    "In this clip, how many {np} can you hear?",
]


def render_count_question(rng, np_: str, ctx_sentences: Sequence[str]):
    qs = [a.format(np=np_) for a in rng.sample(COUNT_ASKS, 3)]
    return _with_context(rng, qs, ctx_sentences)


def render_generic_question(rng, asks: Sequence[str], ctx_sentences: Sequence[str], **kw):
    qs = [a.format(**kw) for a in rng.sample(list(asks), min(3, len(asks)))]
    return _with_context(rng, qs, ctx_sentences)
