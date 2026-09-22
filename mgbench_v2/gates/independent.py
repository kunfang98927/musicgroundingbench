"""Independent re-derivation of every v2 answer from the released JSON alone (notes + frame + stored parameters).

It deliberately does not import oracle2b: positions are measured against the *nominal* bar frame (absolute time), whereas the oracle
measures against each bar's first onset. Agreement between the two conventions is part of the release test.
Each checker returns the expected value: grounding -> set of note indices, understanding -> gold answer (same type as stored).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
NAMES = list(PC)
VALUE_BEATS = (0.5, 1.0, 1.5, 2.0, 3.0)
VALUE_NAME = {False: {0.5: "eighth note", 1.0: "quarter note", 1.5: "dotted quarter note", 2.0: "half note", 3.0: "dotted half note"},
              True: {0.5: "sixteenth note", 1.0: "eighth note", 1.5: "dotted eighth note", 2.0: "quarter note", 3.0: "dotted quarter note"}}


class View:
    def __init__(self, rec):
        n = sorted(rec["notes"], key=lambda x: (x["start"], x["end"]))
        self.rec = rec
        f = rec["v2"]["frame"]
        self.meter, self.bpm, self.t0, self.bar_dur = f["meter"], f["tempo_bpm"], f["music_start"], f["bar_dur"]
        self.spb = 60.0 / self.bpm
        self.bpb = 6 if self.meter == "6/8" else int(self.meter.split("/")[0])
        self.st = [x["start"] for x in n]; self.en = [x["end"] for x in n]; self.p = [x["pitch"] for x in n]
        self.N = len(n)
        self.bar = [int((s - self.t0 + 0.10) // self.bar_dur) + 1 for s in self.st]
        self.pos = [(s - self.t0 - (b - 1) * self.bar_dur) / self.spb for s, b in zip(self.st, self.bar)]
        self.grid = [round(x * 2) / 2 for x in self.pos]
        self.dur = [(e - s) / self.spb for s, e in zip(self.st, self.en)]
        # rc3 note length = distance to the next onset; the last note lasts up to the end of bar 2. Only meaningful when no rest follows a note
        # and the last note ends (within 60 ms) where bar 2 ends -> every length-based family requires rest_free
        end = self.t0 + 2 * self.bar_dur
        self.ioi = [((self.st[i + 1] if i + 1 < self.N else end) - self.st[i]) / self.spb for i in range(self.N)]
        self.frame_full = self.N > 0 and abs(self.st[0] - self.t0) <= 0.10 and abs(end - self.en[-1]) <= 0.06      # "two equal bars from the first note's start to the last note's end"
        self.rest_free = self.N > 1 and all(self.st[i + 1] - self.en[i] <= 0.06 for i in range(self.N - 1)) and abs(end - self.en[-1]) <= 0.06

    def value(self, i):
        v = min(VALUE_BEATS, key=lambda c: abs(c - self.ioi[i]))
        return v if abs(v - self.ioi[i]) <= 0.2 else None

    def in_bar(self, b):
        return [i for i in range(self.N) if self.bar[i] == b]

    def spans_to_idx(self, spans_sec):
        out = []
        for a, b in spans_sec:
            hit = [i for i in range(self.N) if abs(self.st[i] - a) < 2e-5 and abs(self.en[i] - b) < 2e-5]
            out.append(hit[0] if hit else None)
        return out


# strength of the half-beat positions of a bar (own copy of the table; every other position, i.e. every off-beat, has strength 1)
_STRENGTH = {"4/4": {0: 4, 4: 3, 2: 2, 6: 2}, "3/4": {0: 3, 2: 2, 4: 2}, "2/4": {0: 3, 2: 2}}


def _syncopated(v: View) -> List[int]:
    """a note that starts at a weaker position and is still sounding (> 0.225 beat: the midpoint of the oracle's 0.10 / 0.35 band) when a stronger position is reached.
    Times are measured against the nominal frame (absolute time), the oracle measures against the first onset of each bar."""
    H = 2 * v.bpb
    st = _STRENGTH[v.meter]
    out = []
    for i in range(v.N):
        h0 = int(round(v.grid[i] * 2))
        a = (v.bar[i] - 1) * H + h0
        t_next = v.st[i + 1] if i + 1 < v.N else v.t0 + 2 * v.bar_dur
        w0 = st.get(h0, 1)
        for s in range(a + 1, 2 * H + 1):
            if st.get(s % H, 1) <= w0:
                continue
            if (t_next - (v.t0 + s * v.spb / 2)) / v.spb > 0.225:
                out.append(i)
                break
    return out


def _ord(s: str) -> int:
    return {"first": 1, "second": 2}[s]


def expected(v: View, rec) -> Dict:
    """returns {'kind': 'idx'|'gold'|'span', 'value': ...} or {'kind': 'skip'}"""
    fam, prm, sub = rec["v2"]["family"], rec["v2"]["params"], rec["v2"]["sub"]
    N, p, g, bpb = v.N, v.p, v.grid, v.bpb
    idx = lambda L: {"kind": "idx", "value": sorted(L)}
    gold = lambda x: {"kind": "gold", "value": x}
    mv = [None] + [p[i] - p[i - 1] for i in range(1, N)]
    bar_of_sub = lambda: int(sub.split("=")[-1])
    if fam == "R-G1":
        return idx([i for i in range(N) if abs(v.st[i] - (v.t0 + (v.bar[i] - 1) * v.bar_dur)) <= 0.10])
    if fam == "R-G2":
        return idx([i for i in range(N) if g[i] == prm["k"] - 1])
    if fam == "R-G3":
        return idx([i for i in range(N) if g[i] == bpb / 2])
    if fam in ("R-G4", "R-G5", "R-G6", "R-U3") and not v.rest_free:
        return {"kind": "skip"}
    if fam == "R-U4" and prm.get("ref_kind") == "longest" and not v.rest_free:
        return {"kind": "skip"}
    if fam == "R-G4":
        return idx([i for i in range(N) if v.value(i) == prm["value_beats"]])
    if fam == "R-G5":
        if v.meter not in ("2/4", "3/4", "4/4"):
            return {"kind": "skip"}
        return idx(_syncopated(v))
    if fam == "R-G6":
        b = bar_of_sub(); L = sorted(v.in_bar(b), key=lambda i: -v.ioi[i])
        return idx([L[0]]) if len(L) > 1 and v.ioi[L[0]] >= prm["min_ratio"] * v.ioi[L[1]] else {"kind": "skip"}
    if fam == "R-U1":
        return gold(bpb)
    if fam == "R-U2":
        return gold(v.bpm)
    if fam == "R-U3":
        return gold(VALUE_NAME[v.meter == "6/8"].get(v.value(prm["note_index"]), "UNMEASURABLE"))
    if fam == "R-U4":
        return gold(g[prm["note_index"]] + 1.0)
    if fam == "R-U5":
        vals = [v.value(i) for i in range(N)]
        return gold("UNMEASURABLE" if (None in vals or not vals) else int(round(bpb / min(vals))))
    if fam == "R-U6":
        return gold(sum(1 for i in v.in_bar(1) if g[i] >= bpb / 2))
    if fam == "T-G1":
        s = {PC[x.strip()] for x in prm["set"].split(",")}
        return idx([i for i in range(N) if p[i] % 12 not in s])
    if fam in ("T-U1", "T-U2", "T-U5"):
        s = {PC[x.strip()] for x in prm["set"].split(",")}
        out = [i for i in range(N) if p[i] % 12 not in s]
        if fam == "T-U1":
            return gold(len(out))
        if fam == "T-U5":
            return gold(len(out) == 0)
        return gold([NAMES[c] for c in sorted(s - {x % 12 for x in p})])
    if fam == "T-U3":
        return gold(len({x % 12 for x in p}))
    if fam == "T-U4":
        return gold([NAMES[c] for c in sorted({x % 12 for x in p})])
    if fam in ("I-G1", "I-G2", "I-G3"):
        lo, hi = prm["lo"], prm["hi"]
        return idx([i for i in range(1, N) if abs(mv[i]) >= lo and (hi is None or abs(mv[i]) <= hi)])
    if fam == "I-G4":
        s, d = prm["semitones"], prm["direction"]
        return idx([i for i in range(1, N) if mv[i] == (s if d == "ascending" else -s)])
    if fam == "I-G5":
        return idx([i for i in range(1, N) if mv[i] == 0])
    if fam == "I-G6":
        ps = sorted(set(p), reverse=(prm["which"] == "highest"))
        return idx([i for i in range(N) if p[i] == ps[prm["rank"] - 1]]) if prm["rank"] <= len(ps) else idx([])
    if fam == "I-G7":
        return idx([i for i in range(1, N) if (mv[i] > 0 if prm["direction"] == "upward" else mv[i] < 0)])
    if fam == "I-U1":
        return gold(sum(1 for i in range(1, N) if abs(mv[i]) == prm["semitones"]))
    if fam == "I-U2":
        return gold(max(abs(x) for x in mv[1:]))
    if fam == "I-U3":
        return gold(sum(1 for i in range(1, N) if (mv[i] > 0 if prm["direction"] == "upward" else mv[i] < 0)))
    if fam == "B-G1":
        L = v.in_bar(bar_of_sub())
        return {"kind": "span", "value": (v.st[L[0]], v.en[L[-1]])}
    if fam == "B-G2":
        L1, L2 = v.in_bar(1), v.in_bar(2)
        return idx([L1[0], L2[0]] if sub.startswith("first") else [L1[-1], L2[-1]])
    if fam == "B-U1":
        return gold(len(v.in_bar(bar_of_sub())))
    if fam == "B-U2":
        return gold(len(v.in_bar(2)) - len(v.in_bar(1)))
    if fam == "B-U3":
        f = max if sub.startswith("highest") else min
        return gold(f(p[i] for i in v.in_bar(2)) - f(p[i] for i in v.in_bar(1)))
    if fam == "B-U4":
        return gold(len(v.in_bar(2)) > len(v.in_bar(1)))
    if fam in ("H-G1", "H-G2", "H-U1", "H-U2", "H-U3", "A-U1", "A-G1", "A-G2", "A-U2"):
        return _harmony(v, rec, fam, sub)
    if fam in ("Rp-G1", "Rp-G2", "Rp-U1", "Rp-U2"):
        b1, b2 = v.in_bar(1), v.in_bar(2)
        if len(b1) != len(b2):
            return {"kind": "skip"}
        d = [p[j] - p[i] for i, j in zip(b1, b2)]
        if fam == "Rp-G1":
            return idx([j for (i, j), x in zip(zip(b1, b2), d) if x != 0])
        if fam == "Rp-G2":
            return idx([j for (i, j), x in zip(zip(b1, b2), d) if x == 0])
        if fam == "Rp-U2":
            return gold(sum(1 for x in d if x != 0))
        return gold(d[0]) if all(x == d[0] for x in d) else {"kind": "skip"}
    return {"kind": "skip"}


def _triad(v: View, b: int):
    pcs = [v.p[i] % 12 for i in v.in_bar(b)]
    if len(pcs) < 3:
        return None
    hits = []
    for r in range(12):
        for q, iv in (("major", (0, 4, 7)), ("minor", (0, 3, 7))):
            t = {(r + x) % 12 for x in iv}
            if t <= set(pcs) and sum(1 for x in pcs if x in t) >= 0.9 * len(pcs):
                hits.append((r, q))
    return hits[0] if len(hits) == 1 else None


def _harmony(v, rec, fam, sub):
    p = v.p
    t1, t2 = _triad(v, 1), _triad(v, 2)
    idx = lambda L: {"kind": "idx", "value": sorted(L)}
    gold = lambda x: {"kind": "gold", "value": x}
    if t1 is None or t2 is None:
        return {"kind": "skip"}
    tri = lambda t: {(t[0] + x) % 12 for x in ((0, 4, 7) if t[1] == "major" else (0, 3, 7))}
    b = int(sub.split("=")[-1]) if "bar=" in sub else None
    if fam == "H-U1":
        t = (t1, t2)[b - 1]
        return gold(f"{NAMES[t[0]]} {t[1]}")
    if fam == "H-U2":
        return gold((t2[0] - t1[0]) % 12)
    if fam == "H-U3":
        return gold(t1 == t2)
    if fam == "H-G1":
        return idx([i for i in v.in_bar(2) if p[i] % 12 not in tri(t1)])
    if fam == "H-G2":
        member = sub.split("_")[1]
        t = (t1, t2)[b - 1]
        off = {"root": 0, "third": 4 if t[1] == "major" else 3, "fifth": 7}[member]
        return idx([i for i in v.in_bar(b) if p[i] % 12 == (t[0] + off) % 12])
    if fam == "A-U1":
        L = v.in_bar(b)
        lo = min(p[i] for i in L)
        return gold(sum(1 for a, c in zip(L, L[1:]) if p[c] < p[a] and p[c] == lo))
    if fam == "A-G1":
        L = v.in_bar(b)
        lo = min(p[i] for i in L)
        return idx([c for a, c in zip(L, L[1:]) if p[c] < p[a] and p[c] == lo])
    if fam in ("A-G2", "A-U2"):
        L = v.in_bar(b)
        pit = sorted({p[i] for i in L})
        if len(L) < 4 or len(pit) != 3:
            return {"kind": "skip"}
        r = [pit.index(p[i]) for i in L]
        kind = "ascending" if r == [(0, 1, 2)[k % 3] for k in range(len(r))] else ("updown" if r == [(0, 1, 2, 1)[k % 4] for k in range(len(r))] else None)
        if kind is None:
            return {"kind": "skip"}
        if fam == "A-U2":
            return gold("ascending" if kind == "ascending" else "up-and-down")
        c = 3 if kind == "ascending" else 4
        return {"kind": "span", "value": (v.st[L[0]], v.en[L[c - 1]])}
    return {"kind": "skip"}
