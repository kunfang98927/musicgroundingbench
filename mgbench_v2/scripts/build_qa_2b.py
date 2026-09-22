"""WP4: build the v2 QA layer for MGBench-2B from the released MIDI files (audio / MERT features are reused unchanged).

python -m mgbench_v2.scripts.build_qa_2b --out /path/to/out [--splits train val test] [--seed 20260919]
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

from mgbench_v2 import phrasing as P
from mgbench_v2.oracle2b import Excerpt, Note, frame_from_basic_info, load_notes
from mgbench_v2.qa2b import EMPTY_CAP, FPS, ORACLE_VERSION, YESNO_CAP, Cand, candidates

MIDI_ROOT = "/scratch/kunfang/two_bar_dataset"
META_ROOT = "/scratch/kunfang/mgbench_regen_check/two_bar_dataset"


# ------------------------------------------------------------------------------------------------ selection
# rc10: inside a stratum a class with fewer than TINY_CLASS_SHARE of the stratum's items only counts towards the number of classes (old behaviour); with DROP_TINY_CLASSES it
# is removed from the stratum instead. Before, a 3 % class of the train 4/4 stratum of A-U1 forced the cap of the two big classes down to 51 items while val/test (which have
# no such class in 4/4) were not trimmed: TV(train, test) = 0.21. rc11 default: 5 % floor, tiny classes dropped (A-U1 0.21 -> 0.04, I-U3 0.13 -> 0.02, R-U4 0.11 -> 0.03).
TINY_CLASS_SHARE = 0.05
DROP_TINY_CLASSES = True


def max_share(k: int, boolean: bool = False, stratum: bool = False) -> float:
    if boolean:
        return 0.55
    if stratum and k >= 3:
        return 1.1 / k                 # inside a stratum the answer must be (nearly) independent of the question parameter / context
    return max(0.35, 1.5 / max(k, 1))


def waterfill_cap(counts, share: float, total_cap=None):
    """largest per-class cap c with c <= share * sum_k min(n_k, c); optionally also sum_k min(n_k, c) <= total_cap"""
    ns = sorted(counts.values())
    hi = ns[-1] if ns else 0
    best = 1
    for c in range(hi, 0, -1):
        tot = sum(min(n, c) for n in ns)
        if c <= share * tot + 1e-9 and (total_cap is None or tot <= total_cap):
            best = c
            break
    return best


# rc10: how many items of one family a clip may contribute (different sub-questions). Only families of the scarce concepts (harmony, harmonic pattern) need more
# than one; everything else keeps the one-item-per-clip rule.
SLOTS = {"H-G2": 3, "H-U1": 2, "A-G1": 2, "A-G2": 2, "A-U1": 2, "A-U2": 2}


def waterfill_alloc(caps, total):
    """split `total` over the keys of `caps` as equally as possible without exceeding any cap (integers)"""
    alloc = {k: 0 for k in caps}
    active = [k for k in sorted(caps) if caps[k] > 0]
    left = total
    while active and left > 0:
        share = left // len(active)
        if share == 0:                                   # fewer items left than keys: one each for the keys with the most room
            for k in sorted(active, key=lambda k: (-(caps[k] - alloc[k]), k))[:left]:
                alloc[k] += 1
            break
        small = [k for k in active if caps[k] - alloc[k] <= share]
        if small:
            for k in small:
                left -= caps[k] - alloc[k]
                alloc[k] = caps[k]
            active = [k for k in active if k not in small]
        else:
            for k in active:
                alloc[k] += share
            left -= share * len(active)
    return alloc


def concept_quota(sel, mode, train_concept_cap, rng_seed):
    """rc10: give every music concept the same number of items (mode 'equal': val/test) or at most `train_concept_cap` items (mode 'soft': train), separately for
    grounding and understanding. Inside a concept the families share the quota as equally as their capacity allows; the yes/no items of a concept stay <= YESNO_CAP.
    Every family is thinned with `proportional_subsample`, i.e. uniformly inside every (stratum, answer class) cell, so the balance rules that keep the questions free of
    shortcuts are not touched. Returns a report."""
    by = defaultdict(lambda: defaultdict(list))                    # (kind, concept) -> family -> keys
    for fam, d in sel.items():
        if d:
            c0 = next(iter(d.values()))
            by[(c0.kind, c0.concept)][fam] = c0
    info = {}
    cap_of = {}
    for (kind, con), fams in by.items():
        N = sum(len(sel[f]) for f, c in fams.items() if c.gtype != "boolean")
        B = sum(len(sel[f]) for f, c in fams.items() if c.gtype == "boolean")
        cap_of[(kind, con)] = N + min(B, N // 9) if kind == "understanding" else N
    kinds = {k for k, _ in by}
    rep = {}
    for kind in sorted(kinds):
        cons = sorted(c for k, c in by if k == kind)
        if mode == "equal":
            T = {c: min(cap_of[(kind, x)] for x in cons) for c in cons}
        else:
            T = {c: min(cap_of[(kind, c)], train_concept_cap) for c in cons}
        for con in cons:
            fams = by[(kind, con)]
            nb = {f: len(sel[f]) for f, c in fams.items() if c.gtype != "boolean"}
            bl = {f: len(sel[f]) for f, c in fams.items() if c.gtype == "boolean"}
            b_total = min(sum(bl.values()), int(T[con] * YESNO_CAP)) if bl else 0
            alloc = waterfill_alloc(nb, T[con] - b_total)
            alloc.update(waterfill_alloc(bl, b_total) if bl else {})
            for f, n in alloc.items():
                if n < len(sel[f]):
                    sel[f] = proportional_subsample(sel[f], n, random.Random(f"{rng_seed}|quota|{f}"))
            rep[f"{kind}/{con}"] = {"capacity": cap_of[(kind, con)], "target": T[con], "families": {f: len(sel[f]) for f in sorted(fams)}}
    return rep


def proportional_subsample(cur, target, rng):
    """rc8: keep exactly `target` items, drawn uniformly inside every (stratum, answer class) cell, so that the class and stratum marginals are those of
    `cur`. (Before rc8 the per-family train cap was applied by lowering the per-class cap, which flattened the train answer distribution while val/test,
    which have no cap, kept the natural mode up to the share cap: TV(train, test) up to 0.34.)"""
    n = len(cur)
    if target >= n:
        return cur
    cells = defaultdict(list)
    for sid, c in cur.items():
        cells[(c.stratum, c.akey)].append(sid)
    quota = {k: len(v) * target / n for k, v in cells.items()}
    base = {k: int(q) for k, q in quota.items()}
    for k in sorted(cells, key=lambda k: (-(quota[k] - base[k]), str(k)))[:target - sum(base.values())]:      # largest remainder: the total is exactly `target`
        base[k] += 1
    keep = {}
    for k, v in sorted(cells.items(), key=lambda kv: str(kv[0])):
        v = sorted(v)
        rng.shuffle(v)
        for sid in v[:base[k]]:
            keep[sid] = cur[sid]
    return keep


def select_family(by_clip, rng, total_cap=None):
    """by_clip: sid -> [Cand]; returns sid -> Cand (<= 1 per clip), balanced over Cand.akey; `total_cap` (train only) is applied last, proportionally"""
    sids = sorted(by_clip)
    rng.shuffle(sids)
    counts, chosen, n_empty = Counter(), {}, 0
    for sid in sids:
        opts = by_clip[sid]
        def key(c):
            pen = 1e9 if (c.empty and n_empty >= EMPTY_CAP * (len(chosen) + 1)) else 0
            return (pen + counts[(c.stratum, c.akey)], rng.random())
        c = min(opts, key=key)
        chosen[sid] = c
        counts[(c.stratum, c.akey)] += 1
        n_empty += int(c.empty)
    # --- trim to the class-share rule, first inside every stratum (question parameter / context variable), then over the family
    understanding = bool(chosen) and next(iter(chosen.values())).kind == "understanding"
    boolean = all(c.gtype == "boolean" for c in chosen.values()) if chosen else False
    def strata_trim(cur):
        by_stratum = defaultdict(lambda: defaultdict(list))
        for sid, c in cur.items():
            by_stratum[c.stratum][c.akey].append(sid)
        out = {}
        for st, groups in by_stratum.items():
            n_s = sum(len(v) for v in groups.values())
            floor = max(2, TINY_CLASS_SHARE * n_s)
            k_eff = sum(1 for v in groups.values() if len(v) >= floor)
            if DROP_TINY_CLASSES and st and understanding:
                groups = {a: v for a, v in groups.items() if len(v) >= floor}      # a class that is (almost) absent from the stratum must not lower the cap of the others
            if st and not understanding:                       # grounding: the empty share is controlled inside every stratum below
                for a, v in groups.items():
                    for sid in v:
                        out[sid] = cur[sid]
                continue
            if st and understanding and k_eff < 2:
                continue                                        # a parameter value whose answer is fixed: predictable from the text -> drop
            cap = waterfill_cap({a: len(v) for a, v in groups.items()}, max_share(k_eff, boolean, stratum=bool(st)))
            for a, v in groups.items():
                v = sorted(v)
                rng.shuffle(v)
                for sid in v[:cap]:
                    out[sid] = cur[sid]
        return out

    keep = strata_trim(chosen)
    if not understanding:
        # empty answers must not be predictable from the stated context: inside every stratum (meter, k / note value) at most EMPTY_CAP of the items are empty
        by_s = defaultdict(list)
        for sid, c in keep.items():
            if c.stratum:
                by_s[c.stratum].append(sid)
        for st, ids in by_s.items():
            ne = sum(1 for s_ in ids if not keep[s_].empty)
            em = sorted(s_ for s_ in ids if keep[s_].empty)
            rng.shuffle(em)
            for s_ in em[int(EMPTY_CAP / (1 - EMPTY_CAP) * ne):]:
                del keep[s_]
    def global_trim(cur):
        groups = defaultdict(list)
        for sid, c in cur.items():
            groups[c.akey].append(sid)
        n = len(cur)
        k_eff = sum(1 for v in groups.values() if len(v) >= max(2, 0.01 * n))
        cap = waterfill_cap({a: len(v) for a, v in groups.items()}, max_share(k_eff, boolean))      # the class-share rule only; the size cap comes afterwards
        out = {}
        for a, v in groups.items():
            v = sorted(v)
            rng.shuffle(v)
            for sid in v[:cap]:
                out[sid] = cur[sid]
        return out

    keep = global_trim(keep)
    for _ in range(8 if understanding else 0):        # the two trims remove items unevenly: alternate until both rules hold (rc3: R-U3 text-only leak via the onset time)
        nxt = global_trim(strata_trim(keep))
        if len(nxt) == len(keep):
            break
        keep = nxt
    if total_cap is not None:
        keep = proportional_subsample(keep, total_cap, rng)
    # --- empty-answer cap
    emp = sorted(s for s, c in keep.items() if c.empty)
    rng.shuffle(emp)
    while emp and len(emp) > EMPTY_CAP * len(keep):
        del keep[emp.pop()]
    return keep


# ------------------------------------------------------------------------------------------------ rendering
def ctx_sentences(c: Cand, e: Excerpt, rng):
    flags = list(c.ctx) + [f for f in c.opt if rng.random() < 0.5]
    out = []
    if "meter" in flags:
        out.append(P.meter_sentence(rng, e.frame.meter, "unit" in flags))
    if "tempo" in flags:
        out.append(P.tempo_sentence(rng, e.frame.tempo_bpm))
    if "metric" in flags:
        out.append(P.metric_sentence(rng, e.frame.meter))
    if "frame" in flags:
        out.append(P.frame_sentence(rng))
    out += [f[5:] for f in flags if f.startswith("text:")]
    return out, flags


def frames(spans):
    return [[int(round(a * FPS)), max(int(round(b * FPS)), int(round(a * FPS)) + 1)] for a, b in spans]


def render(c: Cand, e: Excerpt, split: str, sid: str, rng):
    ctxs, flags = ctx_sentences(c, e, rng)
    if c.kind == "grounding":
        w = rng.choice(c.whats)
        what, plural = w if isinstance(w, tuple) else (w, c.plural)
        spans = c.spans if c.spans is not None else e.spans(c.idx)
        q, variants = P.render_grounding_question(rng, what, plural, ctxs)
        texts, style = P.grounding_answer_texts(spans, what, plural), 1
    else:
        if c.asks:
            q, variants = P.render_generic_question(rng, c.asks, ctxs, **c.fmt)
        else:
            q, variants = P.render_count_question(rng, c.fmt["np"], ctxs)
        spans, texts, style = e.spans(c.idx), c.ans, rng.choice([0, 1, 2])
    rec = {
        "question": q, "question_variants": variants,
        "answer_text": texts[str(style)], "answer_texts": texts, "default_language_style": style,
        "answer_spans": frames([(round(a, 6), round(b, 6)) for a, b in spans]), "answer_spans_sec": [[round(a, 6), round(b, 6)] for a, b in spans],
        "query_key": f"Q/{c.cat}/{c.concept.upper()}/{c.fam}:{c.sub.split('=')[0]}", "qa_category": c.cat, "concept": c.concept,
        "confidence": "high", "audio_path": f"{split}/wav/{sid}.wav", "sample_id": sid, "split": split,
        "source_fact_id": c.fam, "source_fields": [f"oracle2b:{c.fam}"],
        "notes": [{"pitch": n.pitch, "velocity": n.velocity, "start": round(n.start, 6), "end": round(n.end, 6)} for n in e.notes],
        "v2": {"oracle": ORACLE_VERSION, "family": c.fam, "sub": c.sub, "params": c.prov, "context": flags, "answer_class": c.akey,
               "frame": {"meter": e.frame.meter, "tempo_bpm": e.frame.tempo_bpm, "music_start": round(e.frame.music_start, 6), "bar_dur": round(e.frame.bar_dur, 6)}},
    }
    if c.kind == "understanding":
        rec.update({"gold_answer": c.gold, "gold_answer_type": c.gtype, "evidence_note_indices": sorted(c.idx), "evidence_type": "note_span_list"})
        if c.prov.get("tolerance_bpm"):
            rec["gold_tolerance"] = c.prov["tolerance_bpm"]
    return rec


# ------------------------------------------------------------------------------------------------ main
def load_split(split, extra_root=None):
    """released excerpts + (optionally) the new true-transposed excerpts (their own midi/meta folders; audio must exist before release)"""
    roots = [(MIDI_ROOT, META_ROOT)] + ([(extra_root, extra_root)] if extra_root else [])
    out = {}
    for mroot, troot in roots:
        if not os.path.isdir(f"{mroot}/{split}/midi"):
            continue
        for sid in sorted(f[:-4] for f in os.listdir(f"{mroot}/{split}/midi") if f.endswith(".mid")):
            meta = json.load(open(f"{troot}/{split}/meta/{sid}.json"))
            e = Excerpt(load_notes(f"{mroot}/{split}/midi/{sid}.mid"), frame_from_basic_info(meta["basic_info"]))
            e.uid = sid
            if e.usable:
                out[sid] = e
    return out


def cap_yesno(sel_by_fam, rng):
    """yes/no items <= YESNO_CAP of a concept's understanding items; dropping is class-balanced so yes and no stay (nearly) 50/50"""
    by_concept = defaultdict(list)
    for fam, d in sel_by_fam.items():
        for sid, c in d.items():
            if c.kind == "understanding":
                by_concept[c.concept].append((fam, sid, c))
    for concept, items in by_concept.items():
        yn = [(f, s, c) for f, s, c in items if c.gtype == "boolean"]
        limit = int((len(items) - len(yn)) * YESNO_CAP / (1 - YESNO_CAP))      # yes/no <= 10 % of the concept's final understanding items
        if len(yn) <= limit:
            continue
        T = [(f, s) for f, s, c in yn if c.gold]
        F = [(f, s) for f, s, c in yn if not c.gold]
        rng.shuffle(T); rng.shuffle(F)
        nt = min(len(T), limit // 2); nf = min(len(F), limit - nt); nt = min(len(T), limit - nf)
        for f, s in T[nt:] + F[nf:]:
            del sel_by_fam[f][s]


# ------------------------------------------------------------------------------------------------ robustness filter
JIT, DRAWS = 0.005, 10


def _key(c):
    return (c.fam, c.sub, json.dumps({k: v for k, v in c.prov.items() if k != "onset"}, sort_keys=True, default=str))


def _table(cands_by_fam):
    return {_key(c): (tuple(c.idx), json.dumps(c.gold, default=str)) for lst in cands_by_fam.values() for c in lst}


def _jittered(e, rng):
    S = [n.start + rng.uniform(-JIT, JIT) for n in e.notes]
    E = [n.end + rng.uniform(-JIT, JIT) for n in e.notes]
    for i in range(len(S) - 1):
        E[i] = min(E[i], S[i + 1])                         # legato notes share their boundary: jitter must not create overlaps
    E = [max(x, s_ + 0.05) for s_, x in zip(S, E)]
    e2 = Excerpt([Note(s_, x, n.pitch, n.velocity) for s_, x, n in zip(S, E, e.notes)], e.frame)
    e2.uid = e.uid
    return e2


def stable_candidates(e, stats):
    """candidates whose answer is unchanged under +-5 ms jitter of every onset/offset (4 draws); the others sit on a tolerance edge and are dropped"""
    base = candidates(e)
    rng = random.Random(f"jit|{e.uid}")
    exs = [_jittered(e, rng) for _ in range(DRAWS)]
    tabs = [_table(candidates(x)) for x in exs]
    out = {}
    for fam, lst in base.items():
        keep = []
        for c in lst:
            if c.fam == "R-U3":                             # time-referenced notes are re-drawn per excerpt: compare through the note index
                ok = all(x.usable and x.tempo_answer() is not None and x.notes_with_value(0.5) is not None and x.value_name(c.prov["note_index"]) == c.gold for x in exs)
            else:
                v = (tuple(c.idx), json.dumps(c.gold, default=str)); k = _key(c)
                ok = all(t.get(k) == v for t in tabs)
            stats[fam][0] += 1
            stats[fam][1] += (not ok)
            if ok:
                keep.append(c)
        if keep:
            out[fam] = keep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--extra-root", default=None, help="folder with the new true-transposed excerpts (<split>/midi, <split>/meta)")
    ap.add_argument("--train-cap-g", type=int, default=4200)
    ap.add_argument("--train-cap-u", type=int, default=3800)
    ap.add_argument("--train-concept-cap", type=int, default=5000, help="rc10: at most this many train items per music concept and task (soft balance)")
    ap.add_argument("--no-concept-quota", action="store_true", help="rc8 behaviour: no concept balancing")
    ap.add_argument("--train-chunk-clips", type=int, default=0,
                    help="TRIED AND REJECTED (rc9 experiment): select the train items in random chunks of about this many clips (e.g. 1100 = the size of val/test), each chunk with exactly the rules and the "
                         "scale val/test are selected with, and take the union; the balancing thresholds depend on the split size, so this makes the train answer "
                         "distribution follow the val/test one. It does, but it also copies the residual text<->answer correlations of the small-scale val/test balancing into train, and "
                         "B-U4 and R-U3 then fail the text-only gate; keep 0 (= select over the whole train split, rc8).")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/grounding_qa", exist_ok=True)
    os.makedirs(f"{a.out}/understanding_qa", exist_ok=True)
    report, all_recs = {}, {}
    for split in a.splits:
        exc = load_split(split, a.extra_root)
        fam_cands = defaultdict(dict)                       # fam -> sid -> [Cand]
        unstable = defaultdict(lambda: [0, 0])
        for sid, e in exc.items():
            for fam, lst in stable_candidates(e, unstable).items():
                fam_cands[fam][sid] = lst
        sel = {}
        chunk_of = None
        if split == "train" and a.train_chunk_clips:
            order = sorted(exc)
            random.Random(f"{a.seed}|train|chunks").shuffle(order)
            nch = max(1, round(len(order) / a.train_chunk_clips))
            chunk_of = {sid: i % nch for i, sid in enumerate(order)}
        for fam in sorted(fam_cands):
            rng = random.Random(f"{a.seed}|{split}|{fam}")
            kind = next(iter(fam_cands[fam].values()))[0].kind
            cap = (a.train_cap_g if kind == "grounding" else a.train_cap_u) if split == "train" else None
            m = SLOTS.get(fam, 1)
            if chunk_of is None and m == 1:
                sel[fam] = {(sid, 0): c for sid, c in select_family(fam_cands[fam], rng, total_cap=cap).items()}
            elif chunk_of is None:
                picked, remaining = {}, {sid: list(lst) for sid, lst in fam_cands[fam].items()}
                for k in range(m):
                    part = {sid: lst for sid, lst in remaining.items() if lst}
                    if not part:
                        break
                    for sid, c in select_family(part, random.Random(f"{a.seed}|{split}|{fam}|slot{k}"), total_cap=None).items():
                        picked[(sid, k)] = c
                        remaining[sid] = [x for x in remaining[sid] if x is not c]
                sel[fam] = proportional_subsample(picked, cap, rng) if cap is not None else picked
            else:
                union = {}
                for ci in sorted(set(chunk_of.values())):
                    part = {sid: c for sid, c in fam_cands[fam].items() if chunk_of[sid] == ci}
                    if part:
                        union.update(select_family(part, random.Random(f"{a.seed}|train|{fam}|chunk{ci}"), total_cap=None))
                sel[fam] = {(sid, 0): c for sid, c in (proportional_subsample(union, cap, rng) if cap is not None else union).items()}
        quota_report = None
        if not a.no_concept_quota:
            quota_report = concept_quota(sel, "soft" if split == "train" else "equal", a.train_concept_cap, f"{a.seed}|{split}")
        cap_yesno(sel, random.Random(f"{a.seed}|{split}|yesno"))
        recs = {"grounding": [], "understanding": []}
        for fam in sorted(sel):
            for (sid, slot), c in sorted(sel[fam].items()):
                rng = random.Random(f"{a.seed}|{split}|{sid}|{fam}|render" + ("" if slot == 0 else f"|{slot}"))
                recs[c.kind].append(render(c, exc[sid], split, sid, rng))
        # melody-twin flags (reporting aid): does the clip have a train clip with the same pitch sequence / same pitch sequence and rhythm?
        def sigs(ex):
            ns = ex.notes; t0 = ns[0].start
            return tuple(n.pitch for n in ns), tuple((n.pitch, round((n.start - t0) / 0.05), round((n.end - n.start) / 0.05)) for n in ns)
        if split == "train":
            train_s1 = {sigs(x)[0] for x in exc.values()}; train_s2 = {sigs(x)[1] for x in exc.values()}
        else:
            for kind in recs:
                for r in recs[kind]:
                    s1, s2 = sigs(exc[r["sample_id"]])
                    r["v2"]["twin"] = {"pitch_sequence_in_train": s1 in train_s1, "pitch_rhythm_in_train": s2 in train_s2}
        for kind in recs:
            recs[kind].sort(key=lambda r: (r["sample_id"], r["query_key"]))
        all_recs[split] = recs
        report[split] = {"clips": len(exc), "families": {f: len(d) for f, d in sorted(sel.items())},
                         "unstable_candidates_dropped(+-5ms jitter)": {f: {"candidates": v[0], "dropped": v[1]} for f, v in sorted(unstable.items()) if v[1]},
                         "concept_quota": quota_report, "n_grounding": len(recs["grounding"]), "n_understanding": len(recs["understanding"])}
        print(split, "clips", len(exc), "grounding", len(recs["grounding"]), "understanding", len(recs["understanding"]), flush=True)
    # test-small: one item per clip, greedily balanced over families (same construction for grounding and understanding)
    if "test" in all_recs:
        small = {}
        for kind, lst in all_recs["test"].items():
            by_clip = defaultdict(list)
            for r in lst:
                by_clip[r["sample_id"]].append(r)
            rng = random.Random(f"{a.seed}|test-small|{kind}")
            cnt, ccnt, yn, ccls, pick = Counter(), Counter(), Counter(), Counter(), []
            sids = sorted(by_clip)
            rng.shuffle(sids)
            for sid in sids:
                yn_full = lambda x: x.get("gold_answer_type") == "boolean" and yn[x["concept"]] >= 0.08 * (ccnt[x["concept"]] + 1)
                r = min(by_clip[sid], key=lambda x: (yn_full(x), ccnt[x["concept"]], cnt[x["v2"]["family"]], ccls[(x["v2"]["family"], x["v2"]["answer_class"])], rng.random()))
                ccls[(r["v2"]["family"], r["v2"]["answer_class"])] += 1
                cnt[r["v2"]["family"]] += 1
                ccnt[r["concept"]] += 1
                yn[r["concept"]] += r.get("gold_answer_type") == "boolean"
                pick.append(r)
            small[kind] = sorted(pick, key=lambda r: r["sample_id"])
        all_recs["test-small"] = small
    # query ids over all splits
    keys = sorted({r["query_key"] for s in all_recs.values() for k in s.values() for r in k})
    vocab = {k: i for i, k in enumerate(keys)}
    for split, recs in all_recs.items():
        for kind, lst in recs.items():
            for r in lst:
                r["query_id"] = vocab[r["query_key"]]
                r["question_token"] = f"<|Q{r['query_id']}|>"
            json.dump(lst, open(f"{a.out}/{kind}_qa/{split}.json", "w"), ensure_ascii=False)
    json.dump(vocab, open(f"{a.out}/query_vocab_v2.json", "w"), indent=1)
    json.dump(report, open(f"{a.out}/build_report.json", "w"), indent=1)
    print("query keys:", len(vocab))


if __name__ == "__main__":
    main()
