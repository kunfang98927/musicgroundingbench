"""Step 3N-9 (optional): build the out-of-distribution generalization test set.

Queries the same test clips with a hand-authored bank of paraphrased/conceptual
query templates instead of the rule-based ones from step 3N-5 - this measures
whether a trained model generalizes to phrasings and concepts (like "pitch
class" or "melodic n-grams") it never saw in training. The query bank itself
(`reference/query_generalize_ood.json`) is authored content, not generated -
check it in as-is, there's nothing to reproduce there.

Uses the GLOBAL `random` module (seeded once), matching the original.

Reads metadata.csv by column name, not position: the original script indexed
columns positionally (expecting a `duration` column before `midi_path` that
step 3N-2 doesn't actually write - the two had drifted apart), which made it
silently find zero test rows. Reading by name works regardless of column
order/count and doesn't affect the other steps, none of which depend on this
one's output.

Usage:
    python scripts/09_build_generalize_testset.py \\
        --data_dir piano-melody-3notes --queries reference/query_generalize_ood.json \\
        --out query_dataset_for_llm/llm_generalize.json
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NOTE_NAME_TO_PC = {
    "C": 0, "C_sharp": 1, "D_flat": 1, "D": 2, "D_sharp": 3, "E_flat": 3,
    "E": 4, "F": 5, "F_sharp": 6, "G_flat": 6, "G": 7, "G_sharp": 8,
    "A_flat": 8, "A": 9, "A_sharp": 10, "B_flat": 10, "B": 11,
}


def sec_to_tidx(t: float, fps: float) -> int:
    return int(round(float(t) * fps))


def spans_to_answer_token(answer_spans: List[List[int]]) -> str:
    if not answer_spans:
        return "<no_time_span>"
    return "".join(f"<box><|T{int(s)}|><|T{int(e)}|></box>" for s, e in answer_spans) or "<no_time_span>"


def safe_answer_text_from_seconds(spans_sec: List[List[float]]) -> str:
    if not spans_sec:
        return "No matching notes."
    s, e = spans_sec[0]
    return f"From {float(s):.3f} second to {float(e):.3f} second."


def get_id_from_key(qkey: str, qk2id: Dict[str, Any]) -> str:
    return str(qk2id[qkey]) if qkey in qk2id else None


def load_query_bank(query_json_path: str) -> Dict[str, str]:
    data = json.load(open(query_json_path, encoding="utf-8"))
    bank = data["query_key_to_text"] if "query_key_to_text" in data else data
    if not bank:
        raise ValueError("Empty query bank.")
    return bank


def read_metadata_csv(metadata_csv_path: str) -> List[Dict[str, str]]:
    with open(metadata_csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_test_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("split", "").strip().lower() == "test"]


def load_events_from_meta(meta_json_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (events_norm, notes_all): events_norm is {pitch, velocity,
    start, end} (seconds) used for span computation, dropping any note with a
    non-positive duration; notes_all is every note with a valid pitch/onset
    /offset, in the original {pitch, velocity, onset, offset} shape, for the
    output "notes" field - a superset of events_norm."""
    data = json.load(open(meta_json_path, encoding="utf-8"))
    events_norm, notes_all = [], []
    for ev in data.get("events", []):
        pitch, vel = ev.get("pitch"), ev.get("velocity")
        onset, offset = ev.get("onset"), ev.get("offset")

        if pitch is not None and onset is not None and offset is not None:
            notes_all.append({"pitch": int(pitch), "velocity": int(vel) if vel is not None else None,
                               "onset": float(onset), "offset": float(offset)})

        if pitch is None or onset is None or offset is None or float(offset) <= float(onset):
            continue
        events_norm.append({"pitch": int(pitch), "velocity": int(vel) if vel is not None else None,
                             "start": float(onset), "end": float(offset)})

    notes_all.sort(key=lambda x: x["onset"])
    events_norm.sort(key=lambda x: x["start"])
    return events_norm, notes_all


def parse_delta_from_key(qkey: str) -> Optional[int]:
    return int(qkey.split("delta=")[-1]) if "delta=" in qkey else None


def parse_pitch_class_name_from_key(qkey: str) -> Optional[str]:
    return qkey.split("name=")[-1] if "ABS/PITCH_CLASS/name=" in qkey else None


def parse_pcs_list_from_key(qkey: str) -> Optional[List[int]]:
    if "NOTE_SEQ/pcs=" not in qkey:
        return None
    return [int(x) for x in qkey.split("pcs=")[-1].split("/")[0].split(",")]


def parse_ord_idx_from_key(qkey: str) -> Optional[int]:
    return int(qkey.split("idx=")[-1]) if "/ORD/NOTE/idx=" in qkey else None


def dedup_spans(spans: List[List[float]]) -> List[List[float]]:
    out, seen = [], set()
    for s, e in spans:
        if s is None or e is None:
            continue
        key = (float(s), float(e))
        if key in seen:
            continue
        seen.add(key)
        out.append([float(s), float(e)])
    return out


def argmax_or_argmin_note(events: List[Dict[str, Any]], field: str, mode: str) -> List[List[float]]:
    if not events:
        return []
    best_i, best_val = None, None
    for i, ev in enumerate(events):
        val = (ev["end"] - ev["start"]) if field == "duration" else ev.get(field)
        if val is None:
            continue
        if best_i is None or (mode == "argmax" and val > best_val) or (mode == "argmin" and val < best_val):
            best_i, best_val = i, val
    return [[events[best_i]["start"], events[best_i]["end"]]] if best_i is not None else []


def ord_note_idx(events: List[Dict[str, Any]], idx: int) -> List[List[float]]:
    if not events:
        return []
    n = len(events)
    pos = idx if idx >= 0 else (n + idx)
    return [[events[pos]["start"], events[pos]["end"]]] if 0 <= pos < n else []


def pitch_class_spans(events: List[Dict[str, Any]], pc: int) -> List[List[float]]:
    return dedup_spans([[ev["start"], ev["end"]] for ev in events if (ev["pitch"] % 12) == (pc % 12)])


def note_seq_spans_per_note(events: List[Dict[str, Any]], pcs: List[int]) -> List[List[float]]:
    """Per-note spans (not merged): for each matching n-gram, every involved
    note's own span is included."""
    if not events or not pcs:
        return []
    seq = [ev["pitch"] % 12 for ev in events]
    target = [p % 12 for p in pcs]
    L = len(target)
    out = []
    for i in range(len(seq) - L + 1):
        if seq[i:i + L] == target:
            out.extend([events[i + j]["start"], events[i + j]["end"]] for j in range(L))
    return dedup_spans(out)


def pitch_diff_pair_spans_per_note(events: List[Dict[str, Any]], delta: int) -> List[List[float]]:
    """Per-note spans for every pair whose pitch difference matches delta."""
    if not events or len(events) < 2:
        return []
    out = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            if abs(events[j]["pitch"] - events[i]["pitch"]) == delta:
                out += [[events[i]["start"], events[i]["end"]], [events[j]["start"], events[j]["end"]]]
    return dedup_spans(out)


def get_gen_type(qkey: str) -> str:
    parts = qkey.split("/")
    return parts[2] if len(parts) >= 3 and parts[1] == "GEN" and parts[2] in ("PARA", "CONCEPT") else "OTHER"


def get_family(qkey: str) -> str:
    parts = qkey.split("/")
    return parts[3] if len(parts) >= 4 and parts[3] in ("REL", "ORD", "ABS", "PAT") else "OTHER"


def eligible_queries_for_events(events: List[Dict[str, Any]], query_keys: List[str]) -> List[str]:
    """A query is eligible for this clip if a REL/ORD paraphrase (always
    answerable) or its underlying pitch-class/n-gram/interval actually occurs."""
    if not events:
        return []
    n = len(events)
    eligible = [k for k in query_keys if k.startswith("Q/GEN/PARA/REL/") or k.startswith("Q/GEN/PARA/ORD/NOTE/idx=")]

    pcs_present = {ev["pitch"] % 12 for ev in events}
    for k in query_keys:
        if "Q/GEN/CONCEPT/ABS/PITCH_CLASS/name=" in k:
            pc = NOTE_NAME_TO_PC.get(parse_pitch_class_name_from_key(k))
            if pc is not None and pc in pcs_present:
                eligible.append(k)

    if n >= 2:
        seq = [ev["pitch"] % 12 for ev in events]
        bigrams = {tuple(seq[i:i + 2]) for i in range(len(seq) - 1)}
        trigrams = {tuple(seq[i:i + 3]) for i in range(len(seq) - 2)} if n >= 3 else set()
        for k in query_keys:
            if "Q/GEN/CONCEPT/PAT/MELODY/NOTE_SEQ/pcs=" in k:
                pcs = parse_pcs_list_from_key(k)
                if not pcs:
                    continue
                tpcs = tuple(p % 12 for p in pcs)
                if (len(tpcs) == 2 and tpcs in bigrams) or (len(tpcs) == 3 and tpcs in trigrams):
                    eligible.append(k)

        diffs = {abs(events[j]["pitch"] - events[i]["pitch"]) for i in range(n) for j in range(i + 1, n)}
        for k in query_keys:
            if "Q/GEN/CONCEPT/PAT/PITCH_DIFF/delta=" in k:
                d = parse_delta_from_key(k)
                if d is not None and d in diffs:
                    eligible.append(k)

    return list(dict.fromkeys(eligible))


def normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in d.values())
    if s <= 0:
        return {k: 1.0 / len(d) for k in d} if d else {}
    return {k: max(0.0, v) / s for k, v in d.items()}


def stratified_pick_query(eligible: List[str], p_para: float, family_probs: Dict[str, float]) -> str:
    """Pick a query_type (PARA vs CONCEPT) then a family (REL/ORD/ABS/PAT) by
    weighted coin flips, then a random eligible query matching both if any
    exist - falling back to matching just the type, then just the family,
    then any eligible query."""
    family_probs = normalize_weights(family_probs)
    want_type = "PARA" if random.random() < p_para else "CONCEPT"

    by_type, by_family, by_type_family = defaultdict(list), defaultdict(list), defaultdict(list)
    for k in eligible:
        t, f = get_gen_type(k), get_family(k)
        by_type[t].append(k)
        by_family[f].append(k)
        by_type_family[(t, f)].append(k)

    fams = list(family_probs.keys())
    want_fam = random.choices(fams, weights=[family_probs[f] for f in fams], k=1)[0]

    for cand in (by_type_family.get((want_type, want_fam), []), by_type.get(want_type, []), by_family.get(want_fam, [])):
        if cand:
            return random.choice(cand)
    return random.choice(eligible)


def generate_for_metadata(
    data_dir: str, queries_json: str, out_json: str, seed: int, fps: float,
    allow_empty: bool, p_para: float, p_rel: float, p_ord: float, p_abs: float, p_pat: float,
) -> None:
    random.seed(seed)

    data_root = Path(data_dir)
    metadata_path = data_root / "metadata.csv"
    query_bank = load_query_bank(queries_json)
    qk2id = json.load(open(queries_json, encoding="utf-8")).get("query_key_to_id", {})
    query_keys = list(query_bank.keys())

    test_rows = filter_test_rows(read_metadata_csv(str(metadata_path)))
    print(f"Found {len(test_rows)} test rows in metadata.")

    family_probs = {"REL": p_rel, "ORD": p_ord, "ABS": p_abs, "PAT": p_pat}
    out_list: List[Dict[str, Any]] = []

    for r in test_rows:
        audio_path, meta_rel = r["audio_path"].strip(), r["json_path"].strip()
        meta_json_path = data_root / meta_rel if not Path(meta_rel).is_absolute() else Path(meta_rel)

        try:
            events, notes_all = load_events_from_meta(str(meta_json_path))
        except Exception:
            events, notes_all = [], []

        if allow_empty:
            chosen_key = random.choice(query_keys)
        else:
            elig = eligible_queries_for_events(events, query_keys)
            if elig:
                chosen_key = stratified_pick_query(elig, p_para=p_para, family_probs=family_probs)
            else:
                fallback = [k for k in query_keys if k.startswith("Q/GEN/PARA/REL/")]
                chosen_key = random.choice(fallback) if fallback else random.choice(query_keys)

        spans_sec: List[List[float]] = []
        if chosen_key.startswith("Q/GEN/PARA/REL/VEL/"):
            spans_sec = argmax_or_argmin_note(events, "velocity", "argmax" if "argmax" in chosen_key else "argmin")
        elif chosen_key.startswith("Q/GEN/PARA/REL/PITCH/"):
            spans_sec = argmax_or_argmin_note(events, "pitch", "argmax" if "argmax" in chosen_key else "argmin")
        elif chosen_key.startswith("Q/GEN/PARA/ORD/NOTE/idx="):
            idx = parse_ord_idx_from_key(chosen_key)
            spans_sec = ord_note_idx(events, idx if idx is not None else 0)
        elif "Q/GEN/CONCEPT/ABS/PITCH_CLASS/name=" in chosen_key:
            pc = NOTE_NAME_TO_PC.get(parse_pitch_class_name_from_key(chosen_key))
            spans_sec = pitch_class_spans(events, pc) if pc is not None else []
        elif "Q/GEN/CONCEPT/PAT/MELODY/NOTE_SEQ/pcs=" in chosen_key:
            pcs = parse_pcs_list_from_key(chosen_key)
            spans_sec = note_seq_spans_per_note(events, pcs) if pcs else []
        elif "Q/GEN/CONCEPT/PAT/PITCH_DIFF/delta=" in chosen_key:
            delta = parse_delta_from_key(chosen_key)
            spans_sec = pitch_diff_pair_spans_per_note(events, delta) if delta is not None else []

        spans_sec = dedup_spans(spans_sec)
        spans_tidx = [[sec_to_tidx(s, fps), sec_to_tidx(e, fps)] for s, e in spans_sec]

        out_list.append({
            "question": query_bank[chosen_key],
            "answer": safe_answer_text_from_seconds(spans_sec),
            "question_token": get_id_from_key(chosen_key, qk2id),
            "answer_token": spans_to_answer_token(spans_tidx),
            "query_key": chosen_key,
            "answer_spans": spans_tidx,
            "audio_path": audio_path,
            "notes": notes_all,
        })

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out_list)} entries to {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--queries", required=True, help="reference/query_generalize_ood.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=75.0)
    ap.add_argument("--allow_empty", action="store_true")
    ap.add_argument("--p_para", type=float, default=0.40, help="P(paraphrase) vs P(concept)")
    ap.add_argument("--p_rel", type=float, default=0.15)
    ap.add_argument("--p_ord", type=float, default=0.15)
    ap.add_argument("--p_abs", type=float, default=0.10)
    ap.add_argument("--p_pat", type=float, default=0.60)
    args = ap.parse_args()

    generate_for_metadata(
        data_dir=args.data_dir, queries_json=args.queries, out_json=args.out, seed=args.seed, fps=args.fps,
        allow_empty=args.allow_empty, p_para=args.p_para, p_rel=args.p_rel, p_ord=args.p_ord,
        p_abs=args.p_abs, p_pat=args.p_pat,
    )


if __name__ == "__main__":
    main()
