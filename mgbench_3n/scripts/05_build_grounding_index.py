"""Step 3N-5: turn each clip's notes into grounding query-answer pairs
(query_key + answer time-spans), balanced across the ABS/REL/ORD/PAT query
families with a target ~15% empty-answer rate for negatives.

Kept close to a 1:1 port of the original: `sample_k_queries_with_target_empty`
consumes the RNG in an exact sequence (rejection-sampling loops for empty vs.
non-empty candidates), so its control flow is preserved verbatim rather than
restructured. Unlike 2B, `--mert_h5` is only ever stored as a path string here
- it's never opened - so this step needs no rendered audio either.

Usage:
    python scripts/05_build_grounding_index.py \\
        --data_dir piano-melody-3notes --index_csv piano-melody-3notes/metadata.csv \\
        --out_dir query_dataset_for_baselines --mert_h5 <path/for/reference/only> --fps 75
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class Note:
    pitch: int
    vel: int
    onset: float
    offset: float

    @property
    def dur(self) -> float:
        return max(0.0, self.offset - self.onset)


Span = Tuple[int, int]  # [s, e) in frames


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True, help="dataset root; paths in index_csv are relative to this")
    ap.add_argument("--index_csv", required=True, help="metadata.csv from step 3N-2")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mert_h5", required=True, help="path to MERT h5 (stored into output, never read)")
    ap.add_argument("--fps", type=float, required=True, help="frame rate of MERT embeddings (frames/sec)")
    ap.add_argument("--k_per_audio", type=int, default=20, help="queries sampled per clip")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--target_empty_ratio", type=float, default=0.15)
    ap.add_argument("--vel_bins", type=int, default=8)
    ap.add_argument("--pitch_min", type=int, default=21)
    ap.add_argument("--pitch_max", type=int, default=108)
    ap.add_argument("--max_neg_attempts_factor", type=int, default=50)
    ap.add_argument("--w_abs", type=float, default=0.25)
    ap.add_argument("--w_rel", type=float, default=0.30)
    ap.add_argument("--w_ord", type=float, default=0.20)
    ap.add_argument("--w_pat", type=float, default=0.25)
    return ap.parse_args()


def load_json(full_path: Path) -> Dict[str, Any]:
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def seconds_to_frames(t: float, fps: float) -> int:
    return int(round(t * fps))


def note_to_span_frames(note: Note, fps: float) -> Span:
    s = seconds_to_frames(note.onset, fps)
    e = max(seconds_to_frames(note.offset, fps), s + 1)
    return (s, e)


def vel_to_bin(vel: int, n_bins: int) -> int:
    vel = max(0, min(127, int(vel)))
    return min(n_bins - 1, max(0, int(math.floor(vel * n_bins / 128.0))))


def sign3(x: float) -> str:
    return "up" if x > 0 else ("down" if x < 0 else "eq")


def stable_sort_notes(notes: List[Note]) -> List[Note]:
    return sorted(notes, key=lambda n: (n.onset, n.pitch, n.offset))


def uniq_spans(spans: List[Span]) -> List[Span]:
    return list(dict.fromkeys(spans))


def extract_notes_from_meta(meta: Dict[str, Any]) -> List[Note]:
    candidates = meta.get("events", [])
    notes: List[Note] = []
    for it in candidates:
        try:
            n = Note(pitch=int(it["pitch"]), vel=int(it.get("velocity", 64)),
                      onset=float(it["onset"]), offset=float(it["offset"]))
            if n.offset > n.onset:
                notes.append(n)
        except Exception:
            continue
    return stable_sort_notes(notes)


# =============================================================================
# Query answering: query_key -> answer spans, one branch per query family/type
# =============================================================================

def answer_query(query_key: str, notes: List[Note], fps: float, vel_bins: int) -> List[Span]:
    ns = stable_sort_notes(notes)
    N = len(ns)
    if N == 0:
        return []

    if query_key.startswith("Q/ABS/PITCH/vals="):
        vals = [int(x) for x in query_key.split("vals=", 1)[1].split(",") if x.strip()]
        if not vals:
            return []
        s = set(vals)
        return uniq_spans([note_to_span_frames(n, fps) for n in ns if n.pitch in s])

    if query_key.startswith("Q/ABS/VEL/bin="):
        b = int(query_key.split("bin=", 1)[1])
        return uniq_spans([note_to_span_frames(n, fps) for n in ns if vel_to_bin(n.vel, vel_bins) == b])

    if query_key.startswith("Q/ABS/VEL/bins="):
        bins = {int(x) for x in query_key.split("bins=", 1)[1].split(",") if x.strip()}
        return uniq_spans([note_to_span_frames(n, fps) for n in ns if vel_to_bin(n.vel, vel_bins) in bins])

    if query_key == "Q/REL/PITCH/argmax":
        maxp = max(n.pitch for n in ns)
        return [note_to_span_frames(n, fps) for n in ns if n.pitch == maxp]
    if query_key == "Q/REL/PITCH/argmin":
        minp = min(n.pitch for n in ns)
        return [note_to_span_frames(n, fps) for n in ns if n.pitch == minp]
    if query_key == "Q/REL/VEL/argmax":
        maxv = max(n.vel for n in ns)
        return [note_to_span_frames(n, fps) for n in ns if n.vel == maxv]
    if query_key == "Q/REL/VEL/argmin":
        minv = min(n.vel for n in ns)
        return [note_to_span_frames(n, fps) for n in ns if n.vel == minv]
    if query_key == "Q/REL/DUR/argmax":
        maxd = max(n.dur for n in ns)
        return [note_to_span_frames(n, fps) for n in ns if abs(n.dur - maxd) < 1e-9]
    if query_key == "Q/REL/DUR/argmin":
        mind = min(n.dur for n in ns)
        return [note_to_span_frames(n, fps) for n in ns if abs(n.dur - mind) < 1e-9]
    if query_key == "Q/REL/VELBIN/argmax":
        m = max(vel_to_bin(n.vel, vel_bins) for n in ns)
        return [note_to_span_frames(n, fps) for n in ns if vel_to_bin(n.vel, vel_bins) == m]
    if query_key == "Q/REL/VELBIN/argmin":
        m = min(vel_to_bin(n.vel, vel_bins) for n in ns)
        return [note_to_span_frames(n, fps) for n in ns if vel_to_bin(n.vel, vel_bins) == m]

    if query_key.startswith("Q/ORD/NOTE/idx="):
        idx = int(query_key.split("idx=", 1)[1])
        if idx == -1:
            return [note_to_span_frames(ns[-1], fps)]
        if 1 <= idx <= N:
            return [note_to_span_frames(ns[idx - 1], fps)]
        return []
    if query_key == "Q/ORD/NOTE/prefix=k2":
        return [note_to_span_frames(n, fps) for n in ns[:2]] if N >= 2 else []
    if query_key == "Q/ORD/NOTE/suffix=k2":
        return [note_to_span_frames(n, fps) for n in ns[-2:]] if N >= 2 else []

    if query_key.startswith("Q/PAT/PITCH_DIFF/delta="):
        d = int(query_key.split("delta=", 1)[1])
        if N < 2:
            return []
        spans: List[Span] = []
        for i in range(N):
            for j in range(i + 1, N):
                if abs(ns[i].pitch - ns[j].pitch) == d:
                    spans += [note_to_span_frames(ns[i], fps), note_to_span_frames(ns[j], fps)]
        return uniq_spans(spans)

    if query_key == "Q/PAT/PAIR/pitch_closest":
        if N < 2:
            return []
        min_d = min(abs(ns[i].pitch - ns[j].pitch) for i in range(N) for j in range(i + 1, N))
        spans = []
        for i in range(N):
            for j in range(i + 1, N):
                if abs(ns[i].pitch - ns[j].pitch) == min_d:
                    spans += [note_to_span_frames(ns[i], fps), note_to_span_frames(ns[j], fps)]
        return uniq_spans(spans)

    if query_key == "Q/PAT/PAIR/time_closest":
        if N < 2:
            return []
        gaps = [abs(ns[i].onset - ns[j].onset) for i in range(N) for j in range(i + 1, N)]
        min_gap = min(gaps) if gaps else 0.0
        spans = []
        for i in range(N):
            for j in range(i + 1, N):
                if abs(abs(ns[i].onset - ns[j].onset) - min_gap) < 1e-12:
                    spans += [note_to_span_frames(ns[i], fps), note_to_span_frames(ns[j], fps)]
        return uniq_spans(spans)

    if query_key.startswith("Q/PAT/CONTOUR/pitch="):
        if N < 3:
            return []
        want = query_key.split("pitch=", 1)[1]
        got = f"{sign3(ns[1].pitch - ns[0].pitch)}-{sign3(ns[2].pitch - ns[1].pitch)}"
        return [note_to_span_frames(n, fps) for n in ns] if got == want else []

    if query_key.startswith("Q/PAT/CONTOUR/vel="):
        if N < 3:
            return []
        want = query_key.split("vel=", 1)[1]
        v0, v1, v2 = (vel_to_bin(ns[i].vel, vel_bins) for i in range(3))
        got = f"{sign3(v1 - v0)}-{sign3(v2 - v1)}"
        return [note_to_span_frames(n, fps) for n in ns] if got == want else []

    if query_key.startswith("Q/PAT/INT/order="):
        if N < 3:
            return []
        want = query_key.split("order=", 1)[1]
        i1, i2 = abs(ns[1].pitch - ns[0].pitch), abs(ns[2].pitch - ns[1].pitch)
        got = "desc" if i1 > i2 else "asc" if i1 < i2 else "eq"
        return [note_to_span_frames(n, fps) for n in ns] if got == want else []

    return []


# =============================================================================
# Global query space (for sampling negatives) + positive-query enumeration
# =============================================================================

def all_contours() -> List[str]:
    s = ["up", "down", "eq"]
    return [f"{a}-{b}" for a in s for b in s]


def build_query_space_specs(pitch_min: int, pitch_max: int, vel_bins: int) -> Dict[str, Any]:
    return {
        "pitch_range": [pitch_min, pitch_max],
        "vel_bins": vel_bins,
        "rel_keys": [
            "Q/REL/PITCH/argmax", "Q/REL/PITCH/argmin", "Q/REL/VEL/argmax", "Q/REL/VEL/argmin",
            "Q/REL/DUR/argmax", "Q/REL/DUR/argmin", "Q/REL/VELBIN/argmax", "Q/REL/VELBIN/argmin",
        ],
        "ord_keys": [
            "Q/ORD/NOTE/idx=1", "Q/ORD/NOTE/idx=2", "Q/ORD/NOTE/idx=3", "Q/ORD/NOTE/idx=-1",
            "Q/ORD/NOTE/prefix=k2", "Q/ORD/NOTE/suffix=k2", "Q/ORD/NOTE/idx=4",
        ],
        "pat_fixed_keys": [
            "Q/PAT/PAIR/pitch_closest", "Q/PAT/PAIR/time_closest",
            "Q/PAT/INT/order=asc", "Q/PAT/INT/order=desc", "Q/PAT/INT/order=eq",
        ],
        "pitch_contours": all_contours(),
        "vel_contours": all_contours(),
        "pitch_diff_deltas": list(range(0, 13)),
        "vel_bin_values": list(range(vel_bins)),
    }


def sample_random_query_key(rng: random.Random, specs: Dict[str, Any]) -> str:
    """Sample a query key from the global query space; some sampled queries
    are empty on a given clip - that's what we want for negatives."""
    fam = rng.choice(["ABS", "REL", "ORD", "PAT"])

    if fam == "REL":
        return rng.choice(specs["rel_keys"])
    if fam == "ORD":
        return rng.choice(specs["ord_keys"])
    if fam == "PAT":
        r = rng.random()
        if r < 0.45:
            return f"Q/PAT/PITCH_DIFF/delta={rng.choice(specs['pitch_diff_deltas'])}"
        if r < 0.75:
            return f"Q/PAT/CONTOUR/pitch={rng.choice(specs['pitch_contours'])}"
        if r < 0.95:
            return f"Q/PAT/CONTOUR/vel={rng.choice(specs['vel_contours'])}"
        return rng.choice(specs["pat_fixed_keys"])

    # ABS
    r = rng.random()
    lo, hi = specs["pitch_range"]
    if r < 0.6:
        k = rng.choice([1, 2, 3])
        vals = sorted(rng.randint(lo, hi) for _ in range(k))
        return "Q/ABS/PITCH/vals=" + ",".join(map(str, vals))
    k = rng.choice([1, 2, 3])
    bins = sorted(rng.choice(specs["vel_bin_values"]) for _ in range(k))
    return f"Q/ABS/VEL/bin={bins[0]}" if k == 1 else "Q/ABS/VEL/bins=" + ",".join(map(str, bins))


def build_positive_queries(notes: List[Note], fps: float, vel_bins: int) -> List[str]:
    """Queries that are *likely non-empty* on this clip: present pitches/vel
    bins, always-non-empty REL keys, in-range ORD keys, and PAT keys that
    match the clip's actual pattern."""
    ns = stable_sort_notes(notes)
    N = len(ns)
    keys: List[str] = []

    for p in sorted(set(n.pitch for n in ns)):
        keys.append(f"Q/ABS/PITCH/vals={p}")
    if N >= 2:
        v = sorted([ns[0].pitch, ns[1].pitch])
        keys.append(f"Q/ABS/PITCH/vals={v[0]},{v[1]}")
    if N >= 3:
        v = sorted([ns[0].pitch, ns[1].pitch, ns[2].pitch])
        keys.append(f"Q/ABS/PITCH/vals={v[0]},{v[1]},{v[2]}")
    for b in sorted(set(vel_to_bin(n.vel, vel_bins) for n in ns)):
        keys.append(f"Q/ABS/VEL/bin={b}")

    keys += [
        "Q/REL/PITCH/argmax", "Q/REL/PITCH/argmin", "Q/REL/VEL/argmax", "Q/REL/VEL/argmin",
        "Q/REL/DUR/argmax", "Q/REL/DUR/argmin", "Q/REL/VELBIN/argmax", "Q/REL/VELBIN/argmin",
    ]

    for idx in range(1, min(N, 3) + 1):
        keys.append(f"Q/ORD/NOTE/idx={idx}")
    keys.append("Q/ORD/NOTE/idx=-1")
    if N >= 2:
        keys += ["Q/ORD/NOTE/prefix=k2", "Q/ORD/NOTE/suffix=k2"]

    if N >= 2:
        deltas = {abs(ns[i].pitch - ns[j].pitch) for i in range(N) for j in range(i + 1, N)}
        for d in sorted(deltas):
            keys.append(f"Q/PAT/PITCH_DIFF/delta={d}")
        keys += ["Q/PAT/PAIR/pitch_closest", "Q/PAT/PAIR/time_closest"]

    if N >= 3:
        keys.append(f"Q/PAT/CONTOUR/pitch={sign3(ns[1].pitch - ns[0].pitch)}-{sign3(ns[2].pitch - ns[1].pitch)}")
        v0, v1, v2 = (vel_to_bin(ns[i].vel, vel_bins) for i in range(3))
        keys.append(f"Q/PAT/CONTOUR/vel={sign3(v1 - v0)}-{sign3(v2 - v1)}")
        i1, i2 = abs(ns[1].pitch - ns[0].pitch), abs(ns[2].pitch - ns[1].pitch)
        order = "desc" if i1 > i2 else "asc" if i1 < i2 else "eq"
        keys.append(f"Q/PAT/INT/order={order}")

    return list(dict.fromkeys(keys))


# =============================================================================
# Per-clip sampling with a target empty-answer ratio
# =============================================================================

def sample_k_queries_with_target_empty(
    rng: random.Random, positive_keys: List[str], specs: Dict[str, Any], notes: List[Note],
    fps: float, vel_bins: int, k: int, target_empty_ratio: float, max_neg_attempts: int,
) -> List[str]:
    """1) fill k_pos slots from positive_keys (confirmed non-empty); 2) fill
    k_empty slots via rejection sampling from the global space; 3) top up
    positives from the global space if the pool ran out; 4) if the actual
    empty ratio overshoots the target, swap some empties back out for
    positives. Never pad with arbitrary queries beyond what's needed."""
    k_empty = int(round(k * max(0.0, min(1.0, target_empty_ratio))))
    k_pos = max(0, k - k_empty)

    pos_pool = list(dict.fromkeys(positive_keys))
    rng.shuffle(pos_pool)

    chosen: List[str] = []
    used = set()

    for qk in pos_pool:
        if len(chosen) >= k_pos:
            break
        if qk in used:
            continue
        if len(answer_query(qk, notes, fps, vel_bins)) == 0:
            continue
        chosen.append(qk)
        used.add(qk)

    empty_keys: List[str] = []
    attempts = 0
    while len(empty_keys) < k_empty and attempts < max_neg_attempts:
        attempts += 1
        qk = sample_random_query_key(rng, specs)
        if qk in used:
            continue
        if len(answer_query(qk, notes, fps, vel_bins)) == 0:
            empty_keys.append(qk)
            used.add(qk)

    pos_attempts = 0
    while len(chosen) < k_pos and pos_attempts < max(500, k * 50):
        pos_attempts += 1
        qk = sample_random_query_key(rng, specs)
        if qk in used:
            continue
        if len(answer_query(qk, notes, fps, vel_bins)) > 0:
            chosen.append(qk)
            used.add(qk)

    final = chosen + empty_keys
    final_attempts = 0
    while len(final) < k and final_attempts < 500:
        final_attempts += 1
        qk = sample_random_query_key(rng, specs)
        if qk in used:
            continue
        ans = answer_query(qk, notes, fps, vel_bins)
        if len(ans) > 0 or len(final) + 1 <= k:
            final.append(qk)
            used.add(qk)

    actual_empty = sum(1 for qk in final if len(answer_query(qk, notes, fps, vel_bins)) == 0)
    actual_ratio = actual_empty / max(1, len(final))
    if actual_ratio > target_empty_ratio:
        num_to_keep_empty = int(round(len(final) * target_empty_ratio))
        num_to_replace = actual_empty - num_to_keep_empty
        if num_to_replace > 0:
            pos_candidates = [qk for qk in pos_pool if qk not in used and len(answer_query(qk, notes, fps, vel_bins)) > 0]
            pos_attempts = 0
            while len(pos_candidates) < num_to_replace and pos_attempts < 500:
                pos_attempts += 1
                qk = sample_random_query_key(rng, specs)
                if qk in used:
                    continue
                if len(answer_query(qk, notes, fps, vel_bins)) > 0:
                    pos_candidates.append(qk)
                    used.add(qk)
            new_final, replaced = [], 0
            for qk in final:
                if replaced < num_to_replace and len(answer_query(qk, notes, fps, vel_bins)) == 0 and pos_candidates:
                    new_final.append(pos_candidates.pop())
                    replaced += 1
                    continue
                new_final.append(qk)
            final = new_final

    final = final[:k]
    rng.shuffle(final)
    return final


def build_vocab_and_tree(query_keys: List[str]) -> Tuple[Dict[str, int], Dict[str, Any]]:
    q2id: Dict[str, int] = {}
    for qk in query_keys:
        if qk not in q2id:
            q2id[qk] = len(q2id)

    tree: Dict[str, Any] = {}
    for qk, qid in q2id.items():
        parts = qk.split("/")
        if len(parts) < 4 or parts[0] != "Q":
            tree.setdefault("OTHER", {})[qk] = qid
            continue
        tree.setdefault(parts[1], {}).setdefault(parts[2], {})["/".join(parts[3:])] = qid

    return q2id, tree


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    with open(args.index_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"audio_path", "json_path", "midi_path", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"index_csv missing columns: {sorted(missing)}")
        rows = list(reader)

    specs = build_query_space_specs(args.pitch_min, args.pitch_max, args.vel_bins)
    max_neg_attempts = args.k_per_audio * max(1, args.max_neg_attempts_factor)

    pending_records_per_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    global_query_keys: List[str] = []

    stats = {
        "num_audio_total": 0, "num_audio_used": 0, "skipped_no_notes": 0, "skipped_bad_split": 0,
        "num_notes_dist": Counter(), "queries_per_split": Counter(),
        "empty_ratio_per_split_sum": defaultdict(float), "empty_ratio_per_split_count": Counter(),
        "family_counter": Counter(), "type_counter": Counter(),
    }

    for r in rows:
        stats["num_audio_total"] += 1
        split = r["split"].strip()
        if split in ("valid", "dev"):
            split = "val"
        if split not in pending_records_per_split:
            stats["skipped_bad_split"] += 1
            continue

        meta_path = data_dir / r["json_path"]
        if not meta_path.exists():
            continue
        notes = stable_sort_notes(extract_notes_from_meta(load_json(meta_path)))

        stats["num_notes_dist"][len(notes)] += 1
        if not notes:
            stats["skipped_no_notes"] += 1
            continue

        positive_keys = build_positive_queries(notes, fps=args.fps, vel_bins=args.vel_bins)
        sampled_keys = sample_k_queries_with_target_empty(
            rng=rng, positive_keys=positive_keys, specs=specs, notes=notes, fps=args.fps,
            vel_bins=args.vel_bins, k=args.k_per_audio, target_empty_ratio=args.target_empty_ratio,
            max_neg_attempts=max_neg_attempts,
        )

        p = Path(r["audio_path"])
        feat_key, uid = f"wav/{p.stem}{p.suffix}", p.stem

        empty_cnt = 0
        local_records = []
        for qk in sampled_keys:
            ans = answer_query(qk, notes, fps=args.fps, vel_bins=args.vel_bins)
            if not ans:
                empty_cnt += 1

            parts = qk.split("/")
            fam = parts[1] if len(parts) > 1 else "OTHER"
            typ = parts[2] if len(parts) > 2 else "OTHER"
            stats["family_counter"][fam] += 1
            stats["type_counter"][f"{fam}/{typ}"] += 1
            global_query_keys.append(qk)

            local_records.append({
                "uid": uid, "split": split, "mert_h5": str(args.mert_h5), "feat_key": feat_key,
                "fps": float(args.fps), "query_key": qk,
                "answer_spans": [[int(s), int(e)] for (s, e) in ans],
                "audio_path": str(r["audio_path"]), "midi_path": str(r["midi_path"]), "json_path": str(r["json_path"]),
            })

        stats["num_audio_used"] += 1
        er = empty_cnt / max(1, len(sampled_keys))
        stats["empty_ratio_per_split_sum"][split] += er
        stats["empty_ratio_per_split_count"][split] += 1
        pending_records_per_split[split].extend(local_records)

    q2id, tree = build_vocab_and_tree(global_query_keys)

    for split, recs in pending_records_per_split.items():
        out_path = out_dir / f"grounding_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for obj in recs:
                obj["query_id"] = int(q2id[obj["query_key"]])
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        stats["queries_per_split"][split] = len(recs)
        print(f"Wrote {out_path} ({len(recs)} query-samples)")

    with open(out_dir / "query_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"query_key_to_id": q2id}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_dir / 'query_vocab.json'} (num_queries={len(q2id)})")

    with open(out_dir / "query_vocab_tree.json", "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_dir / 'query_vocab_tree.json'}")

    inventory = {
        "args": vars(args), "specs": specs,
        "stats": {
            "num_audio_total": stats["num_audio_total"], "num_audio_used": stats["num_audio_used"],
            "skipped_no_notes": stats["skipped_no_notes"], "skipped_bad_split": stats["skipped_bad_split"],
            "num_notes_dist": dict(stats["num_notes_dist"]), "queries_per_split": dict(stats["queries_per_split"]),
            "avg_empty_ratio_per_split": {
                sp: (stats["empty_ratio_per_split_sum"][sp] / max(1, stats["empty_ratio_per_split_count"][sp]))
                for sp in ["train", "val", "test"]
            },
            "family_counter": dict(stats["family_counter"]),
            "type_counter_top20": stats["type_counter"].most_common(20),
        },
    }
    with open(out_dir / "query_inventory.json", "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_dir / 'query_inventory.json'}")


if __name__ == "__main__":
    main()
