"""Step 3N-6: render every query_key from the grounding index into a natural-
language question (English), one branch per query family/type. Pure string
templating - no randomness, no audio.

Usage:
    python scripts/06_query_key_to_text.py \\
        --vocab_path query_dataset_for_baselines/query_vocab.json \\
        --out query_dataset_for_llm/query_text_vocab.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

VEL_BIN_TO_DYN = {0: "ppp", 1: "pp", 2: "p", 3: "mp", 4: "mf", 5: "f", 6: "ff", 7: "fff"}


def _bins_to_dyn_list(csv_bins: str) -> List[str]:
    out = []
    for x in (v.strip() for v in csv_bins.split(",") if v.strip()):
        try:
            out.append(VEL_BIN_TO_DYN.get(int(x), f"bin{x}"))
        except ValueError:
            out.append(x)
    return out


def _pretty_dyn_list(csv_bins: str) -> str:
    xs = _bins_to_dyn_list(csv_bins)
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return f"{xs[0]} or {xs[1]}"
    return ", ".join(xs[:-1]) + f", or {xs[-1]}"


def _pretty_vals_csv(v: str) -> str:
    xs = [x.strip() for x in v.split(",") if x.strip()]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return f"{xs[0]} or {xs[1]}"
    return ", ".join(xs[:-1]) + f", or {xs[-1]}"


def parse_query_key(qk: str) -> Dict[str, Any]:
    parts = qk.split("/")
    out = {"raw": qk, "family": None, "type": None, "subtype": None, "params": {}, "parts": parts}
    if len(parts) >= 3 and parts[0] == "Q":
        out["family"], out["type"] = parts[1], parts[2]
        out["subtype"] = parts[3] if len(parts) > 3 else ""
        for p in parts[3:]:
            if "=" in p:
                k, v = p.split("=", 1)
                out["params"][k] = v
    return out


def _extract_param(info: Dict[str, Any], key: str) -> str | None:
    """Params may appear as params[key], as subtype "key=...", or wrapped
    inside params['prefix']/params['suffix'] like "key=..."."""
    params: Dict[str, str] = info.get("params", {}) or {}
    sub: str = (info.get("subtype") or "").strip()

    if key in params:
        return params[key]
    if sub.startswith(key + "="):
        return sub.split("=", 1)[1]
    for wrap in ("prefix", "suffix"):
        w = (params.get(wrap) or "").strip()
        if w.startswith(key + "="):
            return w.split("=", 1)[1]
    return None


def _ordinal_word(i: int) -> str:
    return {1: "first", 2: "second", 3: "third"}.get(i, f"{i}th")


def _parse_ord_note(info: Dict[str, Any]) -> str | None:
    """Supports idx=1,2,3,-1 / k=1,2,3,last / prefix=kN / suffix=kN."""
    params = info.get("params", {}) or {}
    sub = (info.get("subtype") or "").strip().lower()

    idx = _extract_param(info, "idx")
    if idx is not None:
        try:
            iv = int(idx)
            if iv == -1:
                return "last"
            if iv >= 1:
                return f"idx:{iv}"
        except ValueError:
            pass

    k = _extract_param(info, "k")
    if k is not None and k.strip().lower() in ("1", "2", "3", "last", "first", "second", "third"):
        return k.strip().lower()

    def decode_kN(token: str) -> int | None:
        return int(token[1:]) if token.startswith("k") and token[1:].isdigit() else None

    prefix, suffix = (params.get("prefix") or "").strip().lower(), (params.get("suffix") or "").strip().lower()
    n_pre, n_suf = (decode_kN(prefix) if prefix else None), (decode_kN(suffix) if suffix else None)
    if n_pre is not None:
        return f"prefix:{n_pre}"
    if n_suf is not None:
        return f"suffix:{n_suf}"

    if sub.startswith("prefix="):
        n = decode_kN(sub.split("=", 1)[1])
        if n is not None:
            return f"prefix:{n}"
    if sub.startswith("suffix="):
        n = decode_kN(sub.split("=", 1)[1])
        if n is not None:
            return f"suffix:{n}"

    return sub if sub in ("first", "second", "third", "last") else None


def _normalize_step_word(s: str) -> str:
    s = s.strip().lower()
    if s in ("eq", "equal", "same", "flat"):
        return "flat"
    if s in ("up", "inc", "increase", "higher"):
        return "up"
    if s in ("down", "dec", "decrease", "lower"):
        return "down"
    return s


def _contour_phrase(code: str) -> str:
    code = code.strip().lower()
    if "-" in code:
        a, b = code.split("-", 1)
        return f"{_normalize_step_word(a)} then {_normalize_step_word(b)}"
    return _normalize_step_word(code)


def query_key_to_text(qk: str) -> str:
    info = parse_query_key(qk)
    fam, typ, sub = info["family"], info["type"], (info["subtype"] or "").strip()

    def fallback() -> str:
        return f"Locate the time segment(s) that match the query: {qk}"

    if not fam or not typ:
        return fallback()

    if fam == "REL":
        if typ == "PITCH":
            return {"argmax": "Locate the highest-pitch note.", "argmin": "Locate the lowest-pitch note."}.get(sub, fallback())
        if typ == "VEL":
            return {"argmax": "Locate the loudest note.", "argmin": "Locate the softest note."}.get(sub, fallback())
        if typ == "VELBIN":
            return {"argmax": "Locate the note(s) in the loudest dynamics (loudness).",
                    "argmin": "Locate the note(s) in the softest dynamics (loudness)."}.get(sub, fallback())
        if typ == "DUR":
            return {"argmax": "Locate the longest note.", "argmin": "Locate the shortest note."}.get(sub, fallback())
        return fallback()

    if fam == "ORD" and typ == "NOTE":
        token = _parse_ord_note(info)
        if token is None:
            return fallback()
        if token in ("1", "first"):
            return "Locate the first note."
        if token in ("2", "second"):
            return "Locate the second note."
        if token in ("3", "third"):
            return "Locate the third note."
        if token == "last":
            return "Locate the last note."
        if token.startswith("idx:"):
            return f"Locate the {_ordinal_word(int(token.split(':', 1)[1]))} note."
        if token.startswith("prefix:"):
            n = int(token.split(":", 1)[1])
            return {2: "Locate the first two notes.", 3: "Locate the first three notes."}.get(n, f"Locate the first {n} notes.")
        if token.startswith("suffix:"):
            n = int(token.split(":", 1)[1])
            return {2: "Locate the last two notes.", 3: "Locate the last three notes."}.get(n, f"Locate the last {n} notes.")
        return fallback()

    if fam == "ABS":
        if typ == "PITCH":
            v = _extract_param(info, "vals")
            return f"Locate the note(s) whose pitch is {_pretty_vals_csv(v)} (MIDI)." if v is not None else fallback()
        if typ == "VEL":
            b = _extract_param(info, "bin")
            if b is not None:
                return f"Locate the note(s) whose dynamics (loudness) is {_pretty_dyn_list(b)}."
            bs = _extract_param(info, "bins")
            if bs is not None:
                return f"Locate the note(s) whose dynamics (loudness) is {_pretty_dyn_list(bs)}."
            v = _extract_param(info, "vals")
            return f"Locate the note(s) whose velocity is {_pretty_vals_csv(v)}." if v is not None else fallback()
        if typ == "VELBIN":
            v = _extract_param(info, "vals")
            return f"Locate the note(s) whose dynamics (loudness) bin is {_pretty_vals_csv(v)}." if v is not None else fallback()
        return fallback()

    if fam == "PAT":
        if typ == "PITCH_DIFF":
            d = _extract_param(info, "delta")
            return f"Locate the pair of notes separated by {d} semitone(s)." if d is not None else fallback()
        if typ == "PAIR":
            return {
                "pitch_closest": "Locate the pair of notes with the smallest pitch interval.",
                "time_closest": "Locate the pair of notes closest in time.",
                "pitch_farthest": "Locate the pair of notes with the largest pitch interval.",
                "time_farthest": "Locate the pair of notes farthest apart in time.",
            }.get(sub, fallback())
        if typ == "CONTOUR":
            pitch, vel = _extract_param(info, "pitch"), _extract_param(info, "vel")
            if pitch is not None:
                return f"Locate the segment where the pitch contour is {_contour_phrase(pitch)}."
            if vel is not None:
                return f"Locate the segment where the dynamics (loudness) contour is {_contour_phrase(vel)}."
            t = _extract_param(info, "type")
            return f"Locate the segment where the contour is {_contour_phrase(t)}." if t is not None else fallback()
        if typ == "INT":
            order = _extract_param(info, "order")
            if order is not None:
                o = order.strip().lower()
                if o in ("asc", "ascending"):
                    return "Locate the segment where the time gaps between notes increase (short → long)."
                if o in ("desc", "descending"):
                    return "Locate the segment where the time gaps between notes decrease (long → short)."
                if o in ("eq", "equal", "constant", "flat"):
                    return "Locate the segment where the time gaps between notes are (approximately) equal."
                return f"Locate the segment where the time-gap pattern is: {order}."
        return fallback()

    return fallback()


def load_query_keys_from_vocab(vocab_path: str) -> Tuple[List[str], Dict[str, int]]:
    obj = json.load(open(vocab_path, "r", encoding="utf-8"))
    qmap = obj.get("query_key_to_id", {})
    if not isinstance(qmap, dict) or not qmap:
        raise ValueError("Invalid query_vocab.json (missing query_key_to_id)")
    items = sorted(qmap.items(), key=lambda kv: int(kv[1]))
    return [k for k, _ in items], qmap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab_path", type=str, required=True, help="path to query_vocab.json")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    query_keys, qkey_to_id = load_query_keys_from_vocab(args.vocab_path)

    qkey_to_text: Dict[str, str] = {}
    fallback_count = 0
    family_counter, type_counter = Counter(), Counter()

    for qk in query_keys:
        txt = query_key_to_text(qk)
        qkey_to_text[qk] = txt
        if txt.endswith(qk):
            fallback_count += 1
        info = parse_query_key(qk)
        fam, typ = info.get("family") or "UNK", info.get("type") or "UNK"
        family_counter[fam] += 1
        type_counter[f"{fam}/{typ}"] += 1

    out_obj = {
        "language": "en", "num_queries": len(query_keys), "query_key_to_id": qkey_to_id,
        "query_key_to_text": qkey_to_text,
        "stats": {"fallback_count": fallback_count, "family_counter": dict(family_counter),
                   "type_counter_top20": type_counter.most_common(20)},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path} (num_queries={len(query_keys)})")
    print(f"Fallback mappings: {fallback_count}/{len(query_keys)}")


if __name__ == "__main__":
    main()
