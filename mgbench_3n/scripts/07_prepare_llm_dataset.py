"""Step 3N-7: combine the grounding index with the rendered question text
into the final LLM-ready QA files (llm_train.json / llm_val.json / llm_test.json).
Deterministic - just joins two files already produced by steps 3N-5/3N-6.

Usage:
    python scripts/07_prepare_llm_dataset.py \\
        --train_jsonl query_dataset_for_baselines/grounding_train.jsonl \\
        --val_jsonl   query_dataset_for_baselines/grounding_val.jsonl \\
        --test_jsonl  query_dataset_for_baselines/grounding_test.jsonl \\
        --query_vocab query_dataset_for_llm/query_text_vocab.json \\
        --out_prefix  query_dataset_for_llm/llm \\
        --root piano-melody-3notes
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_json(arr: List[Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)


def build_question(query_key: str, vocab: Dict[str, Any]) -> str:
    if query_key and query_key in vocab["query_key_to_text"]:
        return vocab["query_key_to_text"][query_key]
    return query_key


def spans_to_answer_text(answer_spans: List[List[int]], fps: float) -> str:
    if not answer_spans:
        return "No time span."
    parts = []
    for sp in answer_spans:
        s_f, e_f = int(sp[0]), int(sp[1])
        s, e = (s_f / fps, e_f / fps) if fps and fps > 0 else (float(s_f), float(e_f))
        parts.append(f"From {s:.2f} second to {e:.2f} second")
    return "; ".join(parts) + "." if parts else "No time span."


def spans_to_answer_token(answer_spans: List[List[int]]) -> str:
    if not answer_spans:
        return "<no_time_span>"
    return "".join(f"<box><|T{int(s)}|><|T{int(e)}|></box>" for s, e in answer_spans) or "<no_time_span>"


def parse_notes_from_meta(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for ev in meta.get("events", []):
        if "onset" not in ev or "offset" not in ev:
            continue
        out.append({"pitch": int(ev.get("pitch", -1)), "velocity": int(ev.get("velocity", -1)),
                     "start": float(ev["onset"]), "end": float(ev["offset"])})
    return out


def resolve_existing_path(p: str, root: str, jsonl_dir: str) -> Optional[str]:
    """Try p as-is, then root/p, then jsonl_dir/p, then parent(jsonl_dir)/p."""
    if not p:
        return None
    candidates = [p]
    if root:
        candidates.append(os.path.join(root, p))
    if jsonl_dir:
        candidates.append(os.path.join(jsonl_dir, p))
        candidates.append(os.path.join(os.path.dirname(jsonl_dir), p))
    return next((c for c in candidates if c and os.path.exists(c)), None)


def convert_entry(entry: Dict[str, Any], vocab: Dict[str, Any], root: str, jsonl_dir: str) -> Dict[str, Any]:
    query_key = entry.get("query_key", "") or ""
    query_id = entry.get("query_id")
    fps = float(entry.get("fps") or 0.0)
    answer_spans = entry.get("answer_spans", []) or []

    notes: List[Dict[str, Any]] = []
    jp = resolve_existing_path(entry.get("json_path", "") or "", root=root, jsonl_dir=jsonl_dir)
    if jp is not None:
        notes = parse_notes_from_meta(load_json(jp))

    return {
        "question": build_question(query_key, vocab),
        "answer": spans_to_answer_text(answer_spans, fps=fps),
        "question_token": "<|Q" + str(query_id) + "|>",
        "answer_token": spans_to_answer_token(answer_spans),
        "query_key": query_key,
        "query_id": int(query_id) if query_id is not None else -1,
        "answer_spans": answer_spans,
        "audio_path": entry.get("audio_path") or entry.get("feat_key") or "",
        "notes": notes,
    }


def convert_file(jsonl_path: str, vocab: Dict[str, Any], root: str) -> List[Dict[str, Any]]:
    jsonl_dir = os.path.dirname(os.path.abspath(jsonl_path))
    return [convert_entry(it, vocab=vocab, root=root, jsonl_dir=jsonl_dir) for it in load_jsonl(jsonl_path)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--query_vocab", required=True, help="query_text_vocab.json from step 3N-6")
    ap.add_argument("--out_prefix", default="data_llm/grounding")
    ap.add_argument("--root", default="", help="dataset root for resolving json_path")
    args = ap.parse_args()

    vocab = load_json(args.query_vocab)

    for split, jsonl_path in [("train", args.train_jsonl), ("val", args.val_jsonl), ("test", args.test_jsonl)]:
        out = convert_file(jsonl_path, vocab=vocab, root=args.root)
        out_path = f"{args.out_prefix}_{split}.json"
        write_json(out, out_path)
        print(f"Wrote: {out_path} ({len(out)})")


if __name__ == "__main__":
    main()
