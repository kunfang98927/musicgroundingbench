"""Reading prediction files and pairing them with the reference QA items."""
import json
import sys
from typing import Dict, List, Optional


def load_predictions(path: str) -> List[Dict]:
    """A JSON list, or JSON lines, of objects with "sample_id", "question", and "prediction"."""
    text = open(path, encoding="utf-8").read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list or JSON lines")
    return data


def align(qa: List[Dict], preds: List[Dict], allow_missing: bool = False) -> List[Optional[str]]:
    """The prediction text for every QA item, matched on (sample_id, question).

    A test set can ask several questions about one clip, so the question text is part of the key.
    Questions without a prediction are an error unless allow_missing is set; they are then scored
    as empty answers.
    """
    by_key = {}
    for p in preds:
        key = (p.get("sample_id"), p.get("question"))
        if key in by_key:
            raise ValueError(f"duplicate prediction for {key}")
        by_key[key] = p.get("prediction") or ""
    out, missing = [], 0
    for item in qa:
        key = (item["sample_id"], item["question"])
        if key not in by_key:
            missing += 1
        out.append(by_key.get(key, ""))
    extra = len(by_key) - (len(qa) - missing)
    if missing and not allow_missing:
        raise SystemExit(f"{missing} of {len(qa)} questions have no prediction (match on sample_id + question); "
                         "use --allow-missing to score them as empty answers")
    if missing:
        print(f"[warning] {missing} questions without a prediction were scored as empty answers", file=sys.stderr)
    if extra:
        print(f"[warning] {extra} predictions do not match any question and were ignored", file=sys.stderr)
    return out
