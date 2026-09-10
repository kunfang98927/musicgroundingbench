"""Step 3N-8: pick one random QA entry per clip out of llm_test.json, giving
the evaluation subset the paper's test numbers are computed on.

Usage:
    python scripts/08_make_small_testset.py query_dataset_for_llm/llm_test.json \\
        query_dataset_for_llm/llm_test_small.json --seed 1024
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def make_small_testset(input_json: str, output_json: str, seed: int) -> None:
    random.seed(seed)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for item in data:
        grouped[item["audio_path"]].append(item)

    sampled = [random.choice(items) for items in grouped.values()]
    sampled = sorted(sampled, key=lambda x: x["audio_path"])

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"Original size: {len(data)}")
    print(f"Unique audio_path: {len(grouped)}")
    print(f"New size: {len(sampled)}")
    print(f"Saved to: {output_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_json")
    ap.add_argument("output_json")
    ap.add_argument("--seed", type=int, default=1024)
    args = ap.parse_args()
    make_small_testset(args.input_json, args.output_json, seed=args.seed)


if __name__ == "__main__":
    main()
