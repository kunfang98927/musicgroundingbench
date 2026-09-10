#!/usr/bin/env bash
# Runs the full MGBench-3N pipeline end to end (steps 1-4 core, 5-8 QA).
# Activate your Python environment first (see README.md), then:
#
#   ./run_pipeline.sh <dataset_root>
#
# Reproduces the official dataset byte-for-byte: 4200/900/900 train/val/test
# clips, 84000/18000/18000 grounding QA pairs, all with the documented
# defaults (generation seed 20260123, grounding-index seed 1234, k=20
# queries/clip, fps=75, test-small seed 1024). See docs/REPRODUCING.md.
set -euo pipefail

DATASET_ROOT="${1:?usage: run_pipeline.sh <dataset_root>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() { echo "+ $*"; "$@"; }

echo "=== MGBench-3N pipeline -> $DATASET_ROOT ==="

echo "--- 3N-1: generate MIDI ---"
run python "$HERE/scripts/01_generate_midi.py" --outdir "$DATASET_ROOT" --target 6000

echo "--- 3N-2: build metadata.csv ---"
run python "$HERE/scripts/02_build_metadata.py" --data_dir "$DATASET_ROOT"

echo "--- 3N-3: check splits don't overlap (report only) ---"
run python "$HERE/scripts/03_check_splits.py" --data_dir "$DATASET_ROOT"

echo "--- 3N-4: dataset stats (report only) ---"
run python "$HERE/scripts/04_dataset_stats.py" --data_root "$DATASET_ROOT"

echo "--- 3N-5: build grounding index (query_key + answer spans) ---"
run python "$HERE/scripts/05_build_grounding_index.py" \
    --data_dir "$DATASET_ROOT" --index_csv "$DATASET_ROOT/metadata.csv" \
    --out_dir "$DATASET_ROOT/query_dataset_for_baselines" \
    --mert_h5 "$DATASET_ROOT/piano-melody-3notes_mert_mean13.h5" --fps 75 --k_per_audio 20

echo "--- 3N-6: render query text ---"
run python "$HERE/scripts/06_query_key_to_text.py" \
    --vocab_path "$DATASET_ROOT/query_dataset_for_baselines/query_vocab.json" \
    --out "$DATASET_ROOT/query_dataset_for_llm/query_text_vocab.json"

echo "--- 3N-7: prepare final LLM QA files ---"
run python "$HERE/scripts/07_prepare_llm_dataset.py" \
    --train_jsonl "$DATASET_ROOT/query_dataset_for_baselines/grounding_train.jsonl" \
    --val_jsonl   "$DATASET_ROOT/query_dataset_for_baselines/grounding_val.jsonl" \
    --test_jsonl  "$DATASET_ROOT/query_dataset_for_baselines/grounding_test.jsonl" \
    --query_vocab "$DATASET_ROOT/query_dataset_for_llm/query_text_vocab.json" \
    --out_prefix  "$DATASET_ROOT/query_dataset_for_llm/llm" --root "$DATASET_ROOT"

echo "--- 3N-8: build the test-small evaluation subset ---"
run python "$HERE/scripts/08_make_small_testset.py" \
    "$DATASET_ROOT/query_dataset_for_llm/llm_test.json" \
    "$DATASET_ROOT/query_dataset_for_llm/llm_test_small.json" --seed 1024

cat <<EOF

=== MGBench-3N pipeline done: $DATASET_ROOT ===
Audio synthesis (PianoTeq) is out of scope, but nothing above needed it - the
audio_path column is only ever carried through as a string. Optional next step
(also audio-free, see docs/REPRODUCING.md):

  09_build_generalize_testset.py      the out-of-distribution generalization
                                       test set, using reference/query_generalize_ood.json
EOF
