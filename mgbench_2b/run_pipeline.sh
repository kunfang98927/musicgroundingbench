#!/usr/bin/env bash
# Runs the full symbolic MGBench-2B pipeline end to end (steps 01-12, 16).
# Activate your Python environment first (see README.md), then:
#
#   ./run_pipeline.sh <dataset_root> [target] [seed]
#
# Defaults (target=12000, seed=1024) are the official generation config -
# they reproduce the paper's pre-audio-filter split sizes (9034/1130/1127
# train/val/test). See docs/REPRODUCING.md for the audio-dependent steps
# (13-15) that come after this script.
set -euo pipefail

DATASET_ROOT="${1:?usage: run_pipeline.sh <dataset_root> [target] [seed]}"
TARGET="${2:-12000}"
SEED="${3:-1024}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() { echo "+ $*"; "$@"; }

echo "=== MGBench-2B pipeline -> $DATASET_ROOT (target=$TARGET, seed=$SEED) ==="

echo "--- 01: generate MIDI ---"
run python "$HERE/scripts/01_generate_midi.py" --outdir "$DATASET_ROOT" --target "$TARGET" --seed "$SEED"

echo "--- 02: inspect + flag low-quality samples ---"
run python "$HERE/scripts/02_inspect_and_blacklist.py" "$DATASET_ROOT"

echo "--- 03: quarantine flagged samples ---"
run python "$HERE/scripts/03_quarantine_flagged_samples.py" "$DATASET_ROOT"

echo "--- 04: dataset stats (report only) ---"
run python "$HERE/scripts/04_dataset_stats.py" "$DATASET_ROOT"

echo "--- 05: build slim metadata ---"
run python "$HERE/scripts/05_build_slim_meta.py" "$DATASET_ROOT"

echo "--- 06: check slim metadata consistency (report only) ---"
run python "$HERE/scripts/06_check_meta_slim_consistency.py" "$DATASET_ROOT"

echo "--- 07: build QA facts ---"
run python "$HERE/scripts/07_build_qa_facts.py" "$DATASET_ROOT"

# NOTE: steps 08/09 deliberately do NOT take $SEED. They have their own render
# seed (20260413); passing the generation seed here silently yields a different
# QA sampling. See docs/REPRODUCING.md.
echo "--- 08: render grounding QA ---"
run python "$HERE/scripts/08_render_grounding_qa.py" "$DATASET_ROOT"

echo "--- 09: render understanding QA ---"
run python "$HERE/scripts/09_render_understanding_qa.py" "$DATASET_ROOT"

echo "--- 10: check grounding facts (report only) ---"
run python "$HERE/scripts/10_check_grounding_facts.py" "$DATASET_ROOT"

echo "--- 11: check understanding facts (report only) ---"
run python "$HERE/scripts/11_check_understanding_facts.py" "$DATASET_ROOT"

echo "--- 12: summarize QA facts (report only) ---"
run python "$HERE/scripts/12_summarize_qa_facts.py" "$DATASET_ROOT"

echo "--- 16: plot final stats (report only) ---"
run python "$HERE/scripts/16_plot_final_stats.py" "$DATASET_ROOT"

cat <<EOF

=== Symbolic pipeline done: $DATASET_ROOT ===
Audio synthesis (PianoTeq) is out of scope for this repo, so 3 steps are
left for you to run manually - see docs/REPRODUCING.md:

  13_filter_overlong_samples.py   remove clips too long for the audio encoder
                                   (no audio? use --from_id_list reference/official_overlong_excluded_ids.txt
                                   with the default target/seed above to get the official 9007/1128/1124 counts)
  14_build_metadata_csv.py         flat CSV index (needs rendered audio)
  15_sample_small_testset.py       the test-small / test-tiny evaluation subsets

If step 13 removes any samples, re-run 07, 08, 09 with --overwrite, then build
the evaluation subsets - both are spelled out in docs/REPRODUCING.md. Doing
that reproduces the released dataset byte-for-byte.
EOF
