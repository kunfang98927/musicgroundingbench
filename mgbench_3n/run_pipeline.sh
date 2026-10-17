#!/usr/bin/env bash
# Generates the MusicGroundingBench-3N clips (MIDI + per-clip metadata) exactly as released.
#
#   ./run_pipeline.sh <dataset_root>
#
# Steps: 1 generate 6000 clips of one to three notes (seed 20260123), 2 write metadata.csv,
# 3-4 read-only checks (no overlap between splits, dataset statistics).
# Result: <dataset_root>/{train,val,test}/{midi,meta} with 4200 / 900 / 900 clips and metadata.csv.
# The question-answer pairs are built from these clips by mgbench_qa (see REPRODUCING.md).
set -euo pipefail
DATASET_ROOT="${1:?usage: run_pipeline.sh <dataset_root>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run() { echo "+ $*"; "$@"; }

run python "$HERE/scripts/01_generate_midi.py" --outdir "$DATASET_ROOT" --target 6000
run python "$HERE/scripts/02_build_metadata.py" --data_dir "$DATASET_ROOT"
run python "$HERE/scripts/03_check_splits.py" --data_dir "$DATASET_ROOT"
run python "$HERE/scripts/04_dataset_stats.py" --data_root "$DATASET_ROOT"
