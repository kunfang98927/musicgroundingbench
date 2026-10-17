#!/usr/bin/env bash
# Generates the MusicGroundingBench-2B clips (MIDI + per-clip metadata) exactly as released.
#
#   ./run_pipeline.sh <dataset_root>
#
# Steps: 1 generate 12000 candidate clips (seed 1024), 2 flag low-quality clips, 3 quarantine them,
# 4 remove the 32 clips whose rendered audio exceeds 10 s (reference list, no audio needed).
# Result: <dataset_root>/{train,val,test}/{midi,meta} with 9007 / 1128 / 1124 clips.
# The question-answer pairs are built from these clips by mgbench_qa (see REPRODUCING.md).
set -euo pipefail
DATASET_ROOT="${1:?usage: run_pipeline.sh <dataset_root>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run() { echo "+ $*"; "$@"; }

run python "$HERE/scripts/01_generate_midi.py" --outdir "$DATASET_ROOT" --target 12000 --seed 1024
run python "$HERE/scripts/02_inspect_and_blacklist.py" "$DATASET_ROOT"
run python "$HERE/scripts/03_quarantine_flagged_samples.py" "$DATASET_ROOT"
run python "$HERE/scripts/04_filter_overlong_samples.py" "$DATASET_ROOT" \
    --from_id_list "$HERE/reference/official_overlong_excluded_ids.txt"
