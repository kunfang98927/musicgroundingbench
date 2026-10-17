#!/usr/bin/env bash
# Rebuilds the released MusicGroundingBench question-answer files from scratch and checks them
# byte for byte against the release (no audio and no download needed).
#
#   ./reproduce.sh <work_dir>
#
# Writes <work_dir>/clips_2b, <work_dir>/clips_3n (generated MIDI), <work_dir>/MGBench-2B/qa and
# <work_dir>/MGBench-3N/qa (the QA files, same paths as on Zenodo), and the release checks in
# <work_dir>/MGBench-*/checks. Takes about 10 minutes on one CPU core.
set -euo pipefail
WORK="${1:?usage: reproduce.sh <work_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
run() { echo "+ $*"; "$@"; }
cd "$HERE"

echo "=== 1/4 MusicGroundingBench-2B clips"
run bash mgbench_2b/run_pipeline.sh "$WORK/clips_2b"
echo "=== 2/4 MusicGroundingBench-3N clips"
run bash mgbench_3n/run_pipeline.sh "$WORK/clips_3n"
echo "=== 3/4 question-answer pairs and release checks"
run python -m mgbench_qa.scripts.build_qa_2b --clips "$WORK/clips_2b" --transposed mgbench_2b/reference/transposed --out "$WORK/MGBench-2B"
run python -m mgbench_qa.scripts.build_qa_3n --clips "$WORK/clips_3n" --out "$WORK/MGBench-3N"
run python -m mgbench_qa.gates.run_gates --qa "$WORK/MGBench-2B" --out "$WORK/MGBench-2B/checks"
run python -m mgbench_qa.gates.run_gates_3n --qa "$WORK/MGBench-3N" --out "$WORK/MGBench-3N/checks"
echo "=== 4/4 comparison with the released QA files"
(cd "$WORK" && sha256sum -c "$HERE/mgbench_qa/release_qa.sha256")
