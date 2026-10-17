# Reproducing the dataset

MusicGroundingBench is built in three stages:

1. **Clips (MIDI).** Algorithmic generators write the MIDI and a small per-clip metadata file
   (`mgbench_2b/`, `mgbench_3n/`).
2. **Audio.** Every MIDI file was rendered with Pianoteq 9 Stage (WAV, 44.1 kHz, 16-bit, stereo).
   Pianoteq is commercial software and is not part of this repository; the rendered audio is on
   [Zenodo](https://doi.org/10.5281/zenodo.23377894).
3. **Question-answer pairs.** `mgbench_qa/` computes every answer from the realized notes of a
   clip with a rule-based oracle, selects balanced questions, and runs the release checks.

Stages 1 and 3 need no audio. One command rebuilds the released QA files from scratch and compares
them byte for byte with the release:

```bash
pip install -r requirements.txt
./reproduce.sh work      # about 10 minutes on one CPU core
```

The last lines must report `OK` for all 12 files of `mgbench_qa/release_qa.sha256`. The steps of
the script are described below.

## 1. MusicGroundingBench-2B clips

```bash
bash mgbench_2b/run_pipeline.sh clips_2b
```

| Step | Script | Result |
|---|---|---|
| 1 | `01_generate_midi.py --target 12000 --seed 1024` | 12,000 candidate clips (MIDI + `meta/<id>.json`), split 80 / 10 / 10 |
| 2 | `02_inspect_and_blacklist.py` | flags clearly low-quality clips (e.g. fewer than 5 notes, one repeated pitch) |
| 3 | `03_quarantine_flagged_samples.py` | moves the flagged clips out: 9,034 / 1,130 / 1,127 clips |
| 4 | `04_filter_overlong_samples.py --from_id_list reference/official_overlong_excluded_ids.txt` | removes the 32 clips whose rendered audio is longer than 10 s: 9,007 / 1,128 / 1,124 clips |

Step 4 needs rendered audio to find the long clips; the list of the 32 removed clips is included,
so the release is reproduced without audio. Generation is deterministic for a given seed.

**Transposed clips.** The generator realizes "bar 2 is a transposed copy of bar 1" only loosely,
so 200 additional clips were built from generated clips of the same split: bar 1 is kept and bar
2 repeats it one bar later, transposed by 1 to 7 semitones
(`mgbench_qa/scripts/make_transposed_midi.py`). These clips are included as files in
`mgbench_2b/reference/transposed/` and the QA build reads them from there: they were selected
with an earlier revision of the clip checks, so rerunning the script picks different source clips.

## 2. MusicGroundingBench-3N clips

```bash
bash mgbench_3n/run_pipeline.sh clips_3n
```

`01_generate_midi.py` writes 6,000 clips of one, two, or three notes (seed 20260123; 4,200 / 900 /
900), `02_build_metadata.py` writes `metadata.csv`, and steps 3-4 are read-only checks (no clip
appears in two splits; dataset statistics).

## 3. Question-answer pairs

```bash
python -m mgbench_qa.scripts.build_qa_2b --clips clips_2b --transposed mgbench_2b/reference/transposed --out MGBench-2B
python -m mgbench_qa.scripts.build_qa_3n --clips clips_3n --out MGBench-3N
```

Each build writes `<out>/qa/.../{train,val,test,test-full}.json` (the same paths as in the Zenodo
archives), `query_vocab.json` (question type to `query_id`), and `build_report.json`.

* **Answers.** `oracle2b.py` / `oracle3n.py` read only the realized notes (from the MIDI file) and,
  for MusicGroundingBench-2B, the timing frame (meter, tempo, start of bar 1, bar length). They
  never use the generator's control variables, and they return no answer when a question would be
  ambiguous for a clip; such questions are skipped. Clips with a note shorter than 0.1 s, overlapping
  notes, or an empty bar are not used (10,954 of the 2B clips are usable; 10,844 receive
  questions).
* **Question families.** `qa2b.py` / `qa3n.py` define the question families and their candidate
  questions per clip; `phrasing.py` holds the question wording.
* **Selection.** For every split and family, the build keeps candidates whose answer does not
  change when every note boundary is moved by up to 5 ms, balances the answer classes (also within
  every value of a question parameter or stated context, so that the question text does not
  predict the answer), limits empty and yes/no answers, and (MusicGroundingBench-2B) gives every music concept the
  same number of questions in val and test. `test` takes one question per test clip.

### Release checks

```bash
python -m mgbench_qa.gates.run_gates    --qa MGBench-2B --out MGBench-2B/checks
python -m mgbench_qa.gates.run_gates_3n --qa MGBench-3N --out MGBench-3N/checks
python -m unittest mgbench_qa.tests.test_oracle2b
```

The checks write `GATES_REPORT.md` and a JSON file with every number:

* every answer is derived a second time by separate code (`gates/independent.py`), from the
  released JSON alone;
* the question text states every parameter and context the answer depends on, and nothing that
  gives the answer away;
* answer distributions: majority-class share per family, yes/no share per concept, empty-answer
  share;
* shortcut baselines: a classifier that sees only the question text (understanding), and a
  predictor that always answers a family's median span (grounding), must stay near chance.

## Per-clip metadata on Zenodo

The `meta/<sample_id>.json` files in the Zenodo archives hold the notes as read from the MIDI file
(pitch, velocity, start, end) and, for MusicGroundingBench-2B, the timing frame from the clip's
generation metadata (`time_signature`, `tempo_bpm`, `bar_duration_sec`, `music_start_sec`,
`music_end_sec`).

## Tested environment

The byte-for-byte comparison was run with Python 3.10.13, pretty_midi 0.2.10 (mido 1.3.3),
numpy 1.26.4, scipy 1.15.1, and scikit-learn 1.5.2. Other versions of pretty_midi or numpy may
change the generated MIDI or the 3N clips.
