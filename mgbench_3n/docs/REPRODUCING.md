# Reproducing MGBench-3N

The three-note subset: clips with one, two, or three notes, queried with a
rule-based grounding QA task (no "understanding" task on this subset - the
clips are too simple to ask conceptual questions about). Audio synthesis
(PianoTeq) is out of scope here too, but nothing in this pipeline actually
needs rendered audio - `audio_path` is only ever carried through as a string.

## Quick start

```bash
pip install -r requirements.txt
./run_pipeline.sh piano-melody-3notes
```

Reproduces the official dataset byte-for-byte: 4200/900/900 train/val/test
clips and 84000/18000/18000 grounding QA pairs, verified against the released
`llm_train.json` / `llm_val.json` / `llm_test.json` / `llm_test_small.json`.

## Pipeline stages

| # | Script | What it does |
|---|--------|---------------|
| 1 | `01_generate_midi.py` | Generates clips from three template pools (1/2/3 notes, fixed timing centers + small jitter), mixed 1:1:1, split 70/15/15. |
| 2 | `02_build_metadata.py` | Writes `metadata.csv` (paths + first/last note time + split) - read from MIDI-derived JSON only. |
| 3 | `03_check_splits.py` | Confirms no clip is byte-identical across splits. Read-only. |
| 4 | `04_dataset_stats.py` | Note-count/pitch/interval/contour distributions and timing sanity checks. Read-only. |
| 5 | `05_build_grounding_index.py` | The core QA-construction step: samples ~20 query/answer pairs per clip across four query families (see below), balanced toward a ~15% empty-answer rate for negatives. |
| 6 | `06_query_key_to_text.py` | Renders every query key into an English question. Pure templating. |
| 7 | `07_prepare_llm_dataset.py` | Joins the index with the question text into the final `llm_{train,val,test}.json`. |
| 8 | `08_make_small_testset.py` | Picks one QA pair per clip from `llm_test.json` - this is what the paper's test numbers are computed on. |
| 9 | `09_build_generalize_testset.py` | Optional: an out-of-distribution test set using hand-authored query paraphrases (`reference/query_generalize_ood.json`) instead of the rule-based ones. |

## The four query families

Every question is one of:

- **ABS** - absolute properties ("locate the note(s) with pitch X").
- **REL** - superlatives ("locate the highest / loudest / longest note").
- **ORD** - position ("locate the first note", "the last two notes").
- **PAT** - patterns across notes ("the pair separated by N semitones", "the
  segment where the pitch contour goes up-then-down").

Step 5 samples per clip from whichever of these are actually true for that
clip (the "positive" pool), plus a target ~15% drawn from the full query
space regardless of truth (giving intentionally-empty answers as negatives),
then trims back toward that 15% target if rejection sampling overshot it.

## Output layout

```
piano-melody-3notes/
  {train,val,test}/{midi,meta}/       one clip each
  metadata.csv
query_dataset_for_baselines/
  grounding_{train,val,test}.jsonl    step 5's raw query index
  query_vocab.json, query_vocab_tree.json, query_inventory.json
query_dataset_for_llm/
  query_text_vocab.json               step 6
  llm_{train,val,test}.json           step 7 - final QA
  llm_test_small.json                 step 8 - the paper's eval set
  llm_generalize.json                 step 9 (optional)
```
