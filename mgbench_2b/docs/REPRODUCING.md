# Reproducing MGBench-2B

This covers the two-bar subset (MGBench-2B) end to end: symbolic generation,
quality filtering, and QA construction. Audio synthesis (PianoTeq 9 Stage) is
proprietary and out of scope - everything here works on MIDI and JSON, and
produces the exact same dataset PianoTeq would later be run over.

## Quick start

```bash
pip install -r requirements.txt
./run_pipeline.sh two_bar_dataset          # target=12000, seed=1024 (the official config)
```

That runs steps 01-12 and 16 below. It takes a few minutes for the full
12000-sample run. Use a smaller `--target`/different `--seed` for a fast
local test:

```bash
./run_pipeline.sh /tmp/test_dataset 200 42
```

## Pipeline stages

Each stage is one script in `scripts/`, run in order, all read-only except
for their own output directory. Run any script with `--help` for its exact
flags.

| # | Script | What it does |
|---|--------|---------------|
| 01 | `01_generate_midi.py` | Procedurally generates two-bar MIDI clips. Each clip samples a random combination of *control attributes* (meter, tempo, mode, structure, ...), realizes a note sequence, then **retries with a fresh sample** if it fails `validate_sample` (see Filter logic below). Splits accepted samples 80/10/10 into train/val/test. |
| 02 | `02_inspect_and_blacklist.py` | Re-derives every sample's attributes from its MIDI and flags samples as `hard` / `soft` / `review` (see below). |
| 03 | `03_quarantine_flagged_samples.py` | Moves `soft`-flagged samples out of the active dataset into `quarantine_soft_blacklist/`. `review`-flagged samples are kept (they're borderline, not broken) and only listed for manual spot-checking. |
| 04 | `04_dataset_stats.py` | Reports the distribution of every control attribute and derived statistic. Read-only. |
| 05 | `05_build_slim_meta.py` | Derives a `meta_slim/*.json` per sample: per-note beat alignment, chord-tone membership, interval sequences, etc. - the shared basis every QA fact is computed from (`src/qa_summary.py`). |
| 06 | `06_check_meta_slim_consistency.py` | Recomputes `qa_summary` fresh from the MIDI and diffs it against what's stored, to catch drift between `meta_slim/` and `src/qa_summary.py`. Read-only. |
| 07 | `07_build_qa_facts.py` | Turns `meta_slim` into two fact lists per sample: `grounding_facts` (note-span answers - "which notes are the exact repeat?") and `understanding_facts` (value answers + an evidence span - "what's the time signature?"). Each fact carries a `confidence` (high/low) - low-confidence facts (e.g. harmony facts on a sample not optimized for harmonic salience) are excluded from QA rendering in the next step. |
| 08 | `08_render_grounding_qa.py` | Samples up to 8 high-confidence grounding facts per clip (quota-balanced across the ABS/REL/ORD/PAT query families) and renders each into a question + token-span answer. |
| 09 | `09_render_understanding_qa.py` | Same idea for understanding facts: up to 8 per clip, rendered into a question + free-text answer + evidence span. |
| 10 | `10_check_grounding_facts.py` | Recomputes what each grounding fact's answer *should* be and flags mismatches. Read-only. |
| 11 | `11_check_understanding_facts.py` | Same, for understanding facts. Read-only. |
| 12 | `12_summarize_qa_facts.py` | Tallies facts by category/concept/answer for a quick label-distribution sanity check. Read-only. |
| 13 | `13_filter_overlong_samples.py` | **Needs rendered audio** (or the reference ID list - see below). Removes clips too long for the audio encoder. |
| 14 | `14_build_metadata_csv.py` | **Needs rendered audio.** Flat CSV index of every clip. |
| 15 | `15_sample_small_testset.py` | Picks the concept-balanced `test-small` / `test-tiny` evaluation subsets (one QA per clip) out of a rendered `test.json`. The released release used seed 20260417 with 1124 and 300 - see below. |
| 16 | `16_plot_final_stats.py` | Plots sample- and QA-level distributions. Read-only. |

## Filter logic

**Generation retries (step 01, `src/validation.py:validate_sample`)** - a
candidate is rejected and re-sampled if it fails a structural check (bad
timing, out-of-scale pitch, polyphonic overlap, wrong bar count) or either
of two musical checks: fewer than 5 notes, or a gap of more than 1.5 beats
between consecutive notes. The official run rejects about 2.4% of attempts,
mostly on those last two.

**Inspection buckets (step 02, `classify_sample`)** - every flag is a cheap,
transparent threshold on already-computed statistics, no models involved:

| Bucket | Meaning | Example flags |
|--------|---------|----------------|
| `hard` | Structurally broken (mismatched counts, MIDI failed to parse) | any `events:`/`basic:`/`midi:`/`pair:` issue |
| `soft` | Clearly low-quality; quarantined out of the dataset (step 03) | `too_few_notes`, `zero_pitch_span`, `single_unique_pitch`, `exact_repeat_bar_note_count_mismatch`, chord-tone ratio `< 0.65` on an explicit-harmony sample |
| `review` | Borderline; kept, just listed for spot-checking | extreme mean velocity, mean note duration `> 2.2` beats, pitch span `>= 24` semitones, only 2 unique pitches, chord-tone ratio in `[0.65, 0.85)` |

The official run flags ~6% `soft` (quarantined) and ~8% `review` (kept).

**Fact confidence (step 07)** - a fact is marked low-confidence when its
label is technically present but not necessarily perceptible by ear on this
subset (e.g. a harmony label on a clip that wasn't generated with harmonic
salience as a control target). Only high-confidence facts are ever rendered
into QA (steps 08-09).

## Reproducing the official counts exactly

`run_pipeline.sh <root> 12000 1024` reproduces the official pre-audio-filter
split sizes exactly: 9034 / 1130 / 1127 (train/val/test) - verified by
diffing this repo's output against the original project's dataset copy,
matching down to the exact accept/reject counts in `sanity_report.txt`.

The published paper numbers (9007 / 1128 / 1124) are after step 13 removes
32 clips (27 train / 2 val / 3 test) whose rendered audio exceeds 10s. Since
audio synthesis is out of scope here, that exact list is checked in at
`reference/official_overlong_excluded_ids.txt`, so you can reproduce the
official final counts without rendering any audio:

```bash
python scripts/13_filter_overlong_samples.py two_bar_dataset \
    --from_id_list reference/official_overlong_excluded_ids.txt
```

(This only reproduces the official *counts*, generated with seed=1024/
target=12000 - it is a list of sample_ids, not a general duration model, so
it isn't meaningful against a dataset generated with different settings.)

## The re-entrant loop

If step 13 removes any samples (either from real audio or the reference
list), the facts and rendered QA for the *remaining* samples don't change,
but the removed samples must disappear from them too. Re-run:

```bash
python scripts/07_build_qa_facts.py two_bar_dataset --overwrite
python scripts/08_render_grounding_qa.py two_bar_dataset --overwrite   # --seed defaults to 20260413
python scripts/09_render_understanding_qa.py two_bar_dataset --overwrite
```

Note steps 08/09 use their own default seed (20260413), separate from step
01's generation seed (1024) - passing `--seed 1024` here is a mistake that
silently gives you a different (but internally consistent) QA sampling.

Then build the evaluation subsets (this is what the paper's test numbers are
computed on - `test-small.json`, exactly one question per clip, not the larger
`test.json` candidate pool):

```bash
for task in grounding_qa understanding_qa; do
  python scripts/15_sample_small_testset.py two_bar_dataset/$task/test.json \
      two_bar_dataset/$task/test-small.json --target_num_samples 1124 --seed 20260417
  python scripts/15_sample_small_testset.py two_bar_dataset/$task/test.json \
      two_bar_dataset/$task/test-tiny.json  --target_num_samples 300  --seed 20260417
done
```

## How exact is the QA content itself

Validated against a real copy of the official dataset (including its
`grounding_qa`/`understanding_qa`, not just split counts):

Everything below is **100% byte-identical** to the official release:

| Artifact | Result |
|---|---|
| MIDI (steps 01-03) | 11,259 / 11,259 files identical |
| train/val/test split | identical sample-id sets; 0 samples on the wrong side |
| `grounding_qa/{train,val,test,all}.json` | 68,972 / 68,972 entries identical |
| `understanding_qa/{train,val,test,all}.json` | 90,072 / 90,072 entries identical |
| `*/query_registry.json`, `*/summary.json` | identical, including the recorded config |
| `*/test-small.json`, `*/test-tiny.json` (step 15) | identical (1,124 and 300 entries) |

Two things are worth knowing if you ever modify these scripts:

- **The render seed is not the generation seed.** Step 01 uses `--seed 1024`;
  steps 08/09 use their own `--seed 20260413` (their default) and step 15 uses
  `--seed 20260417`. Passing the generation seed to a render step silently
  produces a different-but-plausible dataset instead of an error.
- **Concept ordering is load-bearing, not cosmetic.** Each concept's candidate
  list is shuffled from a per-sample RNG, so the *number* of shuffles performed
  before a given pick determines that pick. When a category's quota is filled
  early, the loop breaks and never shuffles the remaining concepts - so moving a
  concept earlier or later in `concept_priority_within_category` changes draws
  downstream and silently changes which questions get asked. The orders in the
  scripts were recovered by matching against the released data; don't "tidy" them.

## Output layout

```
two_bar_dataset/
  {train,val,test}/
    midi/                 one .mid per clip
    meta/                 full generation metadata
    meta_slim/            derived per-note fields (step 05)
    grounding_facts/       per-clip fact pool (step 07)
    understanding_facts/
  grounding_qa/{train,val,test,all}.json     rendered QA (step 08)
  understanding_qa/{train,val,test,all}.json rendered QA (step 09)
  *_results/, *_summary/, final_statistics/  reports from the read-only steps
```
