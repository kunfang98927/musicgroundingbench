# MusicGroundingBench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23377894.svg)](https://doi.org/10.5281/zenodo.23377894)

> ### ▶ Dataset (Zenodo): **<https://doi.org/10.5281/zenodo.23377894>**
>
> Audio, MIDI, per-clip metadata, and all question–answer files of MGBench-3N and
> MGBench-2B (16,844 clips, 152,686 QA pairs; CC BY 4.0). The record's README.md
> describes the file layout and the QA format.
>
> ### ▶ Demo page: **<https://kunfang98927.github.io/musicgroundingbench/>**
>
> Listen to test clips from both subsets and compare every reported model's
> actual prediction against the ground truth, one item per concept (the same
> models as Table 1, grounding, and Table 2, understanding, in the paper).

Code to reproduce **MusicGroundingBench (MGBench)**: a benchmark for
note-level temporal grounding and music understanding on short piano clips.
MGBench has two independent subsets, each in its own self-contained folder -
pick whichever you need, or both.

Both subsets are procedurally generated from a fixed seed and validated
against transparent rules - there's no model or manual curation in the loop.
Audio synthesis (PianoTeq 9 Stage) is proprietary and out of scope for this
repo: everything here operates on MIDI and JSON, and both pipelines below are
verified **byte-for-byte identical** to the released dataset without ever
touching rendered audio.

## MGBench-3N

Clips with one, two, or three notes, queried with note-level grounding
questions ("locate the loudest note", "the pair separated by 5 semitones", ...).

```bash
pip install -r requirements.txt
cd mgbench_3n && ./run_pipeline.sh piano-melody-3notes
```

Full details: **[mgbench_3n/docs/REPRODUCING.md](mgbench_3n/docs/REPRODUCING.md)**.

## MGBench-2B

Two-bar clips with a richer control-attribute spec (meter, tempo, mode,
structure, ...), queried with both **grounding QA** (find a note-span, e.g.
*the two notes that repeat exactly*) and **understanding QA** (free-text
answer + evidence span, e.g. *what's the time signature?*).

```bash
pip install -r requirements.txt
cd mgbench_2b && ./run_pipeline.sh two_bar_dataset
```

That alone reproduces the pre-audio-filter dataset; two documented follow-up
steps then land on the released dataset exactly. Full details, including the
filter logic behind every quality-control step: **[mgbench_2b/docs/REPRODUCING.md](mgbench_2b/docs/REPRODUCING.md)**.

## Layout

Each subset is fully self-contained - its own `scripts/`, `reference/`,
`docs/`, and pipeline script, sharing only this README and `requirements.txt`:

```
mgbench_3n/
  run_pipeline.sh
  scripts/      01_generate_midi.py ... 09_build_generalize_testset.py
  reference/    query_generalize_ood.json (hand-authored, not derivable)
  docs/         REPRODUCING.md
mgbench_2b/
  run_pipeline.sh
  scripts/      01_generate_midi.py ... 16_plot_final_stats.py
  src/          generation + validation + shared QA-derivation logic
  reference/    official_overlong_excluded_ids.txt
  docs/         REPRODUCING.md
```

## Citation

```
(paper citation - to be added)
```
