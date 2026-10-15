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
> Listen to test clips from both subsets and compare the evaluated models'
> predictions with the ground truth, one item per concept.

**MusicGroundingBench (MGBench)** is a controlled benchmark for **note-level temporal
grounding** and **music understanding** in audio-language models. Its clips are
algorithmically generated monophonic piano music with exact symbolic alignment: every clip
comes with its MIDI and note list, and every reference answer is computed from the notes and
independently re-verified. Grounding questions ask *when* the notes that satisfy a musical
query occur; understanding questions ask *what* a clip contains.

Paper: *Note-Level Temporal Grounding of Musical Concepts in Large Audio-Language Models*
(Kun Fang, Ziyu Wang, Ichiro Fujinaga) — [arXiv:2608.29480](https://arxiv.org/abs/2608.29480).

## Overview

| | MGBench-3N | MGBench-2B |
|---|---|---|
| Material | 1–3 sequential notes, all 88 piano keys | two-bar monophonic excerpts |
| Clips (train / val / test) | 6,000 (4,200 / 900 / 900) | 10,844 (8,646 / 1,086 / 1,112) |
| Grounding QA | 69,145 | 39,974 |
| Understanding QA | – | 43,567 |
| Evaluation set `test` | 900 | 1,035 grounding, 1,078 understanding |

* **MGBench-3N** isolates note-level localization. Query categories: ABS (absolute pitch),
  REL (pitch or loudness rank), ORD (position of a note or of the first/last notes), PAT (pitch
  difference, contour, closest/farthest note pair, widening/narrowing intervals).
* **MGBench-2B** adds rhythmic, tonal, intervallic, and harmonic structure, with grounding and
  understanding questions on the same clips over seven concepts: rhythm, tonality, interval,
  bar comparison, harmony, repetition, and harmonic pattern.

Audio was rendered from the MIDI files with Pianoteq 9 Stage (WAV, 44.1 kHz, 16-bit, stereo).
Splits are clip-disjoint.

## Download

Everything is on Zenodo: <https://doi.org/10.5281/zenodo.23377894> (about 18 GB in total; the
audio of each split is spread over several standalone zip archives). Unzip all archives into
one folder:

```
MGBench-2B/
  qa/grounding/{train,val,test,test-full}.json
  qa/understanding/{train,val,test,test-full}.json
  {train,val,test}/wav/<sample_id>.wav
  {train,val,test}/midi/<sample_id>.mid
  {train,val,test}/meta/<sample_id>.json
MGBench-3N/
  qa/{train,val,test,test-full}.json
  {train,val,test}/wav/<sample_id>.wav
  {train,val,test}/midi/<sample_id>.mid
  {train,val,test}/meta/<sample_id>.json
```

For evaluation only, the QA archives and the test audio (about 2.1 GB) are enough:

```bash
base=https://zenodo.org/records/23377894/files
for f in MGBench-2B_qa.zip MGBench-3N_qa.zip \
         MGBench-2B_audio_test_part0{1..5}-of-05.zip \
         MGBench-3N_audio_test_part0{1..3}-of-03.zip; do
  wget -O "$f" "$base/$f?download=1"
done
for f in MGBench-*.zip; do unzip -q "$f"; done
```

`SHA256SUMS` (archives) and `MANIFEST.sha256` (every unpacked file) on the record let you verify
the download.

## QA format

Each QA file is a JSON list. The `audio_path` of an item is relative to its subset folder.
Shortened examples from the `test` sets:

```jsonc
// MGBench-3N grounding
{"sample_id": "test_000676_n1_p42", "audio_path": "test/wav/test_000676_n1_p42.wav",
 "question": "Locate the last note.",
 "answer": "From 0.30 second to 1.77 second.",
 "answer_spans_sec": [[0.295455, 1.768182]]}

// MGBench-2B grounding
{"sample_id": "test_000209", "audio_path": "test/wav/test_000209.wav", "concept": "interval",
 "question": "Locate the note(s) with the highest pitch.",
 "answer_text": "From 1.08 seconds to 1.48 seconds; From 4.43 seconds to 4.85 seconds.",
 "answer_spans_sec": [[1.081818, 1.484091], [4.434091, 4.85]]}

// MGBench-2B understanding
{"sample_id": "test_000003", "audio_path": "test/wav/test_000003.wav", "concept": "interval",
 "question": "How many semitones is the largest melodic leap in this clip?",
 "answer_text": "8", "gold_answer": 8, "gold_answer_type": "integer",
 "evidence_note_indices": [1, 2]}
```

Every item also carries the clip's notes (`notes`: MIDI pitch, velocity, start and end in
seconds), paraphrases of the question, and the question type. An empty `answer_spans_sec` means
that no note satisfies the query. The README.md on the Zenodo record lists every field and the
per-clip metadata.

## Evaluation

* **`test`** is the evaluation set: one fixed question per test clip (3N: 900, 2B grounding:
  1,035, 2B understanding: 1,078). Report results on it.
* **`test-full`** contains every question on the test clips (3N: 10,478, 2B grounding: 3,381,
  2B understanding: 4,928); `test` is a subset of it.

Protocol (Section 4 of the paper):

* **Grounding.** Predicted and reference answers are sets of time intervals. We report the
  instruction-following rate (IFR), MergedIoU (IoU of the regions covered after merging
  overlapping spans), and span-level precision, recall, and F1 under greedy one-to-one
  matching at IoU ≥ 0.5 and ≥ 0.7, together with the onset/offset mean absolute error (ms) of
  matched spans. For free-form model outputs, spans are extracted with Qwen2.5-7B-Instruct.
* **Understanding.** Qwen2.5-7B-Instruct judges each answer for instruction following and
  semantic correctness. We report IFR and accuracy over all test questions, overall and per
  concept.

Reference evaluation scripts will be added to this repository.

## Repository contents

* `docs/` — the demo page (served with GitHub Pages).
* `mgbench_3n/`, `mgbench_2b/` — generation of the clips' MIDI.
* `mgbench_v2/` — construction and validation of the released question–answer pairs: the
  rule-based answer oracles, question templates, and validation gates.

Audio rendering (Pianoteq 9 Stage) is commercial software and not part of this repository. A
step-by-step reproduction guide will be added.

## License

* **Code** in this repository: [MIT License](LICENSE).
* **Dataset** (on Zenodo, including the audio clips of the demo page):
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use MusicGroundingBench, please cite the paper:

```bibtex
@article{fang2026notelevel,
  title   = {Note-Level Temporal Grounding of Musical Concepts in Large Audio-Language Models},
  author  = {Fang, Kun and Wang, Ziyu and Fujinaga, Ichiro},
  journal = {arXiv preprint arXiv:2608.29480},
  year    = {2026},
  doi     = {10.48550/arXiv.2608.29480}
}
```
