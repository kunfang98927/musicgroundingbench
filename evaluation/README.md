# Evaluation

Scoring code for MusicGroundingBench, as described in Section 4 of the paper. Grounding
answers are scored with rules; understanding answers are judged by Qwen2.5-7B-Instruct.

```bash
pip install -r requirements.txt      # the LLM judge additionally needs torch, transformers, accelerate and a GPU
```

Run the commands from the repository root. The QA files come from the
[Zenodo record](https://doi.org/10.5281/zenodo.23377894); report results on the `test` sets.

## Prediction files

A JSON list (or JSON lines) with one object per question:

```json
[{"sample_id": "test_000209", "question": "Locate the note(s) with the highest pitch.",
  "prediction": "From 1.08 seconds to 1.48 seconds; From 4.43 seconds to 4.85 seconds."}]
```

Predictions are paired with the QA items on `sample_id` and `question` (a set can ask more than
one question about a clip). Every question needs a prediction; `--allow-missing` scores missing
ones as empty answers.

## Grounding

```bash
python -m evaluation.grounding --qa MGBench-2B/qa/grounding/test.json --pred preds.json
python -m evaluation.grounding --qa MGBench-3N/qa/test.json           --pred preds.json
```

Time spans are read from the answer text in the form `X second(s) to Y second(s)`, for example
`From 1.08 seconds to 1.48 seconds; From 4.43 seconds to 4.85 seconds.` Ask your model to answer
in this form; for the zero-shot models in the paper we used this system prompt:

```
You are a careful music listening assistant.
Answer using only information that can be inferred from the audio.
Be concise and precise.
If the question asks you to locate a note or event in time, answer in the form:
From X.XX seconds to Y.YY seconds.
If the question does not ask for a time span, answer with a short direct answer only.
Do not add extra explanation unless necessary.
```

When no note satisfies a query, the reference answer has no span; the expected answer is a
statement that there is no such note (e.g. `There is no such note in this clip.`).

| Metric | Definition |
|---|---|
| IFR | share of answers that contain at least one time span, or that state there is no matching note (`There is no such note in this clip.`, `No matching note was found in this clip.`, `No time span.`, `None`) |
| MergedIoU | per question, the IoU of the regions covered by the predicted and the reference spans after merging overlapping spans within each set (0 when both are empty), averaged over questions |
| P / R / F1 @ IoU ≥ t | spans are matched greedily one to one (highest IoU first) while IoU ≥ t; precision = matched / predicted spans, recall = matched / reference spans, pooled over all questions |
| on / off MAE @ t | mean absolute onset and offset error (ms) of the matched spans |

Results are printed overall and per group (MusicGroundingBench-2B: music concept;
MusicGroundingBench-3N: query category ABS / REL / ORD / PAT); `--out results.json` saves them.

## Understanding

```bash
python -m evaluation.judge         --qa MGBench-2B/qa/understanding/test.json --pred preds.json --out judged.json
python -m evaluation.understanding --qa MGBench-2B/qa/understanding/test.json --judged judged.json
```

`evaluation.judge` asks [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
(greedy decoding, float16) whether each answer follows the instruction and whether it matches
the reference answer; paraphrases and musically equivalent answers count as correct. A judge
output that is not valid JSON falls back to an exact string comparison with the reference.
The judge takes about one second per question on a single GPU, and `--out` is resumed if the
run is interrupted. Use `--model` to point to a local copy of the judge model.

| Metric | Definition |
|---|---|
| IFR | share of questions whose answer the judge marks as following the instruction |
| Accuracy | share of all questions answered correctly (an answer that does not follow the instruction counts as wrong), overall and per music concept |
