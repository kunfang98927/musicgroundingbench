"""LLM judge for MusicGroundingBench understanding answers (Section 4.2 of the paper).

Qwen2.5-7B-Instruct reads the question, the model's answer, and the reference answer and
returns two binary labels: whether the answer follows the instruction (answers the question)
and whether it is correct. Paraphrases and musically equivalent answers count as correct.
Decoding is greedy, so a run is repeatable on the same hardware and library versions.

Usage:
    python -m evaluation.judge --qa MGBench-2B/qa/understanding/test.json --pred preds.json --out judged.json
    python -m evaluation.understanding --qa MGBench-2B/qa/understanding/test.json --judged judged.json
"""
import argparse
import json
import os
import re

from .io import load_predictions, align

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPT_TEMPLATE = """You are an expert evaluator for music understanding tasks.

You are given:
1. A question about a music clip
2. A model's prediction
3. The ground truth answer

Your task is to evaluate the prediction.

--------------------------------
Step 1: Instruction Following
--------------------------------
Determine whether the prediction actually answers the question.

- If the prediction is irrelevant, vague, or does NOT answer the question -> instruction_following_score = 0
- Otherwise -> instruction_following_score = 1

--------------------------------
Step 2: Accuracy (ONLY if instruction_following_score = 1)
--------------------------------
Evaluate how correct the prediction is compared to the ground truth.

IMPORTANT:
- Do NOT require exact wording match
- Accept paraphrases and equivalent musical descriptions
- Focus on semantic meaning

Use ONLY the following discrete scores:

1.0 -> Completely correct (fully matches ground truth meaning, or is a valid and accurate paraphrase)
0.0 -> Incorrect (does not match ground truth meaning)

--------------------------------
Step 3: Confidence
--------------------------------
Estimate how confident you are:

1.0 -> Very confident (clear evidence in text)
0.5 -> Somewhat confident (some evidence, but not fully clear)
0.0 -> Not confident (no clear evidence)

Confidence should reflect certainty of your judgment, NOT correctness of the prediction.

--------------------------------
Step 4: Error Reason
--------------------------------
Briefly explain why the prediction is correct or incorrect.

Requirements:
- Keep it concise (one short sentence)
- Focus on key difference from ground truth
- If fully correct, state "correct" or "semantically equivalent"
- Maximum 20 words

--------------------------------
Output format (STRICT)
--------------------------------

Return ONLY valid JSON:

{{
  "instruction_following_score": 0 or 1,
  "accuracy_score": one of [0.0, 1.0],
  "confidence": one of [0.0, 0.5, 1.0],
  "error_reason": "string"
}}

--------------------------------
Rules:
--------------------------------
- If instruction_following_score = 0:
    - accuracy_score MUST be 0.0
    - error_reason MUST explain why it does not answer the question
- Do NOT include explanations outside JSON
- Do NOT include extra text
- Output MUST be valid JSON. If your output is not valid JSON, the evaluation is invalid.

--------------------------------
Now evaluate:
--------------------------------

Question:
{question}

Prediction:
{prediction}

Ground Truth:
{answer}
"""
# The prompt used for the paper ends these two lines with two spaces; kept so that the judge input
# is identical (written out here because editors tend to strip trailing whitespace).
PROMPT_TEMPLATE = (PROMPT_TEMPLATE
                   .replace("(clear evidence in text)\n", "(clear evidence in text)  \n")
                   .replace("(some evidence, but not fully clear)\n", "(some evidence, but not fully clear)  \n"))


def extract_json(text: str):
    """The first JSON object in the judge output: a ```json fenced block first, else the first balanced {...}."""
    if not text:
        return None
    for block in re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        try:
            return json.loads(block)
        except Exception:
            pass
    for start in [i for i, ch in enumerate(text) if ch == "{"]:
        depth, in_string, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    return None


class Judge:
    def __init__(self, model: str = DEFAULT_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto").eval()

    def __call__(self, question: str, prediction: str, reference: str) -> dict:
        prompt = PROMPT_TEMPLATE.format(question=question, prediction=prediction, answer=reference)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
        verdict = extract_json(text)
        if verdict is None:  # unparseable judge output: fall back to an exact string match
            same = prediction == reference
            verdict = {"instruction_following_score": int(same), "accuracy_score": float(same), "confidence": float(same),
                       "error_reason": "correct (fallback)" if same else "Fallback to string match (incorrect)"}
        return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", required=True, help="reference QA file, e.g. MGBench-2B/qa/understanding/test.json")
    ap.add_argument("--pred", required=True, help="predictions: JSON list or JSON lines with sample_id, question, prediction")
    ap.add_argument("--out", required=True, help="judged items (JSON); an existing file is resumed")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="judge model: Hugging Face id or local path")
    ap.add_argument("--allow-missing", action="store_true", help="judge questions without a prediction as empty answers")
    a = ap.parse_args()
    qa = json.load(open(a.qa, encoding="utf-8"))
    preds = align(qa, load_predictions(a.pred), a.allow_missing)
    done = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else []
    if done and [(d["sample_id"], d["question"]) for d in done] != [(q["sample_id"], q["question"]) for q in qa[:len(done)]]:
        raise SystemExit(f"{a.out} does not belong to {a.qa}; remove it to start over")
    judge = Judge(a.model)
    for i in range(len(done), len(qa)):
        item, pred = qa[i], preds[i]
        verdict = judge(item["question"], pred, item["answer_text"])
        done.append({"sample_id": item["sample_id"], "question": item["question"], "prediction": pred,
                     "reference": item["answer_text"], **verdict})
        if (i + 1) % 50 == 0 or i + 1 == len(qa):
            tmp = a.out + ".tmp"
            json.dump(done, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            os.replace(tmp, a.out)
            print(f"judged {i + 1}/{len(qa)}", flush=True)


if __name__ == "__main__":
    main()
