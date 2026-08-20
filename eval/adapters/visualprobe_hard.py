"""Official-metric adapter for VisualProbe (hard split), from Mini-o3.

Mini-o3's REPORTED VisualProbe-hard accuracy is open-ended and uses a GPT-4o
LLM judge (verl/utils/reward_score/general_qa_tool.py):
  1. extract the prediction from the first <answer>..</answer> tag (re.DOTALL);
  2. call the judge with the official SYSTEM_PROMPT + QUERY_PROMPT below,
     filling question / ground_truth / prediction;
  3. correct iff the judge reply contains '1' after "Score:".
(The repo's general_qa_tool_mc.py exact/letter string-match is only the TRAINING
reward, not the reported metric — we keep it as an offline fallback.)

Adaptation: our unified prompt does not force <answer> tags, so extraction is
"use the <answer> span if present, else the whole stripped prediction" before
judging — otherwise tagless-but-correct answers would all score 0.

Judge model: env VS_JUDGE_MODEL (default 'qwen-max'), via the OpenAI-compatible
endpoint VS_JUDGE_BASE_URL (default DashScope), key from DASHSCOPE_API_KEY /
QWEN_API_KEY (loaded from the repo-root .env if present; see .env.example).
Set VS_JUDGE=0 to force the offline rule-based fallback. The `metric` field in
the result records which path actually ran.
"""
from __future__ import annotations
import os, re, json

SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of "
    "generative outputs for question-answer pairs.\nYour task is to compare the "
    "predicted answer with the correct answer and determine if they match "
    "meaningfully. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the correct "
    "answer.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate the "
    "correctness of the prediction compared to the answer."
)
QUERY_PROMPT = (
    "I will give you a question related to an image and the following text as "
    "inputs:\n\n1. **Question Related to the Image**: {question}\n2. **Ground Truth "
    "Answer**: {ground_truth}\n3. **Model Predicted Answer**: {prediction}\n\nYour "
    "task is to evaluate the model's predicted answer against the ground truth "
    "answer, based on the context provided by the question related to the image. "
    "Consider the following criteria for evaluation:\n- **Relevance**: Does the "
    "predicted answer directly address the question posed, considering the "
    "information provided by the given question?\n- **Accuracy**: Compare the "
    "predicted answer to the ground truth answer. You need to evaluate from the "
    "following two perspectives:\n(1) If the ground truth answer is open-ended, "
    "consider whether the prediction accurately reflects the information given in "
    "the ground truth without introducing factual inaccuracies. If it does, the "
    "prediction should be considered correct.\n(2) If the ground truth answer is a "
    "definitive answer, strictly compare the model's prediction to the actual "
    "answer. Pay attention to unit conversions such as length and angle, etc. As "
    "long as the results are consistent, the model's prediction should be deemed "
    "correct.\n**Output Format**:\nYour response should include an integer score "
    "indicating the correctness of the prediction: 1 for correct and 0 for "
    "incorrect. Note that 1 means the model's prediction strictly aligns with the "
    "ground truth, while 0 means it does not.\nThe format should be \"Score: 0 or 1\""
)

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# .env at the repo root (eval/adapters/ -> ../../.env); override with VS_ENV.
_ENV = os.environ.get("VS_ENV", os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def _extract(pred: str) -> str:
    m = _ANSWER.search(pred or "")
    return (m.group(1) if m else (pred or "")).strip()


def _rule_match(pred: str, gt: str) -> bool:
    """Official training-reward match (general_qa_tool_mc.inner_acc_reward)."""
    gt = (gt or "").strip()
    p = (pred or "").strip()
    if p == gt:
        return True
    m = re.match(r"^\(([A-Z])\).*$", p, re.DOTALL)
    if m and m.group(1) == gt:
        return True
    m = re.match(r"^([A-Z])\..*$", p, re.DOTALL)
    if m and m.group(1) == gt:
        return True
    return False


def _load_key() -> str | None:
    for k in ("DASHSCOPE_API_KEY", "QWEN_API_KEY"):
        if os.environ.get(k):
            return os.environ[k]
    if os.path.isfile(_ENV):
        for line in open(_ENV):
            line = line.strip()
            for k in ("DASHSCOPE_API_KEY", "QWEN_API_KEY"):
                if line.startswith(k + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _judge(client, model, question, gt, pred) -> bool | None:
    prompt = QUERY_PROMPT.format(question=question, ground_truth=gt, prediction=pred)
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
        )
        txt = (resp.choices[0].message.content or "").lower()
    except Exception:
        return None
    if "score:" not in txt:
        return "1" in txt and "0" not in txt.split("1")[0]
    tail = txt.split("score:")[-1].strip().split("\n")[0].strip()
    return "1" in tail


def score(predictions: list[dict]) -> dict:
    use_judge = os.environ.get("VS_JUDGE", "1") != "0"
    key = _load_key() if use_judge else None
    client = model = None
    if use_judge and key:
        try:
            from openai import OpenAI
            base = os.environ.get(
                "VS_JUDGE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1")
            model = os.environ.get("VS_JUDGE_MODEL", "qwen-max")
            client = OpenAI(api_key=key, base_url=base)
        except Exception:
            client = None

    n = correct = 0
    judged = 0
    for r in predictions:
        pred = _extract(r.get("prediction") or r.get("raw_output") or "")
        gt = r.get("gt_answer") or ""
        q = r.get("question") or ""
        ok = None
        if client is not None:
            ok = _judge(client, model, q, gt, pred)
            if ok is not None:
                judged += 1
        if ok is None:  # offline / judge error -> official mc rule fallback
            ok = _rule_match(pred, gt)
        n += 1
        correct += int(bool(ok))

    path = (f"gpt-judge({model})" if judged == n and n
            else f"rule-fallback (judged {judged}/{n})")
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "correct": correct,
        "source": "visualprobe_hard",
        "metric": f"open-ended, Mini-o3 official: {path}",
    }
