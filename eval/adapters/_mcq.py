"""Shared MCQ scorer for the five letter-MCQ sources
(mme_realworld_lite, hr_bench_4k, treebench, o3_bench, vstar_bench).

Our unified eval set normalized every one of these benchmarks into a SINGLE
canonical letter-MCQ format: options are embedded in the question text and the
gold `answer` is one uppercase letter, with the fixed instruction suffix
"Answer with the option's letter from the given choices directly."

Because of that normalization the sources' *native* official scorers are not
directly applicable to a generative letter answer:
  - V*Bench's official metric is per-option LM-likelihood argmin (needs the raw
    model, not free text);
  - HR-Bench's official metric is a local-LLM yes/no judge + circular option
    permutation (needs the un-permuted option columns we no longer keep split).
So for the normalized generative outputs we reuse the ONE official rule-based
letter extractor that IS meant for free text: MME-RealWorld's
`extract_characters_regex` (evaluation/eval_your_results.py), reproduced
verbatim below and only generalized so the accepted letter set matches the
options actually present in each question (MME hardcodes A-E; O3-Bench needs F).

Correctness = exact match of extracted letter to the gold letter (micro acc),
exactly as MME-RealWorld does (`cnt = ground_truth == text`).
"""
from __future__ import annotations
import re, string

# Parse "(A) foo" / "(B) bar" OR "A. foo" / "B. bar" enumerations. The dot style
# may put several options on ONE line ("A. .. B. .. C. .."), so we scan for the
# consecutive letters A,B,C,... in order rather than anchoring to line starts.
_PAREN = re.compile(r"\(([A-Z])\)\s*([^\(\n]*)")


def parse_choices(question: str) -> list[str]:
    """Return official-MME-style choice strings ['(A) foo', '(B) bar', ...] in
    letter order, parsed from the (options-embedded) question text."""
    found: dict[str, str] = {}
    for m in _PAREN.finditer(question):
        found.setdefault(m.group(1), m.group(2).strip())
    if len(found) >= 2:
        letters = sorted(found)
        return [f"({L}) {found[L]}" for L in letters]

    # dot / paren-less style, possibly multiple options per line
    spans: list[tuple[str, int, int]] = []
    pos = 0
    for L in string.ascii_uppercase:
        m = re.search(r"(?:(?<=\s)|^)" + L + r"[\.\):]\s", question[pos:])
        if not m:
            break
        spans.append((L, pos + m.start(), pos + m.end()))
        pos += m.end()
    out = []
    for i, (L, _s, e) in enumerate(spans):
        end = spans[i + 1][1] if i + 1 < len(spans) else len(question)
        text = question[e:end].strip()
        # drop the trailing instruction line if it bled into the last option
        text = re.split(r"\n\s*Answer with", text)[0].strip()
        out.append(f"({L}) {text}")
    return out


def extract_answer(s: str, choices: list[str]) -> str:
    """MME-RealWorld's official extract_characters_regex, verbatim in structure,
    with the [ABCDE] class widened to the letters present in `choices`."""
    letters = "".join(c[1] for c in choices) or "ABCDE"
    cls = f"[{letters}]"
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option is",
        "Best answer:", "Best option:",
    ]
    for p in answer_prefixes:
        s = s.replace(p, "")
    if len(s.split()) > 10 and not re.search(cls, s):
        return ""
    matches = re.search(cls, s)
    if matches is None:
        for choice in choices:  # substring-in-choice fallback -> its letter
            if s.lower() in choice.lower():
                return choice[1]
        return ""
    return matches[0]


def score(predictions: list[dict]) -> dict:
    n = correct = 0
    for r in predictions:
        choices = parse_choices(r.get("question", ""))
        pred = extract_answer(r.get("prediction") or r.get("raw_output") or "",
                              choices)
        gt = (r.get("gt_answer") or "").strip().upper()
        n += 1
        if pred and pred.upper() == gt:
            correct += 1
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "correct": correct,
        "metric": "mcq_letter_exact (MME-RealWorld extract_characters_regex)",
    }
