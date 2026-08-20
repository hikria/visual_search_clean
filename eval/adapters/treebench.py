"""Official-metric adapter for treebench.

Native official scorer: inference_treebench.py extracts <answer>..</answer> (DOTALL, upper) with exact letter match, else falls back to raw output. Our unified prompt asks for the letter directly (no <answer> tag), so the robust MME letter extractor is used; both reduce to exact letter match on clean single-letter outputs.
Scoring reuses MME-RealWorld's official rule-based letter extractor via _mcq
(see _mcq.py header for why the normalized set uses one shared extractor).
"""
from __future__ import annotations
from . import _mcq


def score(predictions: list[dict]) -> dict:
    out = _mcq.score(predictions)
    out["source"] = "treebench"
    return out
