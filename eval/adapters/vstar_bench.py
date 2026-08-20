"""Official-metric adapter for vstar_bench.

Native official scorer (vstar_bench_eval.py) is per-option LM-likelihood argmin with GT at index 0 — inapplicable to a generative letter answer. Our unified set stores V* as letter-MCQ with an explicit gold letter, scored by exact letter match.
Scoring reuses MME-RealWorld's official rule-based letter extractor via _mcq
(see _mcq.py header for why the normalized set uses one shared extractor).
"""
from __future__ import annotations
from . import _mcq


def score(predictions: list[dict]) -> dict:
    out = _mcq.score(predictions)
    out["source"] = "vstar_bench"
    return out
