"""Official-metric adapter for hr_bench_4k.

Native official scorer (vlmeval/dataset/utils/hrbench.py) uses a local-LLM yes/no judge + circular option permutation, requiring the un-permuted option columns our unified set no longer keeps split. We score the normalized letter-MCQ by exact letter match instead.
Scoring reuses MME-RealWorld's official rule-based letter extractor via _mcq
(see _mcq.py header for why the normalized set uses one shared extractor).
"""
from __future__ import annotations
from . import _mcq


def score(predictions: list[dict]) -> dict:
    out = _mcq.score(predictions)
    out["source"] = "hr_bench_4k"
    return out
