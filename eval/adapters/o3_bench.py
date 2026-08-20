"""Official-metric adapter for o3_bench.

O3-Bench (InSight-o3) has no standalone GitHub scorer; the HF card口径 is single-letter MCQ (options up to F, incl. 'No right choice'). _mcq widens the letter class to the letters present, so F is handled.
Scoring reuses MME-RealWorld's official rule-based letter extractor via _mcq
(see _mcq.py header for why the normalized set uses one shared extractor).
"""
from __future__ import annotations
from . import _mcq


def score(predictions: list[dict]) -> dict:
    out = _mcq.score(predictions)
    out["source"] = "o3_bench"
    return out
