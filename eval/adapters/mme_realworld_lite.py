"""Official-metric adapter for mme_realworld_lite.

Native official scorer: evaluation/eval_your_results.py -> extract_characters_regex + micro accuracy. Reproduced verbatim in _mcq (this IS the source of the shared extractor).
Scoring reuses MME-RealWorld's official rule-based letter extractor via _mcq
(see _mcq.py header for why the normalized set uses one shared extractor).
"""
from __future__ import annotations
from . import _mcq


def score(predictions: list[dict]) -> dict:
    out = _mcq.score(predictions)
    out["source"] = "mme_realworld_lite"
    return out
