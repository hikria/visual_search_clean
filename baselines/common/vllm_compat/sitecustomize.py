"""Auto-imported by every Python process (incl. vLLM spawn workers) when this
directory is on PYTHONPATH.

Fixes a vLLM 0.15.1 <-> transformers 5.5.x incompatibility for the Qwen-VL
family: vLLM's model_executor/models/qwen2_vl.py reads
`image_processor.max_pixels` / `.min_pixels`, but transformers >=5 removed those
attributes (folded into `size = {shortest_edge, longest_edge}` inside __init__).
The result is `AttributeError: 'Qwen2VLImageProcessor' object has no attribute
'max_pixels'` at engine warm-up, for BOTH Qwen2.5-VL and Qwen3-VL.

We re-expose min/max_pixels as read-only properties mapping to the size dict —
exactly vLLM's own fallback (`... or image_processor.size["longest_edge"]`), so
behavior is identical to a processor that still carried the attributes. Only
applied when the attribute is genuinely missing; never overrides a real one.
"""
from __future__ import annotations


def _install() -> None:
    candidates = []
    for mod, cls in [
        ("transformers.models.qwen2_vl.image_processing_qwen2_vl",
         "Qwen2VLImageProcessor"),
        ("transformers.models.qwen2_vl.image_processing_qwen2_vl_fast",
         "Qwen2VLImageProcessorFast"),
        ("transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl_fast",
         "Qwen2_5_VLImageProcessorFast"),
    ]:
        try:
            m = __import__(mod, fromlist=[cls])
            candidates.append(getattr(m, cls))
        except Exception:
            pass

    def _size_get(self, key):
        size = getattr(self, "size", None)
        if size is None:
            return None
        try:
            return size[key]
        except Exception:
            return getattr(size, key, None)

    for C in candidates:
        if not isinstance(getattr(C, "max_pixels", None), property) and \
                "max_pixels" not in C.__dict__:
            try:
                C.max_pixels = property(lambda self: _size_get(self, "longest_edge"))
                C.min_pixels = property(lambda self: _size_get(self, "shortest_edge"))
            except Exception:
                pass


try:
    _install()
except Exception:
    pass


def _install_config_defaults() -> None:
    """vLLM 0.15.1 reads text_config.tie_word_embeddings, but transformers >=5
    Qwen*VLTextConfig no longer carries it. Add a safe class-level default
    (Qwen2.5-VL/Qwen3-VL do NOT tie embeddings -> False) only when missing."""
    for mod, cls in [
        ("transformers.models.qwen2_5_vl.configuration_qwen2_5_vl",
         "Qwen2_5_VLTextConfig"),
        ("transformers.models.qwen2_vl.configuration_qwen2_vl",
         "Qwen2VLTextConfig"),
        ("transformers.models.qwen3_vl.configuration_qwen3_vl",
         "Qwen3VLTextConfig"),
    ]:
        try:
            m = __import__(mod, fromlist=[cls])
            C = getattr(m, cls)
            if "tie_word_embeddings" not in C.__dict__:
                C.tie_word_embeddings = False
        except Exception:
            pass


try:
    _install_config_defaults()
except Exception:
    pass
