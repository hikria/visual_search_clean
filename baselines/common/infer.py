"""Shared inference utilities for baselines 1 & 2.

Model-agnostic: the base model is read from configs/model.yaml (currently
UNDECIDED). We use the model's OWN chat template via the HF processor so this
works across Qwen2.5-VL / Qwen3-VL / others without hardcoding a prompt format.

Only our own code lives here — evaluation metrics are NOT here; those come from
the official vendored repos via eval/run_eval.py.
"""
from __future__ import annotations
import json, os, base64
from dataclasses import dataclass
from typing import Iterator

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class ModelCfg:
    model_id: str
    family: str            # "qwen" | "internvl" (drives vision preprocessing)
    backend: str
    dtype: str
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    min_pixels: int
    max_pixels: int
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int


def load_cfg(path: str) -> ModelCfg:
    with open(path) as f:
        c = yaml.safe_load(f)
    bm, g = c["base_model"], c["generation"]
    img = c.get("image", {})
    if bm["model_id"].startswith("TODO"):
        raise SystemExit(
            f"{path}: base_model.model_id is still TODO — set it before running."
        )
    return ModelCfg(
        model_id=bm["model_id"], family=bm.get("family", "qwen"),
        backend=bm["backend"], dtype=bm["dtype"],
        tensor_parallel_size=bm["tensor_parallel_size"],
        max_model_len=bm["max_model_len"],
        gpu_memory_utilization=bm.get("gpu_memory_utilization", 0.90),
        max_num_seqs=bm.get("max_num_seqs", 32),
        min_pixels=img.get("min_pixels", 256 * 28 * 28),
        max_pixels=img.get("max_pixels", 1280 * 28 * 28),
        temperature=g["temperature"], top_p=g["top_p"],
        max_new_tokens=g["max_new_tokens"], seed=g["seed"],
    )


def read_eval(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)]


# --- prompt construction -----------------------------------------------------
# Baseline 1: direct answer (no reasoning). Baseline 2: allow a reasoning span.
DIRECT_SUFFIX = (
    "\nAnswer with the option letter only if this is a multiple-choice "
    "question; otherwise answer concisely."
)
REASONING_SUFFIX = (
    "\nThink step by step, then end with a line 'Answer: <your answer>'."
)


def build_messages(rec: dict, image_root: str, mode: str,
                   min_pixels: int = 0, max_pixels: int = 0) -> list[dict]:
    """Return a chat-messages list for processor.apply_chat_template.

    The image entry carries min/max_pixels so Qwen's process_vision_info applies
    the official resize policy; InternVL ignores these and loads the PIL image.
    """
    img = rec["image_path"]
    if not os.path.isabs(img):
        img = os.path.join(image_root, img)
    q = rec["question"].strip()
    q += REASONING_SUFFIX if mode == "reasoning" else DIRECT_SUFFIX
    image_entry = {"type": "image", "image": img}
    if min_pixels:
        image_entry["min_pixels"] = min_pixels
    if max_pixels:
        image_entry["max_pixels"] = max_pixels
    return [{
        "role": "user",
        "content": [
            image_entry,
            {"type": "text", "text": q},
        ],
    }]


def iter_records(path: str, image_root: str, mode: str) -> Iterator[tuple[dict, list]]:
    for rec in read_eval(path):
        yield rec, build_messages(rec, image_root, mode)
