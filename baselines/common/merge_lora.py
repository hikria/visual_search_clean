#!/usr/bin/env python3
"""Merge a VERL/EasyR1 LoRA adapter + base model into a standalone HF checkpoint.

VERL saves each training ckpt under <ckpt>/actor/ with:
    lora_adapter/                 -> standard PEFT LoRA (adapter_config.json + .safetensors)
    huggingface/                  -> base model tokenizer / processor / config
    model_world_size_4_rank_*.pt  -> FSDP shards of the (frozen) base — NOT needed here
    optim_world_size_4_rank_*.pt  -> optimizer states — NOT needed here

The base model's weights come from the HF cache (adapter_config.base_model_name_or_path),
so we do NOT touch the FSDP shards. We just fuse the PEFT adapter into the base and
dump a normal AutoModelForCausalLM-ready directory that `baselines/common/run_baseline.py`
can load into vLLM for eval.

Run env: easyr1 (peft==0.20.0 exactly matches the adapter_config.peft_version).
Usage:
  python baselines/common/merge_lora.py \
      --ckpt results/b3_grpo/train_ckpt/global_step_350/actor \
      --out  results/evals/merged/b3_grpo_step350 \
      --base Qwen/Qwen2.5-VL-7B-Instruct
"""
from __future__ import annotations
import argparse, os, sys

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to <ckpt>/actor (holds lora_adapter/)")
    ap.add_argument("--out", required=True, help="dir to write merged model")
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    adapter_dir = os.path.join(args.ckpt, "lora_adapter")
    if not os.path.isdir(adapter_dir):
        sys.exit(f"no lora_adapter/ under {args.ckpt}")

    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM
    from transformers import AutoProcessor
    from peft import PeftModel

    print(f"[merge] loading base  {args.base}")
    model = None
    for cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            model = cls.from_pretrained(args.base, torch_dtype="auto")
            print(f"[merge] loaded via {cls.__name__}")
            break
        except ValueError:
            continue
    if model is None:
        sys.exit("failed to load base model with known auto classes")
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)

    print(f"[merge] fusing adapter {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    print(f"[merge] saving merged -> {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    processor.save_pretrained(args.out)
    print("[merge] done")


if __name__ == "__main__":
    main()
