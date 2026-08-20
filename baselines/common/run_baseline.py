"""Zero-shot baseline runner (baselines 1 & 2), model-agnostic via vLLM.

  --mode direct     -> Baseline 1: instruction+image -> answer (no reasoning)
  --mode reasoning  -> Baseline 2: instruction+image -> CoT -> answer

Writes results/<tag>/predictions.jsonl with one row per eval record:
  {id, source_dataset, question, gt_answer, prediction, raw_output, mode}
Scoring is done SEPARATELY by eval/run_eval.py using the official metric —
this script does no scoring.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.infer import load_cfg, read_eval, build_messages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model.yaml")
    ap.add_argument("--eval", required=True, help="path to eval jsonl")
    ap.add_argument("--image-root", default="")
    ap.add_argument("--mode", choices=["direct", "reasoning"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    records = read_eval(args.eval)

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    llm = LLM(
        model=cfg.model_id, dtype=cfg.dtype,
        tensor_parallel_size=cfg.tensor_parallel_size,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_seqs=cfg.max_num_seqs,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    sp = SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p,
                        max_tokens=cfg.max_new_tokens, seed=cfg.seed)

    # Vision loading is family-specific but the chat template is always the
    # model's OWN (via AutoProcessor) -> stays version/model-agnostic.
    if cfg.family == "qwen":
        from qwen_vl_utils import process_vision_info

        def load_images(msgs):
            imgs, _ = process_vision_info(msgs)
            return imgs
    else:  # internvl / generic: hand vLLM raw PIL images
        from PIL import Image

        def load_images(msgs):
            out = []
            for c in msgs[0]["content"]:
                if c.get("type") == "image":
                    out.append(Image.open(c["image"]).convert("RGB"))
            return out

    prompts = []
    for rec in records:
        msgs = build_messages(rec, args.image_root, args.mode,
                              min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        prompts.append({"prompt": text,
                        "multi_modal_data": {"image": load_images(msgs)}})

    outs = llm.generate(prompts, sp)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for rec, o in zip(records, outs):
            raw = o.outputs[0].text
            f.write(json.dumps({
                "id": rec["id"],
                "source_dataset": rec["source_dataset"],
                "question": rec["question"],
                "gt_answer": rec.get("answer"),
                "prediction": raw.strip(),
                "raw_output": raw,
                "mode": args.mode,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
