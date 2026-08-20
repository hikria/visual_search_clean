#!/usr/bin/env bash
# Baseline 2: zero-shot REASONING (CoT) then answer.
# NOTE: exact definition of baseline 2 vs baseline 3 is pending teacher
# confirmation — both were described as "zero-shot reasoning". This runs the
# CoT variant of the base model; adjust once the distinction is confirmed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# IMAGE_ROOT = dir holding benchmark images (not distributed). Default $ROOT/data.
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT/data}"

for split in originalQA recoverQA; do
  case $split in
    originalQA) EV="$ROOT/data/splits/eval_originalQA_652.jsonl";;
    recoverQA) EV="$ROOT/data/splits/eval_recoverQA_248.jsonl";;
  esac
  python "$ROOT/baselines/common/run_baseline.py" \
    --config "$ROOT/configs/model.yaml" \
    --eval "$EV" --image-root "$IMAGE_ROOT" \
    --mode reasoning \
    --out "$ROOT/results/b2_zeroshot_reasoning/${split}_predictions.jsonl"
done
