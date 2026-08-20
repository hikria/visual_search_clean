#!/usr/bin/env bash
# Baseline 1: zero-shot DIRECT answer (no reasoning).
# Runs over both exams (originalQA + recoverQA), writes predictions only.
# Scoring is separate: python eval/run_eval.py ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Where image_path (e.g. data/images/...) resolves. Point IMAGE_ROOT at the dir
# that holds the benchmark images (not distributed with this repo).
# For official-faithful high-res eval (HR-Bench / V*), point it at the full-size
# official images. Default = $ROOT/data.
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT/data}"

for split in originalQA recoverQA; do
  case $split in
    originalQA) EV="$ROOT/data/splits/eval_originalQA_652.jsonl";;
    recoverQA) EV="$ROOT/data/splits/eval_recoverQA_248.jsonl";;
  esac
  python "$ROOT/baselines/common/run_baseline.py" \
    --config "$ROOT/configs/model.yaml" \
    --eval "$EV" --image-root "$IMAGE_ROOT" \
    --mode direct \
    --out "$ROOT/results/b1_zeroshot_answer/${split}_predictions.jsonl"
done
