#!/bin/bash
# Full zero-shot baseline sweep: 3 base models x 2 modes on eval_originalQA_652.
#   direct    -> B1 (base zero-shot, direct answer)
#   reasoning -> B2 (base zero-shot, CoT)
# 4x RTX 3090: run two TP=2 jobs concurrently on GPU pairs (0,1) and (2,3),
# in 3 waves. Each finished run is scored with the official-derived adapters.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # sets REPO, HF_HOME, PY, PYTHONPATH, .env
cd "$REPO"
EVAL=data/splits/eval_originalQA_652.jsonl
OUT=results/originalQA
mkdir -p $OUT
: > $OUT/sweep.progress

run() {  # $1=tag $2=config $3=mode $4=gpus
  local tag=$1 cfg=$2 mode=$3 gpus=$4
  echo "[$(date +%H:%M:%S)] START $tag ($mode) on GPU$gpus" >> $OUT/sweep.progress
  CUDA_VISIBLE_DEVICES=$gpus $PY baselines/common/run_baseline.py \
    --config $cfg --eval $EVAL --image-root . --mode $mode \
    --out $OUT/${tag}_${mode}.pred.jsonl > $OUT/${tag}_${mode}.log 2>&1
  if [ -s $OUT/${tag}_${mode}.pred.jsonl ]; then
    $PY eval/run_eval.py --pred $OUT/${tag}_${mode}.pred.jsonl \
      --split originalQA --out $OUT/${tag}_${mode}.score.json >> $OUT/${tag}_${mode}.log 2>&1
    echo "[$(date +%H:%M:%S)] DONE  $tag ($mode) -> ${tag}_${mode}.score.json" >> $OUT/sweep.progress
  else
    echo "[$(date +%H:%M:%S)] FAIL  $tag ($mode) — no predictions (see log)" >> $OUT/sweep.progress
  fi
}

# wave 1
run qwen3vl  configs/model.qwen3vl.yaml  direct 0,1 &
run qwen25vl configs/model.qwen25vl.yaml direct 2,3 &
wait
# wave 2
run internvl3 configs/model.internvl3.yaml direct 0,1 &
run qwen3vl   configs/model.qwen3vl.yaml   reasoning 2,3 &
wait
# wave 3
run qwen25vl  configs/model.qwen25vl.yaml  reasoning 0,1 &
run internvl3 configs/model.internvl3.yaml reasoning 2,3 &
wait

echo "SWEEP DONE $(date)" >> $OUT/sweep.progress
