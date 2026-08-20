#!/usr/bin/env bash
# Model eval entry for a baseline that was produced from Qwen2.5-VL via LoRA GRPO
# (B3 and later B4). Handles: LoRA merge -> originalQA eval -> recoverQA eval -> score.
#
# Usage:
#   bash baselines/run_model_eval.sh <tag> <ckpt_path|merged_model_dir> [--original-only] [--recover-only]
#
#   <tag>              result tag, e.g. b3_grpo_step350 (used for output subdirs)
#   <ckpt_path>        path to <global_step_N>/actor  -> LoRA merged on the fly
#                      OR an already-merged model dir (results/evals/merged/<tag>)
#   --original-only    only run eval_originalQA_652
#   --recover-only     only run eval_recoverQA_248
#
# Outputs:
#   results/evals/<tag>/originalQA/{predictions,score}.jsonl/.json
#   results/evals/<tag>/recoverQA/  (same)
#   results/evals/<tag>/... configs and logs.
#
# Env contract (same as run_sweep.sh / B3 training):
#   - LoRA merge runs in the easyr1 conda env (peft==0.20.0 matching the adapter).
#   - Inference + scoring run in the vLLM eval env (vllm 0.15.1 + vllm_compat shim).
#   - Both envs pick their OFFLINE base weights via HF_HOME.
set -u

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # sets REPO, HF_HOME, PYTHONPATH, MERGE_ENV, EVAL_ENV, .env
BASE=Qwen/Qwen2.5-VL-7B-Instruct
MERGE_PY=$REPO/baselines/common/merge_lora.py
RUN_PY=$REPO/baselines/common/run_baseline.py
SCORE_PY=$REPO/eval/run_eval.py
# MERGE_ENV (peft==0.20.0, matches adapter) and EVAL_ENV (vllm eval stack) come from env.sh;
# override via VS_MERGE_PY / VS_PY if they are not both plain `python3`.

# GPU pair for TP=2 inference. Default: honor CUDA_VISIBLE_DEVICES, else 0,1.
GPU_PAIR="${CUDA_VISIBLE_DEVICES:-0,1}"
CUDA_VISIBLE_DEVICES_SAVED="${CUDA_VISIBLE_DEVICES:-}"

TAG=$1; shift || { echo "usage: run_model_eval.sh <tag> <ckpt|model_dir> [--original-only] [--recover-only]"; exit 1; }
SRC=$1; shift || { echo "usage: run_model_eval.sh <tag> <ckpt|model_dir>"; exit 1; }
DO_ORIGINAL=1; DO_RECOVER=1
for a in "$@"; do
  case "$a" in
    --original-only) DO_RECOVER=0 ;;
    --recover-only)  DO_ORIGINAL=0 ;;
  esac
done

OUT=$REPO/results/evals/$TAG
mkdir -p "$OUT/originalQA" "$OUT/recoverQA"

# HF_HOME / PYTHONPATH / .env (judge key for visualprobe_hard) all come from env.sh.

# ---- 1. resolve model dir (merge LoRA if SRC is a ckpt, else use directly) ---
if [ -d "$SRC/lora_adapter" ]; then
  echo "[eval] merging LoRA from $SRC"
  MODEL_DIR=$OUT/merged_model
  [ -d "$MODEL_DIR" ] && rm -rf "$MODEL_DIR"
  $MERGE_ENV $MERGE_PY --ckpt "$SRC" --out "$MODEL_DIR" --base "$BASE" \
      > "$OUT/merge.log" 2>&1 || { echo "[eval] MERGE FAILED (see $OUT/merge.log)"; exit 2; }
else
  MODEL_DIR=$SRC
  echo "[eval] using pre-merged model dir $MODEL_DIR"
fi

# ---- 2. build a temp eval config pointing at the merged model ---------------
# clone model.qwen25vl.yaml (same family/TP/image policy) but override model_id.
CFG=$OUT/eval_config.yaml
sed \
  -e "s|^\( *model_id:\).*|\1 \"$MODEL_DIR\"|" \
  $REPO/configs/model.qwen25vl.yaml > "$CFG"
echo "[eval] config -> $CFG (model_id=$MODEL_DIR)"

# ---- 3. per-split: run inference then score -------------------------------
run_split() {  # $1=split-name  $2=eval-jsonl
  local split=$1 ev=$2
  echo "[eval] === $split ==="
  CUDA_VISIBLE_DEVICES=$GPU_PAIR $EVAL_ENV $RUN_PY \
    --config "$CFG" --eval "$ev" --image-root "$REPO" --mode direct \
    --out "$OUT/$split/predictions.jsonl" > "$OUT/$split/run.log" 2>&1 \
      || { echo "[eval] $split INFERENCE FAILED (see $OUT/$split/run.log)"; return 1; }
  $EVAL_ENV $SCORE_PY --pred "$OUT/$split/predictions.jsonl" \
    --split "$split" --out "$OUT/$split/score.json" \
      > "$OUT/$split/score.log" 2>&1 \
      || { echo "[eval] $split SCORING FAILED (see $OUT/$split/score.log)"; return 1; }
  echo "[eval] $split done -> $(python3 -c "import json;d=json.load(open('$OUT/$split/score.json'));print(d.get('per_source',{}))")"
  return 0
}

if [ "$DO_ORIGINAL" -eq 1 ]; then
  run_split originalQA $REPO/data/splits/eval_originalQA_652.jsonl
fi
if [ "$DO_RECOVER" -eq 1 ]; then
  run_split recoverQA $REPO/data/splits/eval_recoverQA_248.jsonl
fi

echo "[eval] DONE -> scores under $OUT"
