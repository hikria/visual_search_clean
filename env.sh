#!/usr/bin/env bash
# Shared environment for vs_baselines. Source this from any run script:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# All values are overridable from the outer environment, so the repo runs on any
# machine without editing scripts. Point the *_PY vars at the right interpreters
# (a vLLM env for eval, a peft==0.20.0 env for LoRA merge) if they are not on PATH.

# Repo root = the dir this file lives in (self-locating, no hardcoded path).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO

# HF cache: default to the user's home cache; override with HF_HOME.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Python interpreters (override per machine):
#   VS_PY        -> vLLM eval / inference stack (vllm 0.15.x + the vllm_compat shim)
#   VS_MERGE_PY  -> LoRA merge stack (peft==0.20.0, matches the EasyR1 adapter)
PY="${VS_PY:-python3}"
MERGE_ENV="${VS_MERGE_PY:-python3}"
EVAL_ENV="${VS_PY:-python3}"
export PY MERGE_ENV EVAL_ENV

# vLLM / allocator settings common to every run.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Qwen-VL compat shim (monkeypatches vLLM 0.15.1 for Qwen2.5/3-VL); harmless for others.
export PYTHONPATH="$REPO/baselines/common/vllm_compat:${PYTHONPATH:-}"

# Optional secrets (e.g. QWEN_API_KEY for the visualprobe judge). Copy
# .env.example to .env and fill it in; this loads it if present.
if [ -f "$REPO/.env" ]; then
  set -a && . "$REPO/.env" && set +a
fi
