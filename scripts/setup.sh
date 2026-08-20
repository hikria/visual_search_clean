#!/usr/bin/env bash
# One-shot environment setup: vendor official eval repos + clone EasyR1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "== 1/2 fetch official eval pipelines =="
bash "$ROOT/eval/vendor/fetch_official.sh"
echo "== 2/2 clone EasyR1 (veRL-based GRPO) =="
bash "$ROOT/scripts/setup_training.sh"
echo "== data splits =="
echo "  original images/jsonl are not distributed with this repo."
echo "  reproduce the seed=258 split from data/splits_meta/split_manifest.json,"
echo "  then place / symlink it as $ROOT/data/splits (see data/splits_meta/README_切分说明.md)."
echo "Setup done."
