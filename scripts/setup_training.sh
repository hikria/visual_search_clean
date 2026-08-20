#!/usr/bin/env bash
# Clone EasyR1 (veRL-based) at the pinned commit and apply our local patches.
# Used by baselines B3 & B4 (GRPO cold-start, sparse reward, NO injected reasoning:
# we only swap the dataset/reward script — the base training loop stays official).
#
# Reproducibility contract:
#   - EASYR1_PIN.txt           : the exact upstream commit we build on.
#   - training/patches/*.patch : our minimal edits on top of that commit
#       (RTX-3090 P2P/NCCL + Ray temp/object-store caps + local-driver debug in
#        verl/trainer/main.py; SDPA attention fallback in verl/workers/fsdp_workers.py).
# EasyR1 itself is gitignored (training/EasyR1/); this script reconstructs it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/training/EasyR1"
PIN="$(tr -d '[:space:]' < "$ROOT/training/EASYR1_PIN.txt")"
PATCH_DIR="$ROOT/training/patches"

if [ ! -d "$DEST/.git" ]; then
  # Full clone (not --depth 1): the pinned commit may be behind the branch tip.
  git clone https://github.com/hiyouga/EasyR1 "$DEST"
fi

# Pin to the exact commit our patches were authored against.
git -C "$DEST" fetch --quiet origin "$PIN" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$PIN"
echo "[ok] EasyR1 checked out at $PIN"

# Apply each patch idempotently: skip if already applied (reverse-check passes).
shopt -s nullglob
for p in "$PATCH_DIR"/*.patch; do
  if git -C "$DEST" apply --reverse --check "$p" 2>/dev/null; then
    echo "[skip] already applied: $(basename "$p")"
  elif git -C "$DEST" apply --check "$p" 2>/dev/null; then
    git -C "$DEST" apply "$p"
    echo "[ok] applied: $(basename "$p")"
  else
    echo "[FAIL] $(basename "$p") does not apply on $PIN — was the pin bumped?" >&2
    exit 1
  fi
done

echo "EasyR1 ready at: $DEST (pin $PIN + $(ls "$PATCH_DIR"/*.patch 2>/dev/null | wc -l) patch(es))"
