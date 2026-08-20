#!/usr/bin/env bash
# ============================================================
# Fetch the OFFICIAL eval pipelines for the 6 benchmarks.
# We do NOT hand-write metrics — we vendor the official repos
# (shallow, pinned) and call their scorers through thin adapters.
#
# HF dataset provenance (source of truth):
#   V* Bench          craigwu/vstar_bench
#   TreeBench         HaochenWang/TreeBench
#   VisualProbe_Hard  Mini-o3/datasets (VisualProbe_Hard)
#   HR-Bench 4K       DreamMr/HR-Bench
#   MME-RealWorld-Lite yifanzhang114/MME-RealWorld-Lite
#   O3-Bench          m-Just/O3-Bench
#
# Pin commits after first clone: replace HEAD with the SHA in
# each repo so the eval stays byte-identical for the teacher.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

clone() {  # name  url
  local name="$1" url="$2"
  if [ -d "$name/.git" ]; then
    echo "[skip] $name already present"
  else
    echo "[clone] $name <- $url"
    git clone --depth 1 "$url" "$name" || echo "  !! FAILED: $url (resolve manually)"
  fi
}

# source -> official eval repo (metric / answer-extraction lives here)
clone mme_MME-RealWorld   https://github.com/yfzhang114/MME-RealWorld
clone hr_HR-Bench         https://github.com/DreamMr/HR-Bench
clone vstar_vstar         https://github.com/penghao-wu/vstar
clone tree_TreeVGR        https://github.com/Haochen-Wang409/TreeVGR
clone visualprobe_Mini-o3 https://github.com/Mini-o3/Mini-o3
# O3-Bench: no standalone GitHub under m-Just; eval口径 in HF card —
# resolve into o3_O3-Bench/ manually (see eval/vendor/README.md).

echo
echo "Done. Next: pin each repo to a commit SHA and record it in eval/vendor/PINS.txt"
echo "Then locate each repo's answer-extraction + accuracy scorer (see README.md)."
