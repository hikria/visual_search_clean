# EasyR1 local patches

`training/EasyR1/` is **gitignored** — it is reconstructed by `scripts/setup_training.sh`,
which clones upstream EasyR1, checks out the commit pinned in `training/EASYR1_PIN.txt`,
and applies every `*.patch` in this directory (idempotently).

These are the *minimal* edits needed to run GRPO on our box; the training loop,
GRPO math, and rollout logic stay upstream-official.

## easyr1-vs.patch

Two files, ~90 lines:

- **`verl/trainer/main.py`** — single-node / consumer-GPU adaptations:
  - Ray `runtime_env` env-vars: `NCCL_P2P_DISABLE=1`, `NCCL_CUMEM_ENABLE=0`
    (RTX 3090 has no NVLink/PCIe P2P → NCCL "Cuda failure 217"),
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (incompatible with vLLM
    sleep-mode's CuMemAllocator).
  - `ray.init` caps `object_store_memory` (default 4 GB) + honors `RAY_TEMP` so
    Ray's plasma/session dirs land on the big disk, not the ~98%-full root fs.
  - `VS_DEBUG_LOCAL=1` runs the trainer body in the driver process so real
    tracebacks surface instead of being swallowed by a dead Runner actor.
- **`verl/workers/fsdp_workers.py`** — fall back to SDPA attention when
  flash-attn isn't available for the model.

## Regenerating the patch (after editing EasyR1 in place)

```bash
cd training/EasyR1
git diff > ../patches/easyr1-vs.patch
```

If you bump `EASYR1_PIN.txt`, re-verify the patch still applies:
`git -C training/EasyR1 apply --check training/patches/easyr1-vs.patch`.
