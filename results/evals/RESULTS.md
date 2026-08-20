# 基线对比 — B1/B2/B3(+B4) · originalQA + recoverQA

- 数据: `data/splits/eval_originalQA_652.jsonl` / `eval_recoverQA_248.jsonl`
- 评测口径(官方派生): 5 个 MCQ 源用 MME-RealWorld 官方 `extract_characters_regex`; `visualprobe_hard` 用 Mini-o3 官方 open-ended 指标(GPT-judge)。
- `wt` = micro 加权总 acc = Σcorrect/Σn(仅对当前行有数据的源)。

## originalQA(原始视觉 QA)

### 主表(accuracy)

| 模型/标签 | 模式 | MME-RW | HR-Bench | TreeBench | O3 | V* | VisualProbe | **加权总** |
|---|---|---|---|---|---|---|---|---|
| b3_grpo_final | direct(GRPO) | 0.494 | 0.680 | 0.375 | 0.234 | 0.579 | 0.000 | **0.479** |
| internvl3_direct | direct | 0.512 | 0.730 | 0.438 | 0.281 | 0.658 | 0.050 | **0.515** |
| internvl3_reasoning | reasoning | 0.229 | 0.500 | 0.350 | 0.219 | 0.526 | 0.150 | **0.308** |
| qwen25vl_direct | direct | 0.491 | 0.689 | 0.388 | 0.250 | 0.579 | 0.000 | **0.482** |
| qwen25vl_reasoning | reasoning | 0.220 | 0.311 | 0.287 | 0.172 | 0.342 | 0.150 | **0.245** |
| qwen3vl_direct | direct | 0.494 | 0.828 | 0.412 | 0.312 | 0.816 | 0.050 | **0.534** |
| qwen3vl_reasoning | reasoning | 0.220 | 0.320 | 0.200 | 0.141 | 0.316 | 0.100 | **0.230** |

## recoverQA(recover 评估)

### recover 子表(accuracy;recover 指标待 trace 到位)

| 模型/标签 | 模式 | MME-RW | HR-Bench | TreeBench | O3 | V* | VisualProbe | **加权总** |
|---|---|---|---|---|---|---|---|---|
| b3_grpo_final | direct(GRPO) | 0.400 | 0.531 | 0.333 | 0.162 | 0.533 | 0.000 | **0.355** |

