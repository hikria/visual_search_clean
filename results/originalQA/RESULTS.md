# 零样本基线结果 — eval_originalQA_652

- **数据集:** `data/splits/eval_originalQA_652.jsonl`(652 条,6 个来源)
- **基座(本地开源,非 API):** Qwen3-VL-8B-Instruct、Qwen2.5-VL-7B-Instruct(family=qwen)、InternVL3-8B(family=internvl)
- **模式:** `direct` = B1(直接答,强约束只输出选项字母) / `reasoning` = B2(自由 CoT,末行 `Answer:`)
- **推理:** vLLM TP=2,greedy(temp=0),seed=258。日期 2026-08-14。
- **评测口径(官方派生):**
  - 5 个 MCQ 源用 MME-RealWorld 官方 `extract_characters_regex`(选项字母精确匹配)。
  - `visualprobe_hard` 用 Mini-o3 官方 open-ended 指标,GPT-judge(qwen-max)。
- `wt` = 按各源 n 的 micro 加权总准确率(分子=总正确数 / 分母=652)。

## 主表(accuracy)

| 模型 | 模式 | MME-RW (328) | HR-Bench (122) | TreeBench (80) | O3 (64) | V* (38) | VisualProbe (20) | **加权总** |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | direct (B1) | 0.494 | **0.828** | 0.412 | **0.312** | **0.816** | 0.050 | **0.534** |
| InternVL3-8B | direct (B1) | **0.512** | 0.730 | **0.438** | 0.281 | 0.658 | 0.050 | 0.515 |
| Qwen2.5-VL-7B | direct (B1) | 0.491 | 0.689 | 0.388 | 0.250 | 0.579 | 0.000 | 0.482 |
| InternVL3-8B | reasoning (B2) | 0.229 | 0.500 | 0.350 | 0.219 | 0.526 | **0.150** | 0.308 |
| Qwen2.5-VL-7B | reasoning (B2) | 0.220 | 0.311 | 0.287 | 0.172 | 0.342 | **0.150** | 0.245 |
| Qwen3-VL-8B | reasoning (B2) | 0.220 | 0.320 | 0.200 | 0.141 | 0.316 | 0.100 | 0.230 |

(粗体 = 该列最优)

## 关键结论

1. **direct 全面碾压自由 CoT(B1 >> B2,高 20–30 个加权点)。** 在高分辨率视觉搜索 / 细节定位类 MCQ 上,未经 RL 的 base 模型自由 CoT 会漂移、幻觉、并在长推理后丢失选项字母格式;强约束"只输出字母"反而稳。**这正是 B3/B4(dense-reward GRPO)与本方法要解决的核心 gap:让模型"想得对且想得有用",而非"想得多"。**
2. **最强 base 基线 = Qwen3-VL-8B direct(加权 0.534)**,HR-Bench 0.828、V* 0.816 尤为突出;InternVL3-8B direct 次之(0.515),在 MME-RW/TreeBench 反超。
3. **唯一反例:VisualProbe(open-ended,20 条)** reasoning > direct——open-ended 生成本就需要展开,且 judge 对含推理的完整回答更宽容;但样本量小(n=20),噪声大。
4. VisualProbe direct 近乎全 0:base 模型在 direct 模式把 open-ended 也当 MCQ 用一个词答,judge 判错——属预期。

## 复现

```bash
bash baselines/run_sweep.sh          # 6 个 run,GPU 对 (0,1)+(2,3),3 波
# 单个补跑(如 GPU0 被占导致 OOM,固定到干净卡):
CUDA_VISIBLE_DEVICES=2,3 python baselines/common/run_baseline.py \
  --config configs/model.qwen3vl.yaml --eval data/splits/eval_originalQA_652.jsonl \
  --image-root . --mode direct --out results/originalQA/qwen3vl_direct.pred.jsonl
python eval/run_eval.py --pred results/originalQA/qwen3vl_direct.pred.jsonl \
  --split originalQA --out results/originalQA/qwen3vl_direct.score.json
```

> 注:`qwen3vl_direct` 首轮在被其他进程占用的 GPU0 上 OOM 失败,补跑到干净的 GPU2,3 后正常完成(652/652)。其余 5 个 run 首轮即成功。
