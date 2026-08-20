# vs_baselines

Visual-search 项目的 **baseline + 评测 + 训练** 统一代码框架。一个仓库,多个套件(suite),共用同一套数据切分与评测入口。

## 设计原则

1. **评测不凭经验手写**:每个 benchmark 的 metric 一律 **从官方 GitHub 仓库拉取**(`eval/vendor/`),
   我们只写一层薄薄的 adapter 把模型输出喂进官方 scorer,确保和官方口径完全一致。
2. **base model 可配置**:baseline 用哪个基座模型写在 `configs/model.yaml`,不写死在代码里
   (当前待老师确认;RL 框架 EasyR1 自带 Qwen2.5-VL / Qwen3-VL)。
3. **数据来自固定切分**:训练/考卷用 seed=258 的图级切分,清单见 `data/splits_meta/`
   (`README_切分说明.md` + `split_manifest.json`)。训练集与考卷零图片重叠。**原始图片与 jsonl 不随仓库分发**,
   按 `split_manifest.json` 复现即可。
4. **可复现、可版本管理**:官方 eval 仓库和 EasyR1 都用 `fetch` 脚本按 pinned commit 拉,不进 git;
   代码/配置/adapter 进 git,老师可直接 pull。

## 目录

```
configs/                模型与评测配置(base_model 待定,写在这里)
data/                   指向 splits/ 的软链 + 说明(数据不进 git)
eval/
  vendor/               官方 eval 仓库(fetch_official.sh 拉取,pinned,不进 git)
  adapters/             每个 benchmark 一个薄 adapter:我们的记录 -> 官方 scorer 输入
  run_eval.py           评测入口:predictions.jsonl + source -> 官方 metric -> 分数
baselines/
  b1_zeroshot_answer/       Baseline 1: base model 零样本直接出 answer(无推理)
  b2_zeroshot_reasoning/    Baseline 2: base model 零样本 reasoning/CoT(待和老师确认定义)
  b3_zeroshot_reasoning_base/ Baseline 3: base model zero-shot reasoning(训练后再跑)
  b4_grpo_coldstart/        Baseline 4: base + GRPO cold-start,不用我们的推理过程(sparse reward)
training/                 EasyR1(veRL-based GRPO,setup_training.sh 克隆)
scripts/                  setup / 一键跑脚本
results/                  评测结果(predictions 不进 git,汇总分数进 git)
```

## 四个 baseline(约定)

| # | 名称 | 训练? | 说明 |
|---|---|---|---|
| 1 | zero-shot answer | 否 | 只给 instruction+image,base model 直接出 answer,和 GT 比,跑官方 metric。**现在跑** |
| 2 | zero-shot reasoning | 否 | base model 零样本带推理(CoT)再出 answer。**现在跑**(定义待确认) |
| 3 | base zero-shot reasoning | — | (与 2 的区别待老师确认) |
| 4 | base + GRPO cold-start | 是 | 常规 sparse-reward GRPO,**不注入我们标的推理过程**;train set = 我们标的 `train_2591`。索引对,过程不给。数据定稿后再训 |

> baseline 3/4 需要训练,等数据集评测定稿后再开始;目前 **不做 scaling**。

## 评测集(两份考卷)

- `eval_originalQA_652.jsonl` —— 考卷A,只有 `instruction+image+answer`,考原始 QA 能力(6 源都有)
- `eval_recoverQA_248.jsonl` —— 考卷B,考卷A 子集,带 recover 标注,考 recover 能力

## 快速开始

```bash
# 1) 环境变量:仓库自定位,无需改脚本。指定解释器(可选):
#    VS_PY       -> vLLM 推理/评测环境(vllm 0.15.x + Qwen-VL 兼容 shim)
#    VS_MERGE_PY -> LoRA merge 环境(peft==0.20.0,匹配 EasyR1 adapter)
#    未设置则默认用 PATH 上的 python3。所有路径由 env.sh 自动推导。
cp .env.example .env          # 填 QWEN_API_KEY(仅 VisualProbe 的 LLM judge 需要;可留空走离线兜底)

# 2) 拉官方 eval 仓库(pinned)+ 克隆 EasyR1 + 建数据软链:
bash scripts/setup.sh

# 3) 跑 zero-shot baseline(B1 direct / B2 reasoning,3 基座 × 2 模式一键扫):
bash baselines/run_sweep.sh
cat results/originalQA/sweep.progress     # 看进度
cat results/originalQA/RESULTS.md         # 看汇总分数
```

单个评测:`python eval/run_eval.py --pred <predictions.jsonl> --split originalQA`。
GRPO 训练后的 checkpoint 评测(LoRA merge → 两考卷官方口径打分):
`bash baselines/run_model_eval.sh <tag> <ckpt_or_merged_dir>`。

> 环境细节:每个 run 脚本开头 `source env.sh`,它自定位 REPO、设好 `HF_HOME`(默认 `~/.cache/huggingface`)、
> `PYTHONPATH`(含 Qwen-VL 兼容 shim)、并按需加载 `.env`。想放别处的 HF 缓存直接 `export HF_HOME=...` 即可。
