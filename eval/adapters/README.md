# 评测 adapter

每个 benchmark 一个文件 `adapters/<source>.py`,只做一件事:把 `run_baseline.py` 产出的
预测行,转成对应**官方 eval 仓库**(`../vendor/`)scorer 期望的输入,并调用官方的
**答案抽取 + 准确率**函数。

## 契约

```python
def score(predictions: list[dict]) -> dict:
    # predictions: 已过滤到本 source 的行,每行含
    #   {id, source_dataset, question, gt_answer, prediction, raw_output, mode}
    # 返回 {"n": int, "accuracy": float, ...官方子指标}
    # 必须调用 ../vendor/<official_repo> 里的官方 scorer,禁止自写 metric
    ...
```

## 状态

adapter 需在 `fetch_official.sh` 把官方仓库拉下来、确认各自 scorer 的函数签名后再实现。
未接通官方 scorer 前,`score` 抛 `NotImplementedError` → `run_eval.py` 显示 `TODO`,
**绝不静默报一个自己编的分数**。

| source | 官方 scorer 待接入的函数(拉取后确认) |
|---|---|
| mme_realworld_lite | MME-RealWorld 的选项抽取 + 分类目 acc |
| hr_bench_4k | HR-Bench 的 A/B/C/D 抽取 + acc |
| vstar_bench | vstar 的 eval(选项匹配) |
| treebench | TreeVGR 的选择/grounding scorer |
| visualprobe_hard | Mini-o3 的 VisualProbe_Hard eval |
| o3_bench | 待查 HF 卡片口径 |
