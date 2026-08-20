# 官方 eval 仓库(vendored)

这里存放 6 个 benchmark 的**官方评测代码**,由 `fetch_official.sh` 按 pinned commit 拉取。
**不进 git**(见根目录 `.gitignore`),老师 pull 后自己跑一次 `fetch_official.sh` 即可复现。

## 原则:metric 一律用官方的,我们不重写

每个 benchmark 的 adapter(在 `../adapters/`)只做一件事:把我们统一格式的模型输出
(`predictions.jsonl`)转成该官方仓库 scorer 期望的输入,然后调用官方的
**答案抽取 + 准确率计算**函数。任何 metric 逻辑都不在我们这边手写。

## source → 官方仓库 → scorer 位置(拉取后补全 SHA + 具体文件)

| source | HF 数据 | 官方 eval 仓库 | scorer 文件(拉取后确认) | pinned SHA |
|---|---|---|---|---|
| mme_realworld_lite | yifanzhang114/MME-RealWorld-Lite | yfzhang114/MME-RealWorld | `eval_*.py`(选项抽取+按类目算 acc) | TODO |
| hr_bench_4k | DreamMr/HR-Bench | DreamMr/HR-Bench | `eval/`(A/B/C/D 抽取+acc) | TODO |
| vstar_bench | craigwu/vstar_bench | penghao-wu/vstar | `vstar_bench/` eval | TODO |
| treebench | HaochenWang/TreeBench | Haochen-Wang409/TreeVGR | eval 脚本(选择+grounding) | TODO |
| visualprobe_hard | Mini-o3/datasets | Mini-o3/Mini-o3 | eval(VisualProbe_Hard) | TODO |
| o3_bench | m-Just/O3-Bench | (无独立 GitHub) | **待查 HF 卡片评测口径** | TODO |

## O3-Bench 特殊处理

`m-Just/O3-Bench` 没有独立 GitHub eval 仓库。评测口径需从 HF dataset 卡片确认,
拉取后放入 `o3_O3-Bench/` 并在此登记。在确认前,O3-Bench 的评测**先不接入**,
避免我们凭经验瞎写 metric。
