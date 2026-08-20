"""Summarize all baseline eval scores into one comparison table.

Aggregates:
  - B1/B2 originalQA scores  -> results/originalQA/*.score.json  (already existing)
  - B3/B4 originalQA + recoverQA scores -> results/evals/<tag>/... score files

Emits results/evals/RESULTS.md with:
  - a main table:  rows = model/mode/tag, cols = 6 sources accuracy + micro-weighted total
  - a recoverQA sub-table: same shape (recover split; includes recover-metric placeholders
    when a score.json has them).

Weighted total = micro-average over the union of rows present per source (names drawn
from eval/run_eval.py SOURCES), i.e. sum(correct)/sum(n) — identical to what RESULTS.md
already reports as the "wt" column.
"""
from __future__ import annotations
import argparse, json, collections, glob, os, sys

HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_eval import SOURCES  # noqa: E402

SHORT = {
    "mme_realworld_lite": "MME-RW",
    "hr_bench_4k": "HR-Bench",
    "treebench": "TreeBench",
    "o3_bench": "O3",
    "vstar_bench": "V*",
    "visualprobe_hard": "VisualProbe",
}

HEADER = ("# 基线对比 — B1/B2/B3(+B4) · originalQA + recoverQA\n\n"
          "- 数据: `data/splits/eval_originalQA_652.jsonl` / `eval_recoverQA_248.jsonl`\n"
          "- 评测口径(官方派生): 5 个 MCQ 源用 MME-RealWorld 官方 "
          "`extract_characters_regex`; `visualprobe_hard` 用 Mini-o3 官方 open-ended "
          "指标(GPT-judge)。\n"
          "- `wt` = micro 加权总 acc = Σcorrect/Σn(仅对当前行有数据的源)。\n")


def read_score(path: str) -> tuple[str | None, dict]:
    """Read a score.json. Returns (split_label, per_source dict)."""
    try:
        d = json.load(open(path))
    except Exception:
        return None, {}
    if "per_source" in d:          # run_eval.py / run_recover_eval qa_accuracy
        return d.get("split"), d["per_source"]
    if "qa_accuracy" in d:         # run_recover_eval wrapper
        qa = d["qa_accuracy"]
        return qa.get("split"), qa["per_source"]
    return None, {}


def weighted_total(per_src: dict) -> float | None:
    n = c = 0
    for k, v in per_src.items():
        if not isinstance(v, dict) or "accuracy" not in v:
            continue
        nn = v.get("n", 0)
        n += nn
        c += round(v["accuracy"] * nn)
    return (c / n) if n else None


def to_row(per_src: dict) -> list[str]:
    cells = []
    for s in SOURCES:
        v = per_src.get(s)
        acc = v.get("accuracy") if isinstance(v, dict) else None
        cells.append(f"{acc:.3f}" if acc is not None else "—")
    wt = weighted_total(per_src)
    cells.append(f"**{wt:.3f}**" if wt is not None else "**—**")
    return cells


def collect(glob_patterns: list[str]) -> list[tuple[str, str, dict]]:
    """Return [(label, split, per_src), ...] labelled by filename stem."""
    rows: dict[str, dict] = {}
    files = []
    for p in glob_patterns:
        files += glob.glob(p)
    for f in sorted(files):
        split, per_src = read_score(f)
        stem = os.path.basename(f).replace(".score.json", "").replace(".json", "")
        # stem like "b3_grpo_step350" (from results/evals/<tag>/{originalQA,recoverQA}/score.json)
        label = os.path.basename(os.path.dirname(os.path.dirname(f))) if "/evals/" in f else stem
        rows[(label, split or "originalQA")] = per_src
    return [(l, s, ps) for (l, s), ps in rows.items()]


def table(rows: list[tuple[str, str, dict]], title: str) -> str:
    cols = ["模型/标签", "模式"]
    for s in SOURCES:
        cols.append(SHORT[s])
    cols.append("**加权总**")
    out = [title or "", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for label, _split, per_src in rows:
        out.append("| " + " | ".join([label, mode_for(label), *to_row(per_src)]) + " |")
    out.append("")
    return "\n".join(out)


def mode_for(label: str) -> str:
    if "_reasoning" in label:
        return "reasoning"
    if "grpo" in label or label.startswith(("b3", "b4")):
        return "direct(GRPO)"
    return "direct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results/evals/RESULTS.md"))
    args = ap.parse_args()

    orig = collect([
        os.path.join(REPO, "results/originalQA/*.score.json"),
        os.path.join(REPO, "results/evals/*/originalQA/score.json"),
    ])
    recv = collect([
        os.path.join(REPO, "results/evals/*/recoverQA/score.json"),
    ])

    parts = [HEADER, "## originalQA(原始视觉 QA)", ""]
    if orig:
        parts.append(table(orig, "### 主表(accuracy)"))
    else:
        parts.append("无 originalQA score 文件")

    parts += ["## recoverQA(recover 评估)", ""]
    if recv:
        parts.append(table(recv, "### recover 子表(accuracy;recover 指标待 trace 到位)"))
    else:
        parts.append("无 recoverQA score 文件 — 训练中/未跑")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(parts) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
