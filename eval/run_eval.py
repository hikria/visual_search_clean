"""Evaluation entrypoint.

Routes a predictions.jsonl to the OFFICIAL metric of each benchmark. We do not
implement any metric here — each adapter in eval/adapters/<source>.py wraps the
corresponding official scorer vendored under eval/vendor/ and must expose:

    def score(predictions: list[dict]) -> dict:
        '''predictions: rows from run_baseline.py filtered to this source.
        Returns {"n": int, "accuracy": float, ...official sub-metrics}.
        MUST call the official repo's answer-extraction + scoring — no
        home-grown metric logic.'''

Until an adapter is wired to its official scorer it raises NotImplementedError,
so nothing silently reports a made-up number.
"""
from __future__ import annotations
import argparse, importlib, json, collections, os, sys

sys.path.insert(0, os.path.dirname(__file__))

SOURCES = [
    "mme_realworld_lite", "hr_bench_4k", "treebench",
    "o3_bench", "vstar_bench", "visualprobe_hard",
]


def load_adapter(source: str):
    try:
        return importlib.import_module(f"adapters.{source}")
    except ModuleNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.jsonl")
    ap.add_argument("--split", default="", help="originalQA|recoverQA (label only)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pred)]
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r["source_dataset"]].append(r)

    report = {"split": args.split, "pred_file": args.pred, "per_source": {}}
    for src in SOURCES:
        preds = by_src.get(src, [])
        if not preds:
            continue
        adapter = load_adapter(src)
        if adapter is None or not hasattr(adapter, "score"):
            report["per_source"][src] = {"n": len(preds), "status": "NO_ADAPTER"}
            continue
        try:
            report["per_source"][src] = adapter.score(preds)
        except NotImplementedError as e:
            report["per_source"][src] = {"n": len(preds), "status": f"TODO: {e}"}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
