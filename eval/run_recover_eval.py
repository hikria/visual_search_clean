"""RecoverQA-specific scoring.

Two families of scores for the recoverQA benchmark split:

1. QA accuracy (THIS STAGE): the model's plain answer correctness on recover
   samples, via the SAME official adapters as run_eval.py (5 MCQ sources use
   MME's extract_characters_regex, visualprobe uses the Mini-o3 GPT-judge).
   This is the "traditional accuracy" the recover annotations do not change.

2. RECOVER METRICS (PLACEHOLDER): detect-rate / hit-rate of the model's
   self-correction step, scored against human recover *traces*. The 3432
   labelled trace samples are still being manually annotated (ETA ~1-2 days).
   The schema shape is reserved and wired to `--recover_gold` so recovery
   metrics drop in without touching the inference/scoring skeleton.

We do NOT re-implement any metric here: QA accuracy reuses eval/adapters/*,
recover metrics will use the gold traces verbatim.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from run_eval import SOURCES, load_adapter  # noqa: E402


def qa_accuracy(pred_file: str, recover_gold: str | None) -> dict:
    rows = [json.loads(l) for l in open(pred_file)]
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source_dataset"], []).append(r)

    per_source = {}
    for src in SOURCES:
        preds = by_src.get(src, [])
        if not preds:
            continue
        adapter = load_adapter(src)
        if adapter is None or not hasattr(adapter, "score"):
            per_source[src] = {"n": len(preds), "status": "NO_ADAPTER"}
            continue
        try:
            per_source[src] = adapter.score(preds)
        except NotImplementedError as e:
            per_source[src] = {"n": len(preds), "status": f"TODO: {e}"}

    return {"n": len(rows), "per_source": per_source}


def recover_metrics(recover_gold: str, pred_file: str) -> dict:
    """Reserved slot for recover detect/hit metrics.

    Once the labelled 3432 trace set is delivered, implement:
      - recover_detect_rate: P(model's response marks it needs to re-examine)
      - recover_hit_rate:    P(model's re-answer matches the trace's gold)
    scored by joining preds to the trace annotation by id, using the
    annotation.search_trace[] recover markers (recover_to_step, recovery_pattern,
    recover_scope). Until then this is unreachable (guarded by --recover_gold).
    """
    return {
        "status": "NOT_IMPLEMENTED",
        "message": "recover metrics pending 3432-label trace set — see run_recover_eval.py",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.jsonl (recoverQA split)")
    ap.add_argument("--out", default="", help="score json path")
    ap.add_argument(
        "--recover_gold", default="", metavar="trace.jsonl",
        help="OPTIONAL: human recover-traces gold (3432, arrives ~1-2 days). "
             "When present, adds recover metrics to the report.",
    )
    args = ap.parse_args()

    report: dict[str, object] = {"qa_accuracy": qa_accuracy(args.pred, args.recover_gold or None)}
    if args.recover_gold:
        report["recover_metrics"] = recover_metrics(args.recover_gold, args.pred)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
