"""Benchmark retrieval methods (tfidf, dense, hybrid_rrf) on dev or test split."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter"))
from baseline import hit, pack_context, validate

from retrievers import (
    DenseRetriever,
    HybridRRFRetriever,
    LexicalRetriever,
    pack_context_by_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval methods on Day 7")
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("starter/data/converted/questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--method", choices=("tfidf", "dense", "hybrid_rrf"), required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--sort-by-score", action="store_true")
    parser.add_argument("--use-filter", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [q for q in questions if q.get("split") == args.split]
    validate(chunks, selected)

    lexical = LexicalRetriever(chunks)
    dense = None
    retriever: LexicalRetriever | DenseRetriever | HybridRRFRetriever
    if args.method == "tfidf":
        retriever = lexical
    elif args.method == "dense":
        dense = DenseRetriever()
        retriever = dense
    else:
        dense = DenseRetriever()
        retriever = HybridRRFRetriever(dense, lexical)

    traces = []
    for q in selected:
        filters = q.get("filter") if args.use_filter else None
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=args.top_k)
        if args.sort_by_score:
            context, ids = pack_context_by_score(results, budget=args.budget)
        else:
            context, ids = pack_context(results, budget=args.budget)
        elapsed_ms = (time.perf_counter() - start) * 1000

        top3 = results[:3]
        marked = [{**c, "hit": hit(c, q["gold"])} for c in top3]
        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "strategy": args.method,
                "top_k": args.top_k,
                "budget_chars": args.budget,
                "gold": q["gold"],
                "results": marked,
                "context": context,
                "context_ids": ids,
                "context_chars": len(context),
                "elapsed_ms": elapsed_ms,
                "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(c["hit"] for c in marked),
                "context_hit": any(hit(c, q["gold"]) and c["id"] in ids for c in results),
            }
        )

    embed_tokens = dense.total_tokens if dense else 0
    # text-embedding-3-small: $0.02 / 1M tokens
    service_cost_usd = round(embed_tokens * 0.02 / 1_000_000, 8)
    summary = {
        "strategy": args.method,
        "split": args.split,
        "top_k": args.top_k,
        "budget_chars": args.budget,
        "sort_by_score": args.sort_by_score,
        "use_filter": args.use_filter,
        "n": len(traces),
        "hit1": sum(t["hit1"] for t in traces),
        "hit3": sum(t["hit3"] for t in traces),
        "context_hit": sum(t["context_hit"] for t in traces),
        "avg_context_chars": statistics.mean(t["context_chars"] for t in traces),
        "median_ms": statistics.median(t["elapsed_ms"] for t in traces),
        "embed_tokens": embed_tokens,
        "service_cost_usd": service_cost_usd,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

