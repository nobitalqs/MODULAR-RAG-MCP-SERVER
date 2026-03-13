"""Benchmark Hybrid Search Weight Combinations.

Traverse a dense_weight × sparse_weight grid using pre-computed retrieval
results. Only the lightweight RRF fusion is re-run per combination —
embedding and BM25 retrieval happen once (O(N_queries) not O(N_combos)).

Usage:
    python scripts/benchmark_weights.py [--top-k 5] [--collection golden]
    python scripts/benchmark_weights.py --output json > results.json
    python scripts/benchmark_weights.py --output csv  > results.csv
    python scripts/benchmark_weights.py --with-rerank   # also sweep with reranker
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.query_engine.fusion import RRFFusion  # noqa: E402
from src.core.settings import load_settings  # noqa: E402
from src.core.types import RetrievalResult  # noqa: E402
from src.mcp_server.tools.query_knowledge_hub import (  # noqa: E402
    QueryKnowledgeHubConfig,
    QueryKnowledgeHubTool,
)

logger = logging.getLogger(__name__)

GOLDEN_SET = _REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"

# Weight grid: 0.0 → 2.0 with step 0.2
WEIGHT_VALUES = [round(v * 0.2, 1) for v in range(11)]  # [0.0, 0.2, ..., 2.0]


# ── Metrics ──────────────────────────────────────────────────────────


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Fraction of expected chunks found in retrieved set."""
    if not expected_ids:
        return 0.0
    return len(set(retrieved_ids) & set(expected_ids)) / len(expected_ids)


def mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Mean Reciprocal Rank: 1/(rank of first relevant result)."""
    expected_set = set(expected_ids)
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in expected_set:
            return 1.0 / i
    return 0.0


# ── Data Structures ──────────────────────────────────────────────────


@dataclass(frozen=True)
class WeightConfig:
    dense_weight: float
    sparse_weight: float

    @property
    def label(self) -> str:
        return f"d={self.dense_weight:.1f} s={self.sparse_weight:.1f}"


@dataclass(frozen=True)
class CategoryMetrics:
    category: str
    n: int
    recall_at_5: float
    mrr: float


@dataclass(frozen=True)
class BenchmarkResult:
    config: WeightConfig
    recall_at_5: float
    mrr: float
    avg_ms: float
    per_category: tuple[CategoryMetrics, ...]


@dataclass
class PrecomputedQuery:
    """Pre-computed retrieval results for a single query."""
    query: str
    category: str
    expected_chunk_ids: list[str]
    dense_results: list[RetrievalResult]
    sparse_results: list[RetrievalResult]


# ── Pre-compute Retrieval ────────────────────────────────────────────


async def precompute_retrievals(
    test_cases: list[dict],
    tool: QueryKnowledgeHubTool,
    collection: str,
) -> list[PrecomputedQuery]:
    """Run dense + sparse retrieval once per query, cache results.

    The HybridSearch.search(return_details=True) gives us the raw
    dense_results and sparse_results before fusion, which we can
    re-fuse with different weights without touching embedding/BM25 again.
    """
    precomputed: list[PrecomputedQuery] = []

    for i, tc in enumerate(test_cases, 1):
        print(
            f"  Pre-computing [{i:2d}/{len(test_cases)}] {tc['query'][:50]}...",
            file=sys.stderr,
            flush=True,
        )

        # Use HybridSearch directly for raw results.
        # No collection filter needed — the tool already targets the right
        # ChromaDB collection and BM25 index via _ensure_initialized().
        hs = tool._hybrid_search
        result = hs.search(
            query=tc["query"],
            top_k=20,  # Get more candidates for fusion to re-rank
            return_details=True,
        )

        precomputed.append(PrecomputedQuery(
            query=tc["query"],
            category=tc["category"],
            expected_chunk_ids=tc["expected_chunk_ids"],
            dense_results=result.dense_results or [],
            sparse_results=result.sparse_results or [],
        ))

    return precomputed


# ── Fusion-only Evaluation ───────────────────────────────────────────


def evaluate_config_fast(
    config: WeightConfig,
    precomputed: list[PrecomputedQuery],
    rrf_k: int,
    top_k: int,
) -> BenchmarkResult:
    """Evaluate a weight config using only RRF fusion (no retrieval)."""
    fusion = RRFFusion(k=rrf_k)

    cat_metrics: dict[str, list[dict[str, float]]] = {}
    t0 = time.monotonic()

    for pq in precomputed:
        ranking_lists: list[list[RetrievalResult]] = []
        weights: list[float] = []

        if pq.dense_results:
            ranking_lists.append(pq.dense_results)
            weights.append(config.dense_weight)
        if pq.sparse_results:
            ranking_lists.append(pq.sparse_results)
            weights.append(config.sparse_weight)

        if not ranking_lists:
            fused = []
        elif len(ranking_lists) == 1:
            fused = ranking_lists[0][:top_k]
        else:
            fused = fusion.fuse_with_weights(
                ranking_lists=ranking_lists,
                weights=weights,
                top_k=top_k,
            )

        retrieved = [r.chunk_id for r in fused]
        m = {
            "recall": recall_at_k(retrieved, pq.expected_chunk_ids),
            "mrr": mrr(retrieved, pq.expected_chunk_ids),
        }
        cat_metrics.setdefault(pq.category, []).append(m)

    elapsed = (time.monotonic() - t0) * 1000
    all_m = [m for mlist in cat_metrics.values() for m in mlist]
    n_total = len(all_m)

    per_cat = tuple(
        CategoryMetrics(
            category=cat,
            n=len(mlist),
            recall_at_5=sum(m["recall"] for m in mlist) / len(mlist),
            mrr=sum(m["mrr"] for m in mlist) / len(mlist),
        )
        for cat, mlist in sorted(cat_metrics.items())
    )

    return BenchmarkResult(
        config=config,
        recall_at_5=sum(m["recall"] for m in all_m) / n_total,
        mrr=sum(m["mrr"] for m in all_m) / n_total,
        avg_ms=elapsed / n_total,
        per_category=per_cat,
    )


# ── Full Pipeline Evaluation (with reranker) ────────────────────────


async def evaluate_config_full(
    config: WeightConfig,
    test_cases: list[dict],
    tool: QueryKnowledgeHubTool,
    top_k: int,
    collection: str,
) -> BenchmarkResult:
    """Evaluate using the full pipeline (retrieval + fusion + rerank)."""
    tool._hybrid_search.config.dense_weight = config.dense_weight
    tool._hybrid_search.config.sparse_weight = config.sparse_weight

    cat_metrics: dict[str, list[dict[str, float]]] = {}
    total_time = 0.0

    for tc in test_cases:
        t0 = time.monotonic()
        response = await tool.execute(
            query=tc["query"], top_k=top_k, collection=collection,
        )
        elapsed = (time.monotonic() - t0) * 1000

        retrieved = [c.chunk_id for c in response.citations]
        m = {
            "recall": recall_at_k(retrieved, tc["expected_chunk_ids"]),
            "mrr": mrr(retrieved, tc["expected_chunk_ids"]),
        }
        cat_metrics.setdefault(tc["category"], []).append(m)
        total_time += elapsed

    all_m = [m for mlist in cat_metrics.values() for m in mlist]
    n_total = len(all_m)

    per_cat = tuple(
        CategoryMetrics(
            category=cat,
            n=len(mlist),
            recall_at_5=sum(m["recall"] for m in mlist) / len(mlist),
            mrr=sum(m["mrr"] for m in mlist) / len(mlist),
        )
        for cat, mlist in sorted(cat_metrics.items())
    )

    return BenchmarkResult(
        config=config,
        recall_at_5=sum(m["recall"] for m in all_m) / n_total,
        mrr=sum(m["mrr"] for m in all_m) / n_total,
        avg_ms=total_time / n_total,
        per_category=per_cat,
    )


# ── Benchmark Runner ─────────────────────────────────────────────────


async def run_benchmark(
    top_k: int = 5,
    collection: str = "golden",
    with_rerank: bool = False,
) -> list[BenchmarkResult]:
    """Run golden test set under all weight combinations.

    Default mode (fast): pre-compute dense+sparse once, re-run fusion only.
    --with-rerank mode: full pipeline per config (slower but tests reranker interaction).
    """
    with open(GOLDEN_SET, encoding="utf-8") as f:
        golden = json.load(f)

    test_cases = golden["test_cases"]
    base_settings = load_settings()

    # Build weight grid, excluding (0.0, 0.0)
    configs = [
        WeightConfig(dense_weight=dw, sparse_weight=sw)
        for dw in WEIGHT_VALUES
        for sw in WEIGHT_VALUES
        if not (dw == 0.0 and sw == 0.0)
    ]

    total = len(configs)
    mode = "full pipeline (with rerank)" if with_rerank else "fusion-only (fast)"
    print(f"\n{'=' * 80}", file=sys.stderr)
    print(
        f"Weight Benchmark — {len(test_cases)} queries × {total} combos, "
        f"top_k={top_k}, mode={mode}",
        file=sys.stderr,
    )
    print(f"{'=' * 80}\n", file=sys.stderr)

    # Initialize tool and warm up
    print("  Initializing tool...", file=sys.stderr, flush=True)
    tool = QueryKnowledgeHubTool(
        settings=base_settings,
        config=QueryKnowledgeHubConfig(enable_rerank=with_rerank),
    )
    await tool.execute(query="warmup", top_k=1, collection=collection)
    print("  Ready.\n", file=sys.stderr)

    if with_rerank:
        # Full pipeline: re-run everything per config
        results: list[BenchmarkResult] = []
        for i, cfg in enumerate(configs, 1):
            print(
                f"  [{i:3d}/{total}] {cfg.label} ...",
                end="", file=sys.stderr, flush=True,
            )
            result = await evaluate_config_full(
                cfg, test_cases, tool, top_k, collection,
            )
            print(
                f"  Recall@5={result.recall_at_5:.1%}  MRR={result.mrr:.3f}  "
                f"({result.avg_ms:.0f}ms/q)",
                file=sys.stderr,
            )
            results.append(result)
        return results

    # Fast mode: pre-compute retrieval, sweep fusion weights
    print("  Pre-computing retrieval results...\n", file=sys.stderr)
    precomputed = await precompute_retrievals(test_cases, tool, collection)

    rrf_k = base_settings.retrieval.rrf_k
    print(f"\n  Sweeping {total} weight combos (fusion-only, rrf_k={rrf_k})...\n",
          file=sys.stderr)

    results = []
    for i, cfg in enumerate(configs, 1):
        result = evaluate_config_fast(cfg, precomputed, rrf_k, top_k)
        if i <= 5 or i % 20 == 0 or i == total:
            print(
                f"  [{i:3d}/{total}] {cfg.label}  "
                f"Recall@5={result.recall_at_5:.1%}  MRR={result.mrr:.3f}",
                file=sys.stderr,
            )
        results.append(result)

    print(f"\n  Done. ({total} combos evaluated)", file=sys.stderr)
    return results


# ── Output Formatters ────────────────────────────────────────────────


def find_baseline(results: list[BenchmarkResult]) -> BenchmarkResult | None:
    """Find the baseline (1.0, 1.0) result."""
    for r in results:
        if r.config.dense_weight == 1.0 and r.config.sparse_weight == 1.0:
            return r
    return None


def format_table(results: list[BenchmarkResult]) -> str:
    """Human-readable markdown table sorted by Recall@5 desc, MRR desc."""
    baseline = find_baseline(results)

    ranked = sorted(results, key=lambda r: (r.recall_at_5, r.mrr), reverse=True)

    lines: list[str] = []
    lines.append(f"\n{'=' * 90}")
    lines.append("WEIGHT BENCHMARK RESULTS (sorted by Recall@5, then MRR)")
    lines.append(f"{'=' * 90}")
    lines.append("")

    if baseline:
        lines.append(
            f"Baseline (1.0:1.0) — Recall@5={baseline.recall_at_5:.1%}  "
            f"MRR={baseline.mrr:.3f}"
        )
        lines.append("")

    header = (
        f"{'Rank':>4}  {'Dense':>6}  {'Sparse':>7}  "
        f"{'Recall@5':>9}  {'MRR':>8}  {'Avg ms':>8}  {'vs Baseline':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for rank, r in enumerate(ranked, 1):
        delta = ""
        if baseline:
            dr = r.recall_at_5 - baseline.recall_at_5
            dm = r.mrr - baseline.mrr
            if dr > 0 or dm > 0:
                delta = f"+R{dr:+.1%} M{dm:+.3f}"
            elif dr < 0 or dm < 0:
                delta = f" R{dr:+.1%} M{dm:+.3f}"
            else:
                delta = "= baseline"

        marker = " ***" if (
            baseline
            and r.recall_at_5 >= baseline.recall_at_5
            and r.mrr >= baseline.mrr
            and r is not baseline
        ) else ""

        lines.append(
            f"{rank:>4}  {r.config.dense_weight:>6.1f}  {r.config.sparse_weight:>7.1f}  "
            f"{r.recall_at_5:>9.1%}  {r.mrr:>8.3f}  {r.avg_ms:>8.0f}  "
            f"{delta:>12}{marker}"
        )

    # Per-category breakdown for top 5
    lines.append("")
    lines.append(f"{'─' * 90}")
    lines.append("TOP 5 — Per-category breakdown")
    lines.append(f"{'─' * 90}")
    lines.append(
        f"{'Config':<20}  {'Category':<25}  {'N':>3}  {'Recall@5':>9}  {'MRR':>8}"
    )
    lines.append("-" * 75)

    for r in ranked[:5]:
        for cm in r.per_category:
            lines.append(
                f"{r.config.label:<20}  {cm.category:<25}  {cm.n:>3}  "
                f"{cm.recall_at_5:>9.1%}  {cm.mrr:>8.3f}"
            )
        lines.append("")

    # Summary: configs beating baseline
    if baseline:
        better = [
            r for r in ranked
            if r.recall_at_5 >= baseline.recall_at_5
            and r.mrr >= baseline.mrr
            and r is not baseline
        ]
        lines.append(f"Configs matching or beating baseline on BOTH metrics: {len(better)}")
        if better:
            best = better[0]
            lines.append(
                f"RECOMMENDED: dense_weight={best.config.dense_weight:.1f}  "
                f"sparse_weight={best.config.sparse_weight:.1f}  "
                f"(Recall@5={best.recall_at_5:.1%}  MRR={best.mrr:.3f})"
            )

    return "\n".join(lines)


def format_csv(results: list[BenchmarkResult]) -> str:
    """CSV output for spreadsheet analysis."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "dense_weight", "sparse_weight", "recall_at_5", "mrr", "avg_ms",
    ])
    for r in sorted(results, key=lambda r: (r.config.dense_weight, r.config.sparse_weight)):
        writer.writerow([
            f"{r.config.dense_weight:.1f}",
            f"{r.config.sparse_weight:.1f}",
            f"{r.recall_at_5:.4f}",
            f"{r.mrr:.4f}",
            f"{r.avg_ms:.1f}",
        ])
    return buf.getvalue()


def format_json(results: list[BenchmarkResult]) -> str:
    """JSON output for programmatic processing."""
    baseline = find_baseline(results)
    ranked = sorted(results, key=lambda r: (r.recall_at_5, r.mrr), reverse=True)

    data = {
        "baseline": {
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
            "recall_at_5": baseline.recall_at_5 if baseline else None,
            "mrr": baseline.mrr if baseline else None,
        },
        "results": [
            {
                "rank": i,
                "dense_weight": r.config.dense_weight,
                "sparse_weight": r.config.sparse_weight,
                "recall_at_5": round(r.recall_at_5, 4),
                "mrr": round(r.mrr, 4),
                "avg_ms": round(r.avg_ms, 1),
                "per_category": {
                    cm.category: {
                        "n": cm.n,
                        "recall_at_5": round(cm.recall_at_5, 4),
                        "mrr": round(cm.mrr, 4),
                    }
                    for cm in r.per_category
                },
            }
            for i, r in enumerate(ranked, 1)
        ],
    }

    if baseline:
        better = [
            r for r in ranked
            if r.recall_at_5 >= baseline.recall_at_5
            and r.mrr >= baseline.mrr
            and r is not baseline
        ]
        if better:
            best = better[0]
            data["recommended"] = {
                "dense_weight": best.config.dense_weight,
                "sparse_weight": best.config.sparse_weight,
                "recall_at_5": round(best.recall_at_5, 4),
                "mrr": round(best.mrr, 4),
            }

    return json.dumps(data, indent=2, ensure_ascii=False)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", default="golden")
    parser.add_argument(
        "--output", choices=["table", "csv", "json"], default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--golden-set", type=Path, default=None,
        help="Path to golden test set JSON (default: tests/fixtures/golden_test_set.json)",
    )
    parser.add_argument(
        "--with-rerank", action="store_true",
        help="Run full pipeline including reranker (slower, tests reranker interaction)",
    )
    args = parser.parse_args()

    if args.golden_set:
        global GOLDEN_SET
        GOLDEN_SET = args.golden_set

    # Suppress noisy logs when capturing structured output
    if args.output != "table":
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    results = asyncio.run(run_benchmark(
        top_k=args.top_k, collection=args.collection, with_rerank=args.with_rerank,
    ))

    match args.output:
        case "table":
            print(format_table(results))
        case "csv":
            print(format_csv(results))
        case "json":
            print(format_json(results))


if __name__ == "__main__":
    main()
