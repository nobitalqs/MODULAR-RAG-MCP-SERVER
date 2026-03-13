"""Benchmark Confidence-Based Adaptive Retrieval.

Compares retrieval quality (Recall@K, MRR) with adaptive OFF vs ON
across multiple score_threshold values.

Usage:
    python scripts/benchmark_adaptive.py [--top-k 5] [--collection golden]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.settings import (  # noqa: E402
    AdaptiveRetrievalSettings,
    RetrievalSettings,
    load_settings,
)
from src.mcp_server.tools.query_knowledge_hub import (  # noqa: E402
    QueryKnowledgeHubConfig,
    QueryKnowledgeHubTool,
)

GOLDEN_SET = _REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"


# ── Metrics ─────────────────────────────────────────────────────────


def hit_rate(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """1.0 if any expected chunk is in retrieved set, else 0.0."""
    return 1.0 if set(retrieved_ids) & set(expected_ids) else 0.0


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


# ── Runner ──────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    label: str
    adaptive: AdaptiveRetrievalSettings | None


async def run_benchmark(
    configs: list[RunConfig],
    top_k: int = 5,
    collection: str = "golden",
) -> None:
    """Run golden test set under each config and print comparison table."""
    with open(GOLDEN_SET, encoding="utf-8") as f:
        golden = json.load(f)

    test_cases = golden["test_cases"]
    base_settings = load_settings()

    print(f"\n{'=' * 80}")
    print(f"Adaptive Retrieval Benchmark — {len(test_cases)} queries, top_k={top_k}")
    print(f"{'=' * 80}\n")

    all_results: dict[str, dict] = {}

    for cfg in configs:
        # Build settings with specific adaptive config
        new_retrieval = replace(base_settings.retrieval, adaptive=cfg.adaptive)
        settings = replace(base_settings, retrieval=new_retrieval)

        # Fresh tool for each config
        tool = QueryKnowledgeHubTool(
            settings=settings,
            config=QueryKnowledgeHubConfig(
                enable_rerank=settings.rerank.enabled,
            ),
        )

        cat_metrics: dict[str, list[dict]] = {}
        total_time = 0.0

        for tc in test_cases:
            query = tc["query"]
            expected = tc["expected_chunk_ids"]
            category = tc["category"]

            t0 = time.monotonic()
            response = await tool.execute(
                query=query, top_k=top_k, collection=collection,
            )
            elapsed = (time.monotonic() - t0) * 1000

            # Extract chunk IDs from citations
            retrieved = [c.chunk_id for c in response.citations]

            top1_score = response.citations[0].score if response.citations else 0.0
            m = {
                "hit": hit_rate(retrieved, expected),
                "recall": recall_at_k(retrieved, expected),
                "mrr": mrr(retrieved, expected),
                "elapsed_ms": elapsed,
                "top1_score": top1_score,
                "query": query[:50],
            }

            cat_metrics.setdefault(category, []).append(m)
            total_time += elapsed

        # Aggregate
        all_m = [m for mlist in cat_metrics.values() for m in mlist]
        summary = {
            "hit_rate": sum(m["hit"] for m in all_m) / len(all_m),
            "recall@k": sum(m["recall"] for m in all_m) / len(all_m),
            "mrr": sum(m["mrr"] for m in all_m) / len(all_m),
            "avg_ms": total_time / len(all_m),
            "total_ms": total_time,
            "per_category": {},
        }

        for cat, mlist in sorted(cat_metrics.items()):
            summary["per_category"][cat] = {
                "n": len(mlist),
                "hit_rate": sum(m["hit"] for m in mlist) / len(mlist),
                "recall@k": sum(m["recall"] for m in mlist) / len(mlist),
                "mrr": sum(m["mrr"] for m in mlist) / len(mlist),
            }

        all_results[cfg.label] = summary

    # ── Print comparison table ──────────────────────────────────────

    labels = [c.label for c in configs]
    print(f"{'Config':<30} {'Hit@K':>8} {'Recall@K':>10} {'MRR':>8} {'Avg ms':>10}")
    print("-" * 70)
    for label in labels:
        s = all_results[label]
        print(
            f"{label:<30} {s['hit_rate']:>8.1%} {s['recall@k']:>10.1%} "
            f"{s['mrr']:>8.3f} {s['avg_ms']:>10.0f}"
        )

    print(f"\n{'Per-category breakdown':}")
    print(f"{'Config':<30} {'Category':<25} {'Hit@K':>8} {'Recall@K':>10} {'MRR':>8}")
    print("-" * 85)
    for label in labels:
        for cat, cm in sorted(all_results[label]["per_category"].items()):
            print(
                f"{label:<30} {cat:<25} {cm['hit_rate']:>8.1%} "
                f"{cm['recall@k']:>10.1%} {cm['mrr']:>8.3f}"
            )
        print()


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", default="golden")
    args = parser.parse_args()

    configs = [
        RunConfig(
            label="baseline (adaptive OFF)",
            adaptive=AdaptiveRetrievalSettings(
                enabled=False, score_threshold=0.0, expand_factor=2, max_retries=1,
            ),
        ),
        RunConfig(
            label="adaptive threshold=0.0",
            adaptive=AdaptiveRetrievalSettings(
                enabled=True, score_threshold=0.0, expand_factor=2, max_retries=1,
            ),
        ),
        RunConfig(
            label="adaptive threshold=2.0",
            adaptive=AdaptiveRetrievalSettings(
                enabled=True, score_threshold=2.0, expand_factor=2, max_retries=1,
            ),
        ),
        RunConfig(
            label="adaptive threshold=5.0",
            adaptive=AdaptiveRetrievalSettings(
                enabled=True, score_threshold=5.0, expand_factor=2, max_retries=1,
            ),
        ),
        RunConfig(
            label="adaptive thr=2.0 expand=3",
            adaptive=AdaptiveRetrievalSettings(
                enabled=True, score_threshold=2.0, expand_factor=3, max_retries=1,
            ),
        ),
    ]

    asyncio.run(run_benchmark(configs, top_k=args.top_k, collection=args.collection))


if __name__ == "__main__":
    main()
