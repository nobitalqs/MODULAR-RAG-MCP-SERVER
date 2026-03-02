#!/usr/bin/env python
"""Query script for the Modular RAG MCP Server.

Provides a command-line interface for querying the knowledge hub
using HybridSearch (Dense + Sparse + RRF) with optional reranking.

Usage:
    # Run a single query
    python scripts/query.py --query "Azure OpenAI 配置步骤" --collection technical_docs

    # Verbose mode (show dense/sparse/fusion/rerank results)
    python scripts/query.py --query "RRF 是什么" --verbose

    # Disable reranking
    python scripts/query.py --query "RRF 是什么" --no-rerank

Exit codes:
    0 - Success
    1 - Query failure
    2 - Configuration error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so ``src.*`` imports work.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.settings import load_settings  # noqa: E402
from src.core.query_engine.query_processor import QueryProcessor  # noqa: E402
from src.core.query_engine.hybrid_search import create_hybrid_search  # noqa: E402
from src.core.query_engine.dense_retriever import create_dense_retriever  # noqa: E402
from src.core.query_engine.sparse_retriever import create_sparse_retriever  # noqa: E402
from src.core.query_engine.reranker import create_core_reranker  # noqa: E402
from src.core.trace.trace_context import TraceContext  # noqa: E402
from src.ingestion.storage.bm25_indexer import BM25Indexer  # noqa: E402
from src.libs.embedding import (  # noqa: E402
    AzureEmbedding,
    EmbeddingFactory,
    OllamaEmbedding,
    OpenAIEmbedding,
)
from src.libs.vector_store import ChromaStore, VectorStoreFactory  # noqa: E402
from src.observability.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Query documents from the Modular RAG knowledge hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Query string.",
    )
    parser.add_argument(
        "--collection", "-c",
        default="default",
        help="Collection name (default: 'default').",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Max number of results (default: 10).",
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="Path to configuration file (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking even if enabled in settings.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print intermediate results (dense/sparse/fusion/rerank).",
    )

    return parser.parse_args(argv)


# ── Formatting helpers ───────────────────────────────────────────────


def _format_filters(filters: Dict[str, Any]) -> str:
    """Format a filter dict as a human-readable string."""
    if not filters:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in filters.items())


def _print_results(
    results: List[Any],
    top_k: int,
    title: str = "RESULTS",
) -> None:
    """Print a formatted result table.

    Args:
        results: List of RetrievalResult objects.
        top_k: Requested top_k (shown in header).
        title: Section title.
    """
    print(f"\n{'=' * 60}")
    print(f"{title} (top_k={top_k}, returned={len(results)})")
    print("=" * 60)

    for idx, result in enumerate(results, start=1):
        metadata = result.metadata or {}
        source_path = metadata.get("source_path", "")
        chunk_index = metadata.get("chunk_index", "")
        page_num = metadata.get("page_num", "")
        snippet = (result.text or "").replace("\n", " ")[:200]

        print(f"#{idx:02d}  score={result.score:.4f}  id={result.chunk_id}")
        print(f"     source_path={source_path}")
        if chunk_index != "":
            print(f"     chunk_index={chunk_index}")
        if page_num != "":
            print(f"     page_num={page_num}")
        print(f"     text={snippet}...")

    print("=" * 60)


# ── Component initialization ────────────────────────────────────────


def _build_components(settings: Any, collection: str):
    """Create all query-pipeline components from settings.

    Args:
        settings: Loaded Settings dataclass.
        collection: Target collection name.

    Returns:
        Tuple of (hybrid_search, reranker).
    """
    # Vector store — per-collection
    vs_factory = VectorStoreFactory()
    vs_factory.register_provider("chroma", ChromaStore)
    vector_store = vs_factory.create_from_settings(settings.vector_store)

    # Embedding client
    emb_factory = EmbeddingFactory()
    emb_factory.register_provider("openai", OpenAIEmbedding)
    emb_factory.register_provider("azure", AzureEmbedding)
    emb_factory.register_provider("ollama", OllamaEmbedding)
    embedding_client = emb_factory.create_from_settings(settings.embedding)

    # Dense retriever
    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )

    # Sparse retriever (BM25)
    bm25_indexer = BM25Indexer(index_dir="data/db/bm25")
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection

    # HybridSearch (QueryProcessor + Dense + Sparse + RRF)
    query_processor = QueryProcessor()
    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    # Reranker
    reranker = create_core_reranker(settings=settings)

    return hybrid_search, reranker


# ── Query execution ─────────────────────────────────────────────────


def _run_query(
    hybrid_search: Any,
    reranker: Any,
    query: str,
    top_k: int,
    use_rerank: bool,
    verbose: bool,
) -> int:
    """Execute a single query through the full retrieval pipeline.

    Args:
        hybrid_search: HybridSearch instance.
        reranker: CoreReranker instance.
        query: Raw query string.
        top_k: Max results to return.
        use_rerank: Whether to apply reranking.
        verbose: Whether to print intermediate results.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    trace = TraceContext(trace_type="query")
    trace.metadata["query"] = query[:200]
    trace.metadata["top_k"] = top_k

    # Step 1: Hybrid search
    try:
        hybrid_result = hybrid_search.search(
            query=query,
            top_k=top_k,
            filters=None,
            trace=trace,
            return_details=verbose,
        )
    except Exception as exc:
        print(f"[FAIL] Hybrid search failed: {exc}")
        return 1

    # Step 2: Extract results depending on verbose mode
    if verbose:
        results = hybrid_result.results
        if hybrid_result.used_fallback:
            print(
                f"[WARN] HybridSearch fallback used. "
                f"dense_error={hybrid_result.dense_error}, "
                f"sparse_error={hybrid_result.sparse_error}"
            )
        if hybrid_result.processed_query:
            print(
                f"[INFO] ProcessedQuery "
                f"keywords={hybrid_result.processed_query.keywords} "
                f"filters={_format_filters(hybrid_result.processed_query.filters)}"
            )
        _print_results(
            hybrid_result.dense_results or [], top_k=top_k, title="DENSE RESULTS",
        )
        _print_results(
            hybrid_result.sparse_results or [], top_k=top_k, title="SPARSE RESULTS",
        )
        _print_results(
            hybrid_result.results, top_k=top_k, title="FUSION RESULTS",
        )
    else:
        results = hybrid_result

    if not results:
        print("[INFO] 未找到相关文档，请先运行 ingest.py 摄取数据。")
        return 0

    # Step 3: Optional reranking
    if use_rerank and reranker.is_enabled:
        try:
            rerank_result = reranker.rerank(
                query=query, results=results, top_k=top_k, trace=trace,
            )
            results = rerank_result.results
            if verbose and rerank_result.used_fallback:
                print(
                    f"[WARN] Rerank fallback used: {rerank_result.fallback_reason} "
                    f"(reranker={rerank_result.reranker_type})"
                )
            if verbose:
                _print_results(results, top_k=top_k, title="RERANK RESULTS")
        except Exception as exc:
            print(f"[WARN] Reranking failed: {exc}. Using original order.")
    elif verbose and not reranker.is_enabled:
        print("[INFO] Reranking disabled by settings.")

    # Step 4: Final output
    _print_results(results, top_k=top_k)
    trace.finish()
    return 0


# ── Main ─────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns exit code (0/1/2).

    Args:
        argv: Argument list (defaults to sys.argv[1:]). Useful for testing.

    Returns:
        Exit code: 0=success, 1=query failure, 2=config error.
    """
    args = parse_args(argv)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[FAIL] Configuration file not found: {config_path}")
        return 2

    try:
        settings = load_settings(str(config_path))
        print(f"[OK] Configuration loaded from: {config_path}")
    except Exception as exc:
        print(f"[FAIL] Failed to load configuration: {exc}")
        return 2

    print("[*] Modular RAG Query Script")
    print("=" * 60)
    print(f"Collection: {args.collection}")

    # Initialize components
    try:
        hybrid_search, reranker = _build_components(settings, args.collection)
    except Exception as exc:
        print(f"[FAIL] Failed to initialize query components: {exc}")
        logger.exception("Query initialization failed")
        return 2

    use_rerank = not args.no_rerank

    return _run_query(
        hybrid_search=hybrid_search,
        reranker=reranker,
        query=args.query,
        top_k=args.top_k,
        use_rerank=use_rerank,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
