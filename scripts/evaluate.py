"""
Evaluation runner script.

Usage:
    python scripts/evaluate.py [--collection name] [--golden-set path] [--top-k N] [--verbose]

Loads settings from config/settings.yaml, builds HybridSearch + Evaluator,
then runs EvalRunner on the golden test set and prints aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path so ``src.*`` imports work.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.settings import load_settings  # noqa: E402
from src.libs.evaluator import EvaluatorFactory  # noqa: E402
from src.observability.evaluation.eval_runner import EvalRunner  # noqa: E402
from src.observability.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_GOLDEN_SET = "tests/fixtures/golden_test_set.json"


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run evaluation on the golden test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--golden-set", "-g",
        default=DEFAULT_GOLDEN_SET,
        help=f"Path to golden test set JSON (default: {DEFAULT_GOLDEN_SET}).",
    )
    parser.add_argument(
        "--collection", "-c",
        default=None,
        help="Collection name filter.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save JSON evaluation report.",
    )

    return parser.parse_args(argv)


# ── Main ─────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns exit code (0/1).

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        0 on success, 1 on failure.
    """
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s | %(name)s | %(message)s")

    # Load settings
    try:
        settings = load_settings()
    except Exception as exc:
        logger.error("Failed to load settings: %s", exc)
        return 1

    # Build evaluator from settings
    try:
        evaluator = EvaluatorFactory.create(settings)
        logger.info("Evaluator: %s", type(evaluator).__name__)
    except Exception as exc:
        logger.error("Failed to create evaluator: %s", exc)
        return 1

    # Build HybridSearch (optional — EvalRunner handles None gracefully)
    hybrid_search = _build_hybrid_search(settings)

    # Build LLM-based answer generator
    answer_generator = _build_answer_generator(settings)

    # Run evaluation
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        answer_generator=answer_generator,
    )

    try:
        report = runner.run(
            test_set_path=args.golden_set,
            top_k=args.top_k,
            collection=args.collection,
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1

    # Print results
    report_dict = report.to_dict()

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Evaluator : {report.evaluator_name}")
    print(f"Test set  : {report.test_set_path}")
    print(f"Queries   : {len(report.query_results)}")
    print(f"Time      : {report.total_elapsed_ms:.1f} ms")
    print("-" * 60)
    print("Aggregate Metrics:")
    for metric, value in report.aggregate_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 60)

    # Save report if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Report saved to %s", output_path)

    return 0


def _build_hybrid_search(settings: object) -> object | None:
    """Build HybridSearch from settings, returning None on failure."""
    try:
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.query_processor import QueryProcessor
        from src.libs.embedding import (
            EmbeddingFactory,
            AzureEmbedding,
            OpenAIEmbedding,
            OllamaEmbedding,
        )
        from src.libs.vector_store import VectorStoreFactory, ChromaStore
        from src.ingestion.storage.bm25_indexer import BM25Indexer

        # Vector store
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
        dense = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )

        # Sparse retriever (BM25) — index_dir must match pipeline's "data/db/bm25"
        collection = settings.vector_store.collection_name
        bm25_indexer = BM25Indexer(index_dir="data/db/bm25")
        sparse = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse.default_collection = collection

        # HybridSearch
        query_processor = QueryProcessor()
        hybrid = create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
        )
        logger.info("HybridSearch initialized successfully")
        return hybrid
    except Exception as exc:
        logger.warning("Failed to build HybridSearch (will run without retrieval): %s", exc)
        return None


def _build_answer_generator(settings: object):
    """Build an LLM-based answer generator from settings.

    Returns a callable(query, chunks) -> str, or None on failure.
    """
    try:
        from src.libs.llm import (
            LLMFactory,
            AzureLLM,
            OpenAILLM,
            OllamaLLM,
            DeepSeekLLM,
            Message,
        )

        factory = LLMFactory()
        factory.register_provider("azure", AzureLLM)
        factory.register_provider("openai", OpenAILLM)
        factory.register_provider("ollama", OllamaLLM)
        factory.register_provider("deepseek", DeepSeekLLM)
        llm = factory.create_from_settings(settings.llm)
        logger.info("LLM answer generator: %s", type(llm).__name__)

        def generate(query: str, chunks: list) -> str:
            # Extract text from chunks
            texts = []
            for c in chunks[:5]:
                if isinstance(c, str):
                    texts.append(c)
                elif isinstance(c, dict):
                    texts.append(c.get("text", str(c)))
                elif hasattr(c, "text"):
                    texts.append(str(getattr(c, "text")))
                else:
                    texts.append(str(c))

            context = "\n\n---\n\n".join(texts)

            messages = [
                Message(
                    role="system",
                    content=(
                        "You are a helpful assistant. Answer the user's question "
                        "based ONLY on the provided context. Be concise and accurate. "
                        "If the context does not contain enough information, say so."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Context:\n{context}\n\nQuestion: {query}",
                ),
            ]

            response = llm.chat(messages)
            return response.content

        return generate

    except Exception as exc:
        logger.warning("Failed to build LLM answer generator: %s", exc)
        return None


if __name__ == "__main__":
    sys.exit(main())
