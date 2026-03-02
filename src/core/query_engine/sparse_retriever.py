"""Sparse Retriever for keyword-based search using BM25.

This module implements the SparseRetriever component that performs keyword-based
search using BM25 scoring.  It forms the Sparse route in the Hybrid Search Engine.

Retrieval flow:
1. Load BM25 index from disk for the target collection
2. Query BM25 with extracted keywords → [{chunk_id, score}]
3. Hydrate results via vector_store.get_by_ids → [{id, text, metadata}]
4. Merge scores with text/metadata → List[RetrievalResult]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Sparse retriever using BM25 keyword-based search.

    Design Principles Applied:
    - Pluggable: Accepts bm25_indexer and vector_store via dependency injection.
    - Config-Driven: Default top_k and collection read from settings.
    - Observable: Accepts optional TraceContext for observability integration.
    - Fail-Fast: Validates inputs early with clear error messages.
    - Type-Safe: Returns standardized RetrievalResult objects (same as DenseRetriever).

    Attributes:
        bm25_indexer: BM25 index for keyword scoring.
        vector_store: Vector store for text/metadata hydration.
        default_top_k: Default number of results to return.
        default_collection: Default collection name for index loading.

    Example:
        >>> retriever = SparseRetriever(
        ...     bm25_indexer=bm25_indexer,
        ...     vector_store=vector_store,
        ... )
        >>> results = retriever.retrieve(["Azure", "OpenAI", "配置"], top_k=5)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 10,
        default_collection: str = "default",
    ) -> None:
        """Initialize SparseRetriever with dependencies.

        Args:
            settings: Application settings. Used to extract default_top_k.
            bm25_indexer: BM25 index for keyword scoring.
            vector_store: Vector store for text/metadata hydration.
            default_top_k: Default number of results (default: 10).
            default_collection: Default collection name (default: "default").
        """
        self.bm25_indexer = bm25_indexer
        self.vector_store = vector_store
        self.default_collection = default_collection

        # Extract default_top_k from settings if available
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, "retrieval", None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, "sparse_top_k", default_top_k,
                )

        logger.info(
            "SparseRetriever initialized with default_top_k=%d, "
            "default_collection='%s'",
            self.default_top_k,
            self.default_collection,
        )

    def retrieve(
        self,
        keywords: List[str],
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Retrieve chunks matching the given keywords using BM25.

        Args:
            keywords: List of keywords (from QueryProcessor.process().keywords).
            top_k: Maximum number of results. Uses default_top_k if None.
            collection: Collection name for index loading. Uses default if None.
            trace: Optional TraceContext for observability (reserved for Stage F).

        Returns:
            List of RetrievalResult objects, sorted by BM25 score (descending).

        Raises:
            ValueError: If keywords list is empty or not a list.
            RuntimeError: If dependencies are missing or retrieval fails.
        """
        self._validate_keywords(keywords)
        self._validate_dependencies()

        effective_top_k = top_k if top_k is not None else self.default_top_k
        effective_collection = (
            collection if collection is not None else self.default_collection
        )

        logger.debug(
            "Retrieving for keywords=%s, top_k=%d, collection='%s'",
            keywords[:5],
            effective_top_k,
            effective_collection,
        )

        # Step 1: Ensure index is loaded
        if not self._ensure_index_loaded(effective_collection):
            logger.warning(
                "BM25 index for collection '%s' not available. "
                "Returning empty results.",
                effective_collection,
            )
            return []

        # Step 2: Query BM25 index
        try:
            bm25_results = self.bm25_indexer.query(
                query_terms=keywords,
                top_k=effective_top_k,
                trace=trace,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query BM25 index: {e}. "
                "Check index availability and query terms."
            ) from e

        # Early return if no matches
        if not bm25_results:
            logger.debug("BM25 query returned no results")
            return []

        # Step 3: Fetch text and metadata from vector store
        chunk_ids = [r["chunk_id"] for r in bm25_results]
        try:
            records = self.vector_store.get_by_ids(chunk_ids, trace=trace)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch records from vector store: {e}. "
                "Check vector store configuration and data availability."
            ) from e

        # Step 4: Merge BM25 scores with text/metadata
        results = self._merge_results(bm25_results, records)

        logger.debug("Retrieved %d results for keywords", len(results))
        return results

    def _validate_keywords(self, keywords: List[str]) -> None:
        """Validate the keywords list.

        Args:
            keywords: Keywords list to validate.

        Raises:
            ValueError: If keywords is not a list or is empty.
        """
        if not isinstance(keywords, list):
            raise ValueError(
                f"Keywords must be a list, got {type(keywords).__name__}"
            )
        if not keywords:
            raise ValueError("Keywords list cannot be empty")

    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are configured.

        Raises:
            RuntimeError: If bm25_indexer or vector_store is None.
        """
        if self.bm25_indexer is None:
            raise RuntimeError(
                "SparseRetriever requires a bm25_indexer. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "SparseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )

    def _ensure_index_loaded(self, collection: str) -> bool:
        """Ensure the BM25 index is loaded for the given collection.

        Always reloads from disk because the index may have been updated
        by another process (e.g., dashboard ingestion).  The load is
        fast (a single JSON file read) compared to the overall query.

        Args:
            collection: Collection name to load.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            return self.bm25_indexer.load(collection=collection)
        except Exception as e:
            logger.warning(
                "Failed to load BM25 index for collection '%s': %s",
                collection,
                e,
            )
            return False

    def _merge_results(
        self,
        bm25_results: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """Merge BM25 scores with text and metadata from vector store.

        Args:
            bm25_results: BM25 results [{chunk_id, score}, ...].
            records: Vector store records [{id, text, metadata}, ...].

        Returns:
            List of RetrievalResult objects. Missing records are skipped.
        """
        results = []

        for bm25_result, record in zip(bm25_results, records):
            chunk_id = bm25_result["chunk_id"]
            score = bm25_result["score"]

            # Handle case where record was not found
            if not record:
                logger.warning(
                    "No record found in vector store for chunk_id='%s'. "
                    "Skipping this result.",
                    chunk_id,
                )
                continue

            text = record.get("text", "")
            metadata = record.get("metadata", {})

            try:
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=str(text),
                    metadata=metadata,
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to create RetrievalResult for chunk_id='%s': %s. "
                    "Skipping.",
                    chunk_id,
                    e,
                )
                continue

        return results


def create_sparse_retriever(
    settings: Optional[Settings] = None,
    bm25_indexer: Optional[BM25Indexer] = None,
    vector_store: Optional[BaseVectorStore] = None,
) -> SparseRetriever:
    """Factory function to create a SparseRetriever with dependency injection.

    Args:
        settings: Application settings (used for default_top_k extraction).
        bm25_indexer: Pre-configured BM25 indexer.
        vector_store: Pre-configured vector store.

    Returns:
        Configured SparseRetriever instance.
    """
    return SparseRetriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
