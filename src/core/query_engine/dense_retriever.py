"""Dense Retriever for semantic search using vector embeddings.

This module implements the DenseRetriever component that performs semantic search
by embedding the query and retrieving similar chunks from the vector store.
It forms the Dense route in the Hybrid Search Engine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.libs.embedding.base_embedding import BaseEmbedding
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense retriever using embedding-based semantic search.

    This class performs semantic retrieval by:
    1. Embedding the query using the configured embedding client
    2. Querying the vector store for similar vectors
    3. Returning normalized RetrievalResult objects

    Design Principles Applied:
    - Pluggable: Accepts embedding_client and vector_store via dependency injection.
    - Config-Driven: Default top_k read from settings.retrieval.dense_top_k.
    - Observable: Accepts optional TraceContext for observability integration.
    - Fail-Fast: Validates inputs early with clear error messages.
    - Type-Safe: Returns standardized RetrievalResult objects.

    Attributes:
        embedding_client: The embedding provider for query vectorization.
        vector_store: The vector store for similarity search.
        default_top_k: Default number of results to return.

    Example:
        >>> retriever = DenseRetriever(
        ...     embedding_client=embedding_client,
        ...     vector_store=vector_store,
        ... )
        >>> results = retriever.retrieve("What is RAG?", top_k=5)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 10,
    ) -> None:
        """Initialize DenseRetriever with dependencies.

        Args:
            settings: Application settings. Used to extract default_top_k.
            embedding_client: Embedding provider for query vectorization.
            vector_store: Vector store for similarity search.
            default_top_k: Default number of results to return (default: 10).
                           Can be overridden from settings.retrieval.dense_top_k.
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store

        # Extract default_top_k from settings if available
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, "retrieval", None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, "dense_top_k", default_top_k,
                )

        logger.info(
            "DenseRetriever initialized with default_top_k=%d", self.default_top_k,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Retrieve semantically similar chunks for a query.

        Args:
            query: The search query string. Must not be empty.
            top_k: Maximum number of results to return. Uses default_top_k if None.
            filters: Optional metadata filters (e.g., {"collection": "api-docs"}).
            trace: Optional TraceContext for observability (reserved for Stage F).

        Returns:
            List of RetrievalResult objects, sorted by similarity (descending).

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If dependencies are missing or retrieval fails.
        """
        self._validate_query(query)
        self._validate_dependencies()

        effective_top_k = top_k if top_k is not None else self.default_top_k

        logger.debug("Retrieving for query='%s...', top_k=%d", query[:50], effective_top_k)

        # Step 1: Embed the query
        try:
            query_vectors = self.embedding_client.embed([query], trace=trace)
            query_vector = query_vectors[0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to embed query: {e}. "
                "Check embedding client configuration and connectivity."
            ) from e

        # Step 2: Query the vector store
        try:
            raw_results = self.vector_store.query(
                vector=query_vector,
                top_k=effective_top_k,
                filters=filters,
                trace=trace,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query vector store: {e}. "
                "Check vector store configuration and data availability."
            ) from e

        # Step 3: Transform to RetrievalResult objects
        results = self._transform_results(raw_results)

        logger.debug("Retrieved %d results for query", len(results))
        return results

    def _validate_query(self, query: str) -> None:
        """Validate the query string.

        Args:
            query: Query string to validate.

        Raises:
            ValueError: If query is empty or not a string.
        """
        if not isinstance(query, str):
            raise ValueError(
                f"Query must be a string, got {type(query).__name__}"
            )
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are configured.

        Raises:
            RuntimeError: If embedding_client or vector_store is None.
        """
        if self.embedding_client is None:
            raise RuntimeError(
                "DenseRetriever requires an embedding_client. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "DenseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )

    def _transform_results(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """Transform raw vector store results to RetrievalResult objects.

        Args:
            raw_results: Raw results from vector store query.
                Each result should have: id, score, text, metadata.

        Returns:
            List of RetrievalResult objects. Malformed entries are skipped.
        """
        results = []
        for raw in raw_results:
            try:
                # Text may be in top-level "text" or fallback to metadata["text"]
                text = raw.get("text", "") or ""
                if not text:
                    metadata = raw.get("metadata", {})
                    text = metadata.get("text", "") if metadata else ""
                result = RetrievalResult(
                    chunk_id=str(raw.get("id", "")),
                    score=float(raw.get("score", 0.0)),
                    text=str(text),
                    metadata=raw.get("metadata", {}),
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to transform result %s: %s. Skipping.",
                    raw.get("id", "unknown"),
                    e,
                )
                continue

        return results


def create_dense_retriever(
    settings: Optional[Settings] = None,
    embedding_client: Optional[BaseEmbedding] = None,
    vector_store: Optional[BaseVectorStore] = None,
) -> DenseRetriever:
    """Factory function to create a DenseRetriever with dependency injection.

    Args:
        settings: Application settings (used for default_top_k extraction).
        embedding_client: Pre-configured embedding client.
        vector_store: Pre-configured vector store.

    Returns:
        Configured DenseRetriever instance.
    """
    return DenseRetriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
