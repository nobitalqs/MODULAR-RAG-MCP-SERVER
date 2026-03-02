"""VectorStore abstract base class and validation.

Defines the pluggable interface for vector database providers.
All concrete implementations (Chroma, Qdrant, Milvus) must inherit
from ``BaseVectorStore`` and implement ``upsert`` and ``query``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """Abstract base class for VectorStore providers.

    Subclasses must implement :meth:`upsert` and :meth:`query`.
    Optional methods (:meth:`delete`, :meth:`clear`, :meth:`get_by_ids`)
    raise ``NotImplementedError`` by default.
    """

    @abstractmethod
    def upsert(
        self,
        records: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> None:
        """Insert or update records in the vector store.

        Each record dict must contain at least ``id`` (str) and
        ``vector`` (list[float]).  ``metadata`` is optional.

        This operation should be idempotent.

        Args:
            records: Records to upsert.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters.
        """

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Query the store for similar vectors.

        Args:
            vector: Query embedding vector.
            top_k: Maximum number of results.
            filters: Optional metadata filters.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters.

        Returns:
            List of result dicts sorted by similarity (descending),
            each containing ``id``, ``score``, and ``metadata``.
        """

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_records(self, records: list[dict[str, Any]]) -> None:
        """Validate records before upsert.

        Raises:
            ValueError: If records are empty or malformed.
        """
        if not records:
            raise ValueError("Records list cannot be empty")

        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Record at index {i} is not a dict "
                    f"(type: {type(record).__name__})"
                )
            if "id" not in record:
                raise ValueError(
                    f"Record at index {i} is missing required field: 'id'"
                )
            if "vector" not in record:
                raise ValueError(
                    f"Record at index {i} is missing required field: 'vector'"
                )
            vec = record["vector"]
            if not isinstance(vec, (list, tuple)):
                raise ValueError(
                    f"Record at index {i} has invalid vector type: "
                    f"{type(vec).__name__}"
                )
            if not vec:
                raise ValueError(f"Record at index {i} has empty vector")

    def validate_query_vector(
        self,
        vector: list[float],
        top_k: int,
    ) -> None:
        """Validate query parameters.

        Raises:
            ValueError: If vector or top_k is invalid.
        """
        if not isinstance(vector, (list, tuple)):
            raise ValueError(
                f"Query vector must be a list or tuple, "
                f"got {type(vector).__name__}"
            )
        if not vector:
            raise ValueError("Query vector cannot be empty")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {top_k}"
            )

    # ------------------------------------------------------------------
    # Optional lifecycle methods
    # ------------------------------------------------------------------

    def delete(
        self,
        ids: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> None:
        """Delete records by IDs (optional).

        Raises:
            NotImplementedError: If the provider does not support deletion.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement delete()"
        )

    def clear(
        self,
        collection_name: str | None = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> None:
        """Clear all records (optional, primarily for testing).

        Raises:
            NotImplementedError: If the provider does not support clearing.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement clear()"
        )

    def get_by_ids(
        self,
        ids: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve records by IDs (required for SparseRetriever support).

        Raises:
            NotImplementedError: If the provider does not support this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_by_ids()"
        )
