"""Reranker abstract base class and NoneReranker fallback.

Defines the pluggable interface for reranker providers.
All concrete reranker implementations (Cross-Encoder, LLM-based)
must inherit from ``BaseReranker`` and implement the ``rerank`` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """Abstract base class for Reranker providers.

    Subclasses must implement :meth:`rerank`. The base class provides
    :meth:`validate_query` and :meth:`validate_candidates` for input
    validation.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Rerank candidate chunks for a given query.

        Args:
            query: The user query string.
            candidates: List of candidate records to rerank. Each item
                is a dict containing at least an identifier and any
                fields needed by the reranker implementation.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters (top_k, timeout, etc.).

        Returns:
            A list of candidates in reranked order. Implementations
            should preserve candidate dicts and only change ordering.
        """

    def validate_query(self, query: str) -> None:
        """Validate the query string.

        Raises:
            ValueError: If query is not a non-empty string.
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def validate_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> None:
        """Validate candidate list structure.

        Raises:
            ValueError: If candidates is not a non-empty list of dicts.
        """
        if not isinstance(candidates, list):
            raise ValueError("Candidates must be a list of dicts")
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError(
                    f"Candidate at index {i} is not a dict (type: {type(candidate).__name__})"
                )


class NoneReranker(BaseReranker):
    """No-op reranker that preserves original order.

    Used when reranking is disabled or the provider is set to ``none``.
    Validates inputs and returns candidates unchanged.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return candidates in original order.

        Returns:
            A shallow copy of candidates preserving order.
        """
        self.validate_query(query)
        self.validate_candidates(candidates)
        return list(candidates)
