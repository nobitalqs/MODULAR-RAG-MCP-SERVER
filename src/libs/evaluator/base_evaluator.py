"""Evaluator abstract base class and NoneEvaluator fallback.

Defines the pluggable interface for evaluation providers.
All concrete implementations (Custom, Ragas) must inherit from
``BaseEvaluator`` and implement the ``evaluate`` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """Abstract base class for Evaluator providers.

    Subclasses must implement :meth:`evaluate`. The base class provides
    :meth:`validate_query` and :meth:`validate_retrieved_chunks` for
    input validation.
    """

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[Any],
        generated_answer: str | None = None,
        ground_truth: Any = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate retrieval and generation quality.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks or records to evaluate.
            generated_answer: Optional generated answer text.
            ground_truth: Optional ground truth data (ids or answers).
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters.

        Returns:
            Dictionary of metric names to float values.
        """

    def validate_query(self, query: str) -> None:
        """Validate the query string.

        Raises:
            ValueError: If query is not a non-empty string.
        """
        if not isinstance(query, str):
            raise ValueError(
                f"Query must be a string, got {type(query).__name__}"
            )
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def validate_retrieved_chunks(self, retrieved_chunks: list[Any]) -> None:
        """Validate retrieved chunks structure.

        Raises:
            ValueError: If retrieved_chunks is not a non-empty list.
        """
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list")
        if not retrieved_chunks:
            raise ValueError("retrieved_chunks cannot be empty")


class NoneEvaluator(BaseEvaluator):
    """No-op evaluator that returns empty metrics.

    Used when evaluation is disabled or the provider is set to ``none``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[Any],
        generated_answer: str | None = None,
        ground_truth: Any = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Return empty metrics dict after validation."""
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        return {}
