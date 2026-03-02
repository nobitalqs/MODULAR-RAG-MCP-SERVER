"""Abstract base class for query rewriters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewriteResult:
    """Immutable result of a query rewrite operation.

    Attributes:
        original_query: The input query before rewriting.
        rewritten_queries: One or more rewritten query variants.
        reasoning: Optional explanation of why the query was rewritten.
        strategy: Name of the rewrite strategy used.
    """

    original_query: str
    rewritten_queries: list[str]
    reasoning: str | None
    strategy: str


class BaseQueryRewriter(ABC):
    """Pluggable interface for query rewriting.

    Subclasses must implement :meth:`rewrite` to transform a user query
    into one or more expanded/reformulated variants for retrieval.
    """

    @abstractmethod
    def rewrite(
        self,
        query: str,
        conversation_history: list[Any] | None = None,
        **kwargs: Any,
    ) -> RewriteResult:
        """Rewrite a query into one or more search-optimised variants.

        Args:
            query: The original user query.
            conversation_history: Optional prior conversation turns.
            **kwargs: Strategy-specific parameters.

        Returns:
            A RewriteResult containing the rewritten queries.
        """
