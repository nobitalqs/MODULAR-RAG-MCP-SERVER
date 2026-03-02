"""Abstract base class and data types for query routers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    """Immutable result of a routing decision.

    Attributes:
        route: The chosen route name (e.g. ``"knowledge_search"``,
            ``"direct_answer"``, ``"tool_call"``).
        confidence: Confidence score between 0.0 and 1.0.
        tool_name: Optional tool identifier for ``tool_call`` routes.
        reasoning: Optional explanation of the routing decision.
    """

    route: str
    confidence: float
    tool_name: str | None
    reasoning: str | None


class BaseQueryRouter(ABC):
    """Pluggable interface for query intent classification.

    Subclasses must implement :meth:`route` to classify a user query
    into one of the configured route categories.
    """

    @abstractmethod
    def route(self, query: str, **kwargs: Any) -> RouteDecision:
        """Classify a query and return a routing decision.

        Args:
            query: The user query to classify.
            **kwargs: Additional context (conversation_history, etc.).

        Returns:
            A RouteDecision indicating which route to take.
        """
