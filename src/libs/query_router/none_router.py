"""No-op query router — always routes to knowledge_search."""

from __future__ import annotations

from typing import Any

from src.libs.query_router.base_router import BaseQueryRouter, RouteDecision


class NoneRouter(BaseQueryRouter):
    """Pass-through router used when query routing is disabled.

    Always returns ``knowledge_search`` with full confidence,
    ensuring all queries go through the standard RAG pipeline.
    """

    def route(self, query: str, **kwargs: Any) -> RouteDecision:
        return RouteDecision(
            route="knowledge_search",
            confidence=1.0,
            tool_name=None,
            reasoning=None,
        )
