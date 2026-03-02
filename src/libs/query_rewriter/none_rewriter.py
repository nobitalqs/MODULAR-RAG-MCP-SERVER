"""No-op query rewriter — returns the original query unchanged."""

from __future__ import annotations

from typing import Any

from src.libs.query_rewriter.base_rewriter import BaseQueryRewriter, RewriteResult


class NoneRewriter(BaseQueryRewriter):
    """Pass-through rewriter used when query rewriting is disabled.

    Always returns the original query as the sole rewritten variant
    with ``strategy="none"`` and no reasoning.
    """

    def rewrite(
        self,
        query: str,
        conversation_history: list[Any] | None = None,
        **kwargs: Any,
    ) -> RewriteResult:
        return RewriteResult(
            original_query=query,
            rewritten_queries=[query],
            reasoning=None,
            strategy="none",
        )
