"""LLM-based query rewriter — uses an LLM to expand/decompose queries."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.query_rewriter.base_rewriter import BaseQueryRewriter, RewriteResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a search query rewriting assistant. Given a user query, rewrite it \
into one or more optimised search queries for a RAG knowledge base.

Rules:
- Return valid JSON: {{"queries": ["..."], "reasoning": "..."}}
- Each query should be self-contained and clear
- If the query is complex, decompose into sub-queries
- If conversation history is provided, resolve pronouns and references
- Return between 1 and {max_rewrites} queries
"""

_HISTORY_CONTEXT = """\
Previous conversation:
{history}

Given this context, rewrite the following query to be self-contained.
"""


class LLMRewriter(BaseQueryRewriter):
    """Rewrites queries using an LLM for expansion and decomposition.

    The LLM receives a system prompt instructing it to return structured JSON
    with rewritten queries and reasoning. Falls back to the original query
    if the LLM response cannot be parsed.

    Args:
        llm: A BaseLLM instance for generating rewrites.
        max_rewrites: Maximum number of rewritten queries to return.
    """

    def __init__(self, llm: BaseLLM, max_rewrites: int = 3) -> None:
        self._llm = llm
        self._max_rewrites = max_rewrites

    def rewrite(
        self,
        query: str,
        conversation_history: list[Any] | None = None,
        **kwargs: Any,
    ) -> RewriteResult:
        try:
            messages = self._build_messages(query, conversation_history)
            response = self._llm.chat(messages)
            return self._parse_response(query, response.content)
        except Exception as exc:
            logger.warning("LLMRewriter failed, falling back to original: %s", exc)
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                reasoning=None,
                strategy="llm",
            )

    def _build_messages(
        self,
        query: str,
        conversation_history: list[Any] | None,
    ) -> list[Message]:
        system = _SYSTEM_PROMPT.format(max_rewrites=self._max_rewrites)
        messages = [Message("system", system)]

        if conversation_history:
            history_text = "\n".join(
                f"{m.role}: {m.content}"
                if hasattr(m, "role")
                else str(m)
                for m in conversation_history
            )
            context = _HISTORY_CONTEXT.format(history=history_text)
            messages.append(Message("user", f"{context}\nQuery: {query}"))
        else:
            messages.append(Message("user", f"Rewrite this query: {query}"))

        return messages

    def _parse_response(self, original_query: str, content: str) -> RewriteResult:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, falling back to original")
            return RewriteResult(
                original_query=original_query,
                rewritten_queries=[original_query],
                reasoning=None,
                strategy="llm",
            )

        queries = data.get("queries")
        if not queries or not isinstance(queries, list):
            return RewriteResult(
                original_query=original_query,
                rewritten_queries=[original_query],
                reasoning=data.get("reasoning"),
                strategy="llm",
            )

        # Enforce max_rewrites limit
        queries = queries[: self._max_rewrites]

        return RewriteResult(
            original_query=original_query,
            rewritten_queries=queries,
            reasoning=data.get("reasoning"),
            strategy="llm",
        )
