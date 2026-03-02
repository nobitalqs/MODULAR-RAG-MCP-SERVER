"""HyDE (Hypothetical Document Embeddings) query rewriter.

Generates a hypothetical answer document via LLM, which is then used
as the search query. The intuition: a hypothetical answer is closer in
embedding space to real answer documents than the original question.
"""

from __future__ import annotations

import logging
from typing import Any

from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.query_rewriter.base_rewriter import BaseQueryRewriter, RewriteResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a knowledge expert. Given a question, write a short, factual "
    "paragraph that directly answers it. This hypothetical answer will be "
    "used for semantic search. Write only the answer — no preamble, no "
    "disclaimers, no references to the question."
)


class HyDERewriter(BaseQueryRewriter):
    """Generates a hypothetical answer document for embedding-based search.

    Instead of rewriting the query, HyDE asks the LLM to generate a
    plausible answer. This hypothetical document is then used as the
    search query, which tends to match real answer documents more closely
    in embedding space.

    Args:
        llm: A BaseLLM instance for generating the hypothetical answer.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def rewrite(
        self,
        query: str,
        conversation_history: list[Any] | None = None,
        **kwargs: Any,
    ) -> RewriteResult:
        try:
            messages = [
                Message("system", _SYSTEM_PROMPT),
                Message("user", query),
            ]
            response = self._llm.chat(messages)
            hypothetical_doc = response.content.strip()

            if not hypothetical_doc:
                logger.warning("HyDE got empty response, falling back to original")
                return self._fallback(query)

            return RewriteResult(
                original_query=query,
                rewritten_queries=[hypothetical_doc],
                reasoning="Generated hypothetical answer document for semantic search",
                strategy="hyde",
            )
        except Exception as exc:
            logger.warning("HyDERewriter failed, falling back to original: %s", exc)
            return self._fallback(query)

    @staticmethod
    def _fallback(query: str) -> RewriteResult:
        return RewriteResult(
            original_query=query,
            rewritten_queries=[query],
            reasoning=None,
            strategy="hyde",
        )
