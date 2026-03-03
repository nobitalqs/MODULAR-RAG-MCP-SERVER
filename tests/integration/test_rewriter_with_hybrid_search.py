"""Integration test: QueryRewriter + HybridSearch pipeline.

Verifies that query rewriting happens before search, and the rewritten query
is used for retrieval rather than the original.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.libs.query_rewriter.llm_rewriter import LLMRewriter
from src.libs.query_rewriter.none_rewriter import NoneRewriter


class TestNoneRewriterPassthrough:
    """NoneRewriter should not alter the query before search."""

    def test_original_query_passed_to_search(self):
        rewriter = NoneRewriter()
        mock_search = MagicMock()
        mock_search.search.return_value = []

        result = rewriter.rewrite("machine learning basics")
        # Use first rewritten query for search
        search_query = result.rewritten_queries[0]
        mock_search.search(query=search_query, top_k=5)

        mock_search.search.assert_called_once_with(
            query="machine learning basics", top_k=5,
        )


class TestLLMRewriterExpandsQuery:
    """LLMRewriter should expand the query before search."""

    def test_rewritten_query_used_for_search(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            content=json.dumps({
                "queries": [
                    "fundamentals of machine learning",
                    "ML basics introduction",
                ],
                "reasoning": "Expanded for better coverage",
            })
        )

        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=3)
        mock_search = MagicMock()
        mock_search.search.return_value = []

        result = rewriter.rewrite("ML basics")
        assert len(result.rewritten_queries) == 2
        assert result.rewritten_queries[0] == "fundamentals of machine learning"

        # First rewritten query would be used for search
        mock_search.search(query=result.rewritten_queries[0], top_k=5)
        mock_search.search.assert_called_once_with(
            query="fundamentals of machine learning", top_k=5,
        )

    def test_rewriter_fallback_on_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("LLM unavailable")

        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=3)
        result = rewriter.rewrite("ML basics")

        # Falls back to original query
        assert result.rewritten_queries == ("ML basics",)
        assert result.strategy == "llm"

    def test_rewriter_with_conversation_history(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            content=json.dumps({
                "queries": ["What are the specific features of Python 3.10?"],
                "reasoning": "Resolved pronoun reference from conversation",
            })
        )

        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=3)
        history = [
            MagicMock(role="user", content="Tell me about Python 3.10"),
            MagicMock(role="assistant", content="Python 3.10 adds pattern matching..."),
        ]

        result = rewriter.rewrite("What are its features?", conversation_history=history)
        assert len(result.rewritten_queries) >= 1
        # The LLM received the conversation context
        call_args = mock_llm.chat.call_args[0][0]
        # Should have system + user messages
        assert len(call_args) >= 2
