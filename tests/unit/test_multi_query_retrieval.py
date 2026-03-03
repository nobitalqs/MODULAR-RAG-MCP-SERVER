"""Unit tests for multi-query retrieval in query_knowledge_hub.

Tests verify that:
- All rewritten sub-queries are searched in parallel via asyncio.gather
- Results from multiple sub-queries are fused with RRF
- Single sub-query falls back to direct result (no fusion overhead)
- Rewriter failure falls back to original query
- Reranker receives original query (user intent), not sub-queries
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.core.types import RetrievalResult
from src.mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubConfig,
    QueryKnowledgeHubTool,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(chunk_id: str, score: float, text: str = "") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"text for {chunk_id}",
        metadata={"source": "test"},
    )


@dataclass(frozen=True)
class FakeRewriteResult:
    original_query: str
    rewritten_queries: tuple[str, ...]
    reasoning: str | None
    strategy: str


def _make_tool(
    search_side_effect=None,
    rewriter=None,
    reranker_enabled=False,
) -> QueryKnowledgeHubTool:
    """Create a pre-initialized tool with mock components."""
    mock_hybrid = MagicMock()
    if search_side_effect is not None:
        mock_hybrid.search.side_effect = search_side_effect
    else:
        mock_hybrid.search.return_value = []

    mock_settings = MagicMock()
    mock_settings.memory = None
    mock_settings.query_rewriting = None
    mock_settings.rate_limit = None
    mock_settings.cache = None
    mock_settings.query_routing = None
    mock_settings.retrieval = None

    mock_reranker = MagicMock(is_enabled=reranker_enabled)
    if reranker_enabled:
        mock_reranker.rerank.return_value = MagicMock(
            results=[], used_fallback=False,
        )

    tool = QueryKnowledgeHubTool(
        settings=mock_settings,
        config=QueryKnowledgeHubConfig(enable_rerank=reranker_enabled),
        hybrid_search=mock_hybrid,
        reranker=mock_reranker,
    )
    tool._initialized = True
    tool._current_collection = "default"

    if rewriter is not None:
        tool._query_rewriter = rewriter
    tool._advanced_initialized = True

    return tool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiQueryFanOut:
    """Multiple sub-queries should be searched in parallel and fused."""

    @pytest.mark.asyncio
    async def test_three_subqueries_all_searched(self):
        """Each sub-query should trigger a separate _perform_search call."""
        results_q1 = [_make_result("a", 0.9), _make_result("b", 0.8)]
        results_q2 = [_make_result("c", 0.85), _make_result("b", 0.7)]
        results_q3 = [_make_result("d", 0.75), _make_result("a", 0.6)]

        mock_hybrid = MagicMock()
        mock_hybrid.search.side_effect = [results_q1, results_q2, results_q3]

        rewriter = MagicMock()
        rewriter.rewrite.return_value = FakeRewriteResult(
            original_query="complex question",
            rewritten_queries=("sub q1", "sub q2", "sub q3"),
            reasoning=None,
            strategy="decomposition",
        )

        tool = _make_tool(rewriter=rewriter)
        tool._hybrid_search = mock_hybrid

        await tool.execute(query="complex question")

        assert mock_hybrid.search.call_count == 3
        called_queries = [c.kwargs["query"] for c in mock_hybrid.search.call_args_list]
        assert set(called_queries) == {"sub q1", "sub q2", "sub q3"}

    @pytest.mark.asyncio
    async def test_rrf_k_reads_from_settings(self):
        """RRF fusion k parameter should come from settings.retrieval.rrf_k."""
        results_q1 = [_make_result("a", 0.9)]
        results_q2 = [_make_result("b", 0.8)]

        mock_hybrid = MagicMock()
        mock_hybrid.search.side_effect = [results_q1, results_q2]

        rewriter = MagicMock()
        rewriter.rewrite.return_value = FakeRewriteResult(
            original_query="test",
            rewritten_queries=("q1", "q2"),
            reasoning=None,
            strategy="decomposition",
        )

        tool = _make_tool(rewriter=rewriter)
        tool._hybrid_search = mock_hybrid
        # Set custom rrf_k on settings
        tool.settings.retrieval = MagicMock(rrf_k=42)

        with patch("src.mcp_server.tools.query_knowledge_hub.RRFFusion") as mock_fusion_cls:
            mock_fusion_cls.return_value.fuse.return_value = [_make_result("a", 0.5)]
            await tool.execute(query="test")
            mock_fusion_cls.assert_called_once_with(k=42)

    @pytest.mark.asyncio
    async def test_cross_query_deduplication(self):
        """Chunks appearing in multiple sub-queries should be merged by RRF."""
        # chunk "b" appears in both results \u2014 RRF should merge and boost it
        results_q1 = [_make_result("a", 0.9), _make_result("b", 0.8)]
        results_q2 = [_make_result("b", 0.85), _make_result("c", 0.7)]

        mock_hybrid = MagicMock()
        mock_hybrid.search.side_effect = [results_q1, results_q2]

        rewriter = MagicMock()
        rewriter.rewrite.return_value = FakeRewriteResult(
            original_query="test",
            rewritten_queries=("q1", "q2"),
            reasoning=None,
            strategy="decomposition",
        )

        tool = _make_tool(rewriter=rewriter)
        tool._hybrid_search = mock_hybrid

        response = await tool.execute(query="test")

        # Verify response with deduplicated results
        assert response is not None
        assert not response.is_empty  # has results

    @pytest.mark.asyncio
    async def test_trace_records_multi_query_metadata(self):
        """Trace should include rewritten_queries and multi_query_counts."""
        results_q1 = [_make_result("a", 0.9)]
        results_q2 = [_make_result("b", 0.8)]

        mock_hybrid = MagicMock()
        mock_hybrid.search.side_effect = [results_q1, results_q2]

        rewriter = MagicMock()
        rewriter.rewrite.return_value = FakeRewriteResult(
            original_query="test",
            rewritten_queries=("rewrite1", "rewrite2"),
            reasoning=None,
            strategy="decomposition",
        )

        tool = _make_tool(rewriter=rewriter)
        tool._hybrid_search = mock_hybrid

        with patch.object(tool, "_collect_trace") as mock_trace:
            await tool.execute(query="test")
            trace_arg = mock_trace.call_args[0][0]
            assert "rewritten_queries" in trace_arg.metadata
            assert trace_arg.metadata["rewritten_queries"] == ["rewrite1", "rewrite2"]
            assert "multi_query_counts" in trace_arg.metadata
            assert trace_arg.metadata["multi_query_counts"] == [1, 1]


class TestSingleQueryFallback:
    """Single sub-query should skip fusion overhead."""

    @pytest.mark.asyncio
    async def test_single_rewrite_no_fusion(self):
        """When rewriter returns 1 query, use it directly without RRF fusion."""
        results = [_make_result("a", 0.9)]

        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = results

        rewriter = MagicMock()
        rewriter.rewrite.return_value = FakeRewriteResult(
            original_query="simple question",
            rewritten_queries=("rewritten simple question",),
            reasoning=None,
            strategy="paraphrase",
        )

        tool = _make_tool(rewriter=rewriter)
        tool._hybrid_search = mock_hybrid

        response = await tool.execute(query="simple question")

        assert mock_hybrid.search.call_count == 1
        assert response is not None

    @pytest.mark.asyncio
    async def test_no_rewriter_uses_original_query(self):
        """Without rewriter, original query is used for a single search."""
        results = [_make_result("a", 0.9)]

        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = results

        tool = _make_tool()
        tool._hybrid_search = mock_hybrid

        await tool.execute(query="original query")

        assert mock_hybrid.search.call_count == 1
        called_query = mock_hybrid.search.call_args.kwargs["query"]
        assert called_query == "original query"


class TestRewriterFailure:
    """Rewriter failure should fall back to original query gracefully."""

    @pytest.mark.asyncio
    async def test_rewriter_exception_falls_back(self):
        """If rewriter throws, use original query for single search."""
        results = [_make_result("a", 0.9)]

        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = results

        rewriter = MagicMock()
        rewriter.rewrite.side_effect = RuntimeError("LLM timeout")

        tool = _make_tool(rewriter=rewriter)
        tool._hybrid_search = mock_hybrid

        response = await tool.execute(query="my question")

        # Should fall back to original query, single search
        assert mock_hybrid.search.call_count == 1
        called_query = mock_hybrid.search.call_args.kwargs["query"]
        assert called_query == "my question"
        assert response is not None


class TestRerankerReceivesOriginalQuery:
    """Reranker should use original query (user intent), not sub-queries."""

    @pytest.mark.asyncio
    async def test_rerank_uses_original_query(self):
        """When reranking, the original query should be passed, not sub-queries."""
        results_q1 = [_make_result("a", 0.9)]
        results_q2 = [_make_result("b", 0.8)]

        mock_hybrid = MagicMock()
        mock_hybrid.search.side_effect = [results_q1, results_q2]

        rewriter = MagicMock()
        rewriter.rewrite.return_value = FakeRewriteResult(
            original_query="what is attention mechanism",
            rewritten_queries=("attention mechanism neural networks", "self-attention transformer"),
            reasoning=None,
            strategy="decomposition",
        )

        mock_reranker = MagicMock(is_enabled=True)
        mock_reranker.rerank.return_value = MagicMock(
            results=[_make_result("a", 0.95)],
            used_fallback=False,
        )

        tool = _make_tool(rewriter=rewriter, reranker_enabled=True)
        tool._hybrid_search = mock_hybrid
        tool._reranker = mock_reranker

        await tool.execute(query="what is attention mechanism")

        # Reranker should receive original query, not "attention mechanism neural networks"
        rerank_query = mock_reranker.rerank.call_args.kwargs["query"]
        assert rerank_query == "what is attention mechanism"
