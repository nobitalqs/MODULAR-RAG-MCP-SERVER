"""Unit tests for confidence-based adaptive retrieval.

Tests verify the adaptive retry logic inserted between rerank and top_k slicing
in QueryKnowledgeHubTool.execute().

TDD RED phase: these tests define the contract BEFORE implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import AdaptiveRetrievalSettings
from src.core.types import RetrievalResult
from src.mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubConfig,
    QueryKnowledgeHubTool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(chunk_id: str, score: float) -> RetrievalResult:
    """Create a RetrievalResult with the given score."""
    return RetrievalResult(chunk_id=chunk_id, score=score, text=f"text-{chunk_id}")


def _make_rerank_response(results: list[RetrievalResult]) -> MagicMock:
    """Create a mock reranker response wrapping the given results."""
    resp = MagicMock()
    resp.results = results
    resp.used_fallback = False
    resp.fallback_reason = None
    return resp


def _make_settings(
    *,
    adaptive_enabled: bool = True,
    score_threshold: float = 0.5,
    expand_factor: int = 2,
    max_retries: int = 1,
) -> MagicMock:
    """Create mock Settings with retrieval.adaptive configured."""
    adaptive = AdaptiveRetrievalSettings(
        enabled=adaptive_enabled,
        score_threshold=score_threshold,
        expand_factor=expand_factor,
        max_retries=max_retries,
    )
    settings = MagicMock()
    settings.retrieval.adaptive = adaptive
    settings.retrieval.rrf_k = 60
    settings.memory = None
    settings.query_rewriting = None
    settings.rate_limit = None
    settings.cache = None
    settings.query_routing = None
    return settings


def _make_tool(
    settings: MagicMock,
    search_results: list[RetrievalResult],
    rerank_side_effect: list | None = None,
    *,
    enable_rerank: bool = True,
) -> QueryKnowledgeHubTool:
    """Build a tool with mocked hybrid_search and reranker.

    Args:
        settings: Mock Settings object.
        search_results: Results returned by hybrid_search.search on every call.
        rerank_side_effect: If provided, successive return values for reranker.rerank.
        enable_rerank: Whether reranking is enabled in config.
    """
    mock_hybrid = MagicMock()
    mock_hybrid.search.return_value = search_results

    mock_reranker = MagicMock()
    mock_reranker.is_enabled = enable_rerank
    if rerank_side_effect is not None:
        mock_reranker.rerank.side_effect = rerank_side_effect
    else:
        # Default: return results unchanged
        mock_reranker.rerank.return_value = _make_rerank_response(search_results)

    tool = QueryKnowledgeHubTool(
        settings=settings,
        config=QueryKnowledgeHubConfig(enable_rerank=enable_rerank),
        hybrid_search=mock_hybrid,
        reranker=mock_reranker,
    )
    tool._initialized = True
    tool._current_collection = "default"
    return tool


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestAdaptiveRetrieval:
    """Confidence-based adaptive retrieval tests."""

    @pytest.mark.asyncio
    async def test_triggers_on_low_score(self):
        """When top-1 score < threshold, search is called twice (initial + retry)."""
        low_results = [_make_result("a", 0.3)]
        better_results = [_make_result("b", 0.8)]

        settings = _make_settings(score_threshold=0.5, max_retries=1)
        tool = _make_tool(
            settings,
            search_results=low_results,
            rerank_side_effect=[
                _make_rerank_response(low_results),  # initial rerank
                _make_rerank_response(better_results),  # retry rerank
            ],
        )

        await tool.execute(query="test query")

        # hybrid_search.search called twice: initial + adaptive retry
        assert tool._hybrid_search.search.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_on_high_score(self):
        """When top-1 score >= threshold, search is called only once."""
        high_results = [_make_result("a", 0.9)]

        settings = _make_settings(score_threshold=0.5, max_retries=1)
        tool = _make_tool(
            settings,
            search_results=high_results,
            rerank_side_effect=[
                _make_rerank_response(high_results),
            ],
        )

        await tool.execute(query="test query")

        assert tool._hybrid_search.search.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        """When adaptive.enabled=False, search is called only once."""
        low_results = [_make_result("a", 0.2)]

        settings = _make_settings(adaptive_enabled=False, score_threshold=0.5)
        tool = _make_tool(
            settings,
            search_results=low_results,
            rerank_side_effect=[
                _make_rerank_response(low_results),
            ],
        )

        await tool.execute(query="test query")

        assert tool._hybrid_search.search.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_when_rerank_off(self):
        """When rerank is disabled, adaptive retrieval never triggers."""
        low_results = [_make_result("a", 0.2)]

        settings = _make_settings(score_threshold=0.5)
        tool = _make_tool(
            settings,
            search_results=low_results,
            enable_rerank=False,
        )

        await tool.execute(query="test query")

        assert tool._hybrid_search.search.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_respected(self):
        """Adaptive retry happens at most max_retries times."""
        low_results = [_make_result("a", 0.1)]

        settings = _make_settings(
            score_threshold=0.5,
            max_retries=3,
            expand_factor=2,
        )
        # Every rerank returns low score, so all retries fire
        rerank_responses = [_make_rerank_response(low_results)] * 4  # 1 initial + 3 retries
        tool = _make_tool(
            settings,
            search_results=low_results,
            rerank_side_effect=rerank_responses,
        )

        await tool.execute(query="test query")

        # 1 initial search + 3 retry searches = 4
        assert tool._hybrid_search.search.call_count == 4

    @pytest.mark.asyncio
    async def test_trace_metadata_on_retry(self):
        """When adaptive retry fires, trace.metadata has the expected keys."""
        low_results = [_make_result("a", 0.3)]
        better_results = [_make_result("b", 0.8)]

        settings = _make_settings(score_threshold=0.5, max_retries=1, expand_factor=2)

        tool = _make_tool(
            settings,
            search_results=low_results,
            rerank_side_effect=[
                _make_rerank_response(low_results),  # initial
                _make_rerank_response(better_results),  # retry
            ],
        )

        with patch(
            "src.mcp_server.tools.query_knowledge_hub.TraceContext",
        ) as mock_trace_cls:
            mock_trace = MagicMock()
            mock_trace.metadata = {}
            mock_trace_cls.return_value = mock_trace

            # Prevent _collect_trace from serializing MagicMock metadata
            with patch.object(tool, "_collect_trace"):
                await tool.execute(query="test query")

            meta = mock_trace.metadata
            assert meta.get("adaptive_retry") is True
            assert "adaptive_reason" in meta
            assert "top1_score=0.30" in meta["adaptive_reason"]
            assert "threshold=0.5" in meta["adaptive_reason"]
            assert meta.get("adaptive_expanded_top_k") == 10  # 5 * 2^1 = 10
