"""Unit tests for confidence-based adaptive retrieval.

Tests verify the adaptive retry logic inserted between rerank and top_k slicing
in QueryKnowledgeHubTool.execute().

Covers: trigger on low score, skip on high score, skip when disabled,
skip when rerank off, max retries, trace metadata recording.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import AdaptiveRetrievalSettings, RetrievalSettings, Settings
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


# ---------------------------------------------------------------------------
# Settings parsing tests
# ---------------------------------------------------------------------------


def _minimal_settings_dict(**retrieval_overrides: object) -> dict:
    """Build a minimal valid settings dict for Settings.from_dict()."""
    retrieval = {
        "dense_top_k": 20,
        "sparse_top_k": 20,
        "fusion_top_k": 10,
        "rrf_k": 60,
    }
    retrieval.update(retrieval_overrides)
    return {
        "llm": {
            "provider": "ollama",
            "model": "m",
            "temperature": 0.0,
            "max_tokens": 100,
            "base_url": "http://localhost:11434",
        },
        "embedding": {
            "provider": "ollama",
            "model": "m",
            "dimensions": 768,
            "base_url": "http://localhost:11434",
        },
        "vector_store": {
            "provider": "chroma",
            "persist_directory": "./data",
            "collection_name": "test",
        },
        "retrieval": retrieval,
        "rerank": {
            "enabled": False,
            "provider": "none",
            "model": "none",
            "top_k": 5,
        },
        "evaluation": {
            "enabled": False,
            "provider": "custom",
            "metrics": ["context_precision"],
        },
        "observability": {
            "log_level": "WARNING",
            "trace_enabled": False,
            "trace_file": "./logs/traces.jsonl",
            "structured_logging": False,
        },
    }


class TestAdaptiveRetrievalSettings:
    """Settings dataclass and from_dict parsing tests."""

    def test_retrieval_settings_default_no_adaptive(self):
        """RetrievalSettings without adaptive should have None."""
        rs = RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60)
        assert rs.adaptive is None

    def test_adaptive_settings_frozen(self):
        """AdaptiveRetrievalSettings should be immutable."""
        s = AdaptiveRetrievalSettings(
            enabled=True, score_threshold=0.0, expand_factor=2, max_retries=1,
        )
        with pytest.raises(AttributeError):
            s.enabled = False  # type: ignore[misc]

    def test_retrieval_with_adaptive(self):
        """RetrievalSettings should accept adaptive sub-config."""
        adaptive = AdaptiveRetrievalSettings(
            enabled=True, score_threshold=0.5, expand_factor=3, max_retries=2,
        )
        rs = RetrievalSettings(
            dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60,
            adaptive=adaptive,
        )
        assert rs.adaptive is not None
        assert rs.adaptive.score_threshold == 0.5
        assert rs.adaptive.expand_factor == 3

    def test_from_dict_parses_adaptive(self):
        """Settings.from_dict should parse retrieval.adaptive section."""
        data = _minimal_settings_dict(adaptive={
            "enabled": True,
            "score_threshold": -1.0,
            "expand_factor": 2,
            "max_retries": 1,
        })
        settings = Settings.from_dict(data)
        assert settings.retrieval.adaptive is not None
        assert settings.retrieval.adaptive.enabled is True
        assert settings.retrieval.adaptive.score_threshold == -1.0
        assert settings.retrieval.adaptive.expand_factor == 2
        assert settings.retrieval.adaptive.max_retries == 1

    def test_from_dict_no_adaptive_gives_none(self):
        """Settings.from_dict without adaptive section -> None."""
        data = _minimal_settings_dict()
        settings = Settings.from_dict(data)
        assert settings.retrieval.adaptive is None
