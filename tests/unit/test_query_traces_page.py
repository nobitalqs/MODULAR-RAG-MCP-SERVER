"""Unit tests for Query Traces page -- stage renderers and page logic.

Tests verify:
- render() with empty traces shows info message
- render() with traces shows trace history
- Keyword filter narrows displayed traces
- Per-stage renderers receive correct data
- _render_chunk_list produces chunk cards with scores
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: sample trace data matching query pipeline record_stage() output
# ---------------------------------------------------------------------------

def _make_query_trace(
    trace_id: str = "qt1",
    started_at: str = "2026-02-18T14:00:00Z",
    elapsed_ms: float = 250.0,
    query: str = "What is RAG?",
    source: str = "mcp",
    top_k: int = 10,
    collection: str = "default",
    stages: list | None = None,
) -> Dict[str, Any]:
    """Build a query trace dict matching TraceContext.to_dict() format."""
    return {
        "trace_id": trace_id,
        "trace_type": "query",
        "started_at": started_at,
        "finished_at": "2026-02-18T14:00:01Z",
        "elapsed_ms": elapsed_ms,
        "stages": stages or _default_query_stages(),
        "metadata": {
            "query": query,
            "source": source,
            "top_k": top_k,
            "collection": collection,
        },
    }


def _default_query_stages() -> List[Dict[str, Any]]:
    """Return typical 5-stage query pipeline trace data."""
    return [
        {
            "stage": "query_processing",
            "elapsed_ms": 5.0,
            "data": {
                "method": "query_processor",
                "original_query": "What is RAG?",
                "keywords": ["rag", "retrieval", "augmented", "generation"],
            },
        },
        {
            "stage": "dense_retrieval",
            "elapsed_ms": 80.0,
            "data": {
                "method": "dense",
                "top_k": 10,
                "result_count": 5,
                "chunks": [
                    {"chunk_id": "c1", "score": 0.92, "text": "RAG is...", "source": "doc.pdf"},
                    {"chunk_id": "c2", "score": 0.85, "text": "Retrieval augmented...", "source": "doc.pdf"},
                ],
            },
        },
        {
            "stage": "sparse_retrieval",
            "elapsed_ms": 30.0,
            "data": {
                "method": "bm25",
                "keyword_count": 4,
                "top_k": 10,
                "result_count": 3,
                "chunks": [
                    {"chunk_id": "c1", "score": 0.7, "text": "RAG is...", "source": "doc.pdf"},
                    {"chunk_id": "c3", "score": 0.6, "text": "Generation...", "source": "other.pdf"},
                ],
            },
        },
        {
            "stage": "fusion",
            "elapsed_ms": 2.0,
            "data": {
                "method": "rrf",
                "input_lists": 2,
                "top_k": 10,
                "result_count": 4,
                "chunks": [
                    {"chunk_id": "c1", "score": 0.95, "text": "RAG is...", "source": "doc.pdf"},
                    {"chunk_id": "c2", "score": 0.85, "text": "Retrieval augmented...", "source": "doc.pdf"},
                    {"chunk_id": "c3", "score": 0.6, "text": "Generation...", "source": "other.pdf"},
                ],
            },
        },
        {
            "stage": "rerank",
            "elapsed_ms": 120.0,
            "data": {
                "method": "cross_encoder",
                "provider": "ms-marco-MiniLM",
                "input_count": 4,
                "output_count": 3,
                "chunks": [
                    {"chunk_id": "c1", "score": 0.98, "text": "RAG is...", "source": "doc.pdf"},
                    {"chunk_id": "c2", "score": 0.88, "text": "Retrieval augmented...", "source": "doc.pdf"},
                ],
            },
        },
    ]


# ===================================================================
# Tests: render() empty state
# ===================================================================

class TestRenderEmpty:
    """When no query traces exist, show info message."""

    @patch("src.observability.dashboard.pages.query_traces.TraceService")
    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_empty_traces_shows_info(self, mock_st: MagicMock, mock_svc_cls: MagicMock):
        from src.observability.dashboard.pages.query_traces import render

        mock_svc = mock_svc_cls.return_value
        mock_svc.list_traces.return_value = []

        render()

        mock_st.header.assert_called_once()
        mock_st.info.assert_called_once()
        mock_svc.list_traces.assert_called_once_with(trace_type="query")


# ===================================================================
# Tests: render() with traces
# ===================================================================

def _make_col_mock() -> MagicMock:
    """Create a MagicMock that works as a st.columns context manager."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m


def _setup_st_columns(mock_st: MagicMock) -> None:
    """Configure st.columns to return the right number of column mocks."""
    def columns_side_effect(spec, **kwargs):
        if isinstance(spec, int):
            return [_make_col_mock() for _ in range(spec)]
        if isinstance(spec, list):
            return [_make_col_mock() for _ in spec]
        return [_make_col_mock()]
    mock_st.columns.side_effect = columns_side_effect


class TestRenderWithTraces:
    """When query traces exist, render trace history."""

    @patch("src.observability.dashboard.pages.query_traces.TraceService")
    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_renders_trace_count_in_subheader(self, mock_st: MagicMock, mock_svc_cls: MagicMock):
        from src.observability.dashboard.pages.query_traces import render

        mock_svc = mock_svc_cls.return_value
        mock_svc.list_traces.return_value = [
            _make_query_trace("qt1"),
            _make_query_trace("qt2", query="Another query"),
        ]
        mock_svc.get_stage_timings.return_value = []

        mock_st.text_input.return_value = ""
        _setup_st_columns(mock_st)
        exp_mock = MagicMock()
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = exp_mock
        mock_st.tabs.return_value = []

        render()

        calls = [str(c) for c in mock_st.subheader.call_args_list]
        assert any("2" in c for c in calls)

    @patch("src.observability.dashboard.pages.query_traces.TraceService")
    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_keyword_filter_narrows_traces(self, mock_st: MagicMock, mock_svc_cls: MagicMock):
        from src.observability.dashboard.pages.query_traces import render

        mock_svc = mock_svc_cls.return_value
        mock_svc.list_traces.return_value = [
            _make_query_trace("qt1", query="What is RAG?"),
            _make_query_trace("qt2", query="Python best practices"),
        ]
        mock_svc.get_stage_timings.return_value = []

        mock_st.text_input.return_value = "python"
        _setup_st_columns(mock_st)
        exp_mock = MagicMock()
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = exp_mock
        mock_st.tabs.return_value = []

        render()

        calls = [str(c) for c in mock_st.subheader.call_args_list]
        assert any("1" in c for c in calls)


# ===================================================================
# Tests: Per-stage renderers
# ===================================================================

class TestQueryProcessingStageRenderer:
    """Test _render_query_processing_stage."""

    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_renders_original_query_and_keywords(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.query_traces import _render_query_processing_stage

        data = {
            "original_query": "What is RAG?",
            "method": "query_processor",
            "keywords": ["rag", "retrieval"],
        }

        # Mock columns context manager
        col_mock = MagicMock()
        mock_st.columns.return_value = [col_mock, col_mock]
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)

        _render_query_processing_stage(data)

        # Should call st.info with the original query
        mock_st.info.assert_called()
        assert any("What is RAG?" in str(c) for c in mock_st.info.call_args_list)

    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_empty_keywords_shows_warning(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.query_traces import _render_query_processing_stage

        data = {
            "original_query": "test",
            "method": "query_processor",
            "keywords": [],
        }

        col_mock = MagicMock()
        mock_st.columns.return_value = [col_mock, col_mock]
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)

        _render_query_processing_stage(data)

        mock_st.warning.assert_called()


class TestRetrievalStageRenderer:
    """Test _render_retrieval_stage for both Dense and Sparse."""

    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_renders_dense_metrics(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.query_traces import _render_retrieval_stage

        data = {
            "method": "dense",
            "provider": "openai",
            "top_k": 10,
            "result_count": 5,
            "chunks": [
                {"chunk_id": "c1", "score": 0.92, "text": "RAG text", "source": "doc.pdf"},
            ],
        }

        col_mock = MagicMock()
        mock_st.columns.return_value = [col_mock, col_mock, col_mock]
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)

        exp_mock = MagicMock()
        mock_st.expander.return_value = exp_mock
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)

        _render_retrieval_stage(data, "Dense")

        # Should call metric for Results
        metric_calls = [str(c) for c in mock_st.metric.call_args_list]
        assert any("Results" in c for c in metric_calls)

    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_empty_chunks_shows_info(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.query_traces import _render_retrieval_stage

        data = {
            "method": "bm25",
            "keyword_count": 3,
            "top_k": 10,
            "result_count": 0,
            "chunks": [],
        }

        col_mock = MagicMock()
        mock_st.columns.return_value = [col_mock, col_mock, col_mock]
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)

        _render_retrieval_stage(data, "Sparse")

        mock_st.info.assert_called()


class TestFusionStageRenderer:
    """Test _render_fusion_stage."""

    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_renders_fusion_metrics(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.query_traces import _render_fusion_stage

        data = {
            "method": "rrf",
            "input_lists": 2,
            "top_k": 10,
            "result_count": 4,
            "chunks": [],
        }

        col_mock = MagicMock()
        mock_st.columns.return_value = [col_mock, col_mock, col_mock]
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)

        _render_fusion_stage(data)

        metric_calls = [str(c) for c in mock_st.metric.call_args_list]
        assert any("rrf" in c for c in metric_calls)


class TestRerankStageRenderer:
    """Test _render_rerank_stage."""

    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_renders_rerank_metrics(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.query_traces import _render_rerank_stage

        data = {
            "method": "cross_encoder",
            "provider": "ms-marco-MiniLM",
            "input_count": 4,
            "output_count": 3,
            "chunks": [],
        }

        col_mock = MagicMock()
        mock_st.columns.return_value = [col_mock, col_mock, col_mock, col_mock]
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)

        _render_rerank_stage(data)

        metric_calls = [str(c) for c in mock_st.metric.call_args_list]
        assert any("cross_encoder" in c for c in metric_calls)
        assert any("Output" in c for c in metric_calls)


# ===================================================================
# Tests: _render_chunk_list
# ===================================================================

class TestRenderChunkList:
    """Test the shared chunk list renderer."""

    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_renders_chunks_with_score_colors(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.query_traces import _render_chunk_list

        chunks = [
            {"chunk_id": "c1", "score": 0.92, "text": "High score", "source": "a.pdf"},
            {"chunk_id": "c2", "score": 0.55, "text": "Medium score", "source": "b.pdf"},
            {"chunk_id": "c3", "score": 0.2, "text": "Low score", "source": "c.pdf"},
        ]

        exp_mock = MagicMock()
        mock_st.expander.return_value = exp_mock
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)

        col_mock = MagicMock()
        mock_st.columns.return_value = [col_mock, col_mock]
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)

        _render_chunk_list(chunks, prefix="test")

        # Should call expander 3 times (once per chunk)
        assert mock_st.expander.call_count == 3

        # Check score color indicators in expander titles
        expander_calls = [str(c) for c in mock_st.expander.call_args_list]
        # High score (>=0.8) should have [HIGH] indicator
        assert any("0.9200" in c for c in expander_calls)
        # Low score (<0.5) should be present
        assert any("0.2000" in c for c in expander_calls)

    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_empty_chunks_does_nothing(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.query_traces import _render_chunk_list

        _render_chunk_list([], prefix="test")

        mock_st.expander.assert_not_called()

    @patch("src.observability.dashboard.pages.query_traces.st")
    def test_chunk_without_text_shows_fallback(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.query_traces import _render_chunk_list

        chunks = [{"chunk_id": "c1", "score": 0.5, "text": "", "source": ""}]

        exp_mock = MagicMock()
        mock_st.expander.return_value = exp_mock
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)

        col_mock = MagicMock()
        mock_st.columns.return_value = [col_mock, col_mock]
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)

        _render_chunk_list(chunks, prefix="test")

        # Should show fallback caption for empty text
        mock_st.caption.assert_called()
