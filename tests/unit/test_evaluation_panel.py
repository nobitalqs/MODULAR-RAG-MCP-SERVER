"""Unit tests for the Evaluation Panel page.

Tests verify:
- render() with failed settings shows error
- render() with valid settings shows config summary and controls
- _render_config_summary displays evaluation config
- _render_report displays aggregate metrics and per-query details
- _run_evaluation stores report in session state
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_eval_settings() -> MagicMock:
    """Create mock evaluation settings."""
    eval_cfg = MagicMock()
    eval_cfg.enabled = True
    eval_cfg.provider = "custom"
    eval_cfg.metrics = ["hit_rate", "mrr"]
    return eval_cfg


def _make_settings() -> MagicMock:
    """Create mock Settings with evaluation config."""
    settings = MagicMock()
    settings.evaluation = _make_eval_settings()
    return settings


def _make_report_dict() -> Dict[str, Any]:
    """Create a sample evaluation report dict."""
    return {
        "evaluator_name": "StubEvaluator",
        "test_set_path": "tests/fixtures/golden_test_set.json",
        "total_elapsed_ms": 42.5,
        "aggregate_metrics": {"hit_rate": 1.0, "mrr": 0.75},
        "query_count": 2,
        "query_results": [
            {
                "query": "What is RAG?",
                "retrieved_chunk_ids": ["c1", "c2"],
                "generated_answer": "RAG is...",
                "metrics": {"hit_rate": 1.0, "mrr": 1.0},
                "elapsed_ms": 20.0,
            },
            {
                "query": "How does BM25 work?",
                "retrieved_chunk_ids": [],
                "generated_answer": "",
                "metrics": {"hit_rate": 1.0, "mrr": 0.5},
                "elapsed_ms": 22.5,
            },
        ],
    }


# ===================================================================
# Tests: render() with failed settings
# ===================================================================


class TestRenderSettingsFailure:
    """When settings fail to load, show error."""

    @patch("src.observability.dashboard.pages.evaluation_panel._load_settings_safe", return_value=None)
    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_settings_failure_shows_error(self, mock_st: MagicMock, _mock_load: MagicMock):
        from src.observability.dashboard.pages.evaluation_panel import render

        render()

        mock_st.header.assert_called_once_with("Evaluation Panel")
        mock_st.error.assert_called_once()


# ===================================================================
# Tests: render() with valid settings
# ===================================================================


class TestRenderWithSettings:
    """When settings load, show config summary and controls."""

    @patch("src.observability.dashboard.pages.evaluation_panel._load_settings_safe")
    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_shows_config_and_controls(self, mock_st: MagicMock, mock_load: MagicMock):
        from src.observability.dashboard.pages.evaluation_panel import render

        mock_load.return_value = _make_settings()
        _setup_st_columns(mock_st)
        mock_st.text_input.return_value = "tests/fixtures/golden_test_set.json"
        mock_st.slider.return_value = 10
        mock_st.button.return_value = False
        mock_st.session_state = {}

        render()

        mock_st.header.assert_called_once_with("Evaluation Panel")
        # Config summary rendered via st.metric
        assert mock_st.metric.call_count >= 3
        # Run button rendered
        mock_st.button.assert_called_once()


# ===================================================================
# Tests: _render_config_summary
# ===================================================================


class TestRenderConfigSummary:
    """Test evaluation config display."""

    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_displays_enabled_provider_metrics(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.evaluation_panel import _render_config_summary

        _setup_st_columns(mock_st)
        eval_cfg = _make_eval_settings()

        _render_config_summary(eval_cfg)

        metric_calls = [str(c) for c in mock_st.metric.call_args_list]
        assert any("Enabled" in c for c in metric_calls)
        assert any("custom" in c for c in metric_calls)

    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_displays_disabled_status(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.evaluation_panel import _render_config_summary

        _setup_st_columns(mock_st)
        eval_cfg = _make_eval_settings()
        eval_cfg.enabled = False

        _render_config_summary(eval_cfg)

        metric_calls = [str(c) for c in mock_st.metric.call_args_list]
        assert any("Disabled" in c for c in metric_calls)


# ===================================================================
# Tests: _render_report
# ===================================================================


class TestRenderReport:
    """Test evaluation report rendering."""

    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_renders_aggregate_metrics(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.evaluation_panel import _render_report

        _setup_st_columns(mock_st)
        exp_mock = MagicMock()
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = exp_mock

        report = _make_report_dict()
        _render_report(report)

        # Check aggregate metric values rendered
        metric_calls = [str(c) for c in mock_st.metric.call_args_list]
        assert any("hit_rate" in c for c in metric_calls)
        assert any("mrr" in c for c in metric_calls)

    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_renders_per_query_expanders(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.evaluation_panel import _render_report

        _setup_st_columns(mock_st)
        exp_mock = MagicMock()
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = exp_mock

        report = _make_report_dict()
        _render_report(report)

        # 2 query results -> 2 expander calls
        assert mock_st.expander.call_count == 2

    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_renders_download_button(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.evaluation_panel import _render_report

        _setup_st_columns(mock_st)
        exp_mock = MagicMock()
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = exp_mock

        report = _make_report_dict()
        _render_report(report)

        mock_st.download_button.assert_called_once()

    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_empty_metrics_shows_info(self, mock_st: MagicMock):
        from src.observability.dashboard.pages.evaluation_panel import _render_report

        _setup_st_columns(mock_st)
        exp_mock = MagicMock()
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = exp_mock

        report = _make_report_dict()
        report["aggregate_metrics"] = {}
        _render_report(report)

        info_calls = [str(c) for c in mock_st.info.call_args_list]
        assert any("No aggregate metrics" in c for c in info_calls)


# ===================================================================
# Tests: _run_evaluation
# ===================================================================


class TestRunEvaluation:
    """Test the evaluation execution flow."""

    @patch("src.observability.dashboard.pages.evaluation_panel._build_hybrid_search_safe", return_value=None)
    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_missing_file_shows_error(self, mock_st: MagicMock, _mock_hs: MagicMock):
        from src.observability.dashboard.pages.evaluation_panel import _run_evaluation

        mock_st.session_state = {}

        _run_evaluation(
            settings=_make_settings(),
            golden_path="/nonexistent/path.json",
            top_k=10,
            collection=None,
        )

        mock_st.error.assert_called_once()
        assert "not found" in str(mock_st.error.call_args).lower()

    @patch("src.observability.evaluation.eval_runner.EvalRunner")
    @patch("src.libs.evaluator.EvaluatorFactory")
    @patch("src.observability.dashboard.pages.evaluation_panel._build_hybrid_search_safe", return_value=None)
    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_successful_run_stores_report(
        self,
        mock_st: MagicMock,
        _mock_hs: MagicMock,
        mock_factory: MagicMock,
        mock_runner_cls: MagicMock,
    ):
        from src.observability.dashboard.pages.evaluation_panel import _run_evaluation

        session_state: dict = {}
        mock_st.session_state = session_state
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

        mock_evaluator = MagicMock()
        mock_factory.create.return_value = mock_evaluator

        mock_report = MagicMock()
        mock_report.to_dict.return_value = _make_report_dict()
        mock_report.query_results = [MagicMock(), MagicMock()]
        mock_report.total_elapsed_ms = 42.5

        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = mock_report

        _run_evaluation(
            settings=_make_settings(),
            golden_path="tests/fixtures/golden_test_set.json",
            top_k=10,
            collection=None,
        )

        # Report should be stored in session state
        assert "eval_report" in session_state
        mock_st.success.assert_called_once()

    @patch("src.libs.evaluator.EvaluatorFactory")
    @patch("src.observability.dashboard.pages.evaluation_panel._build_hybrid_search_safe", return_value=None)
    @patch("src.observability.dashboard.pages.evaluation_panel.st")
    def test_evaluator_creation_failure_shows_error(
        self,
        mock_st: MagicMock,
        _mock_hs: MagicMock,
        mock_factory: MagicMock,
    ):
        from src.observability.dashboard.pages.evaluation_panel import _run_evaluation

        mock_st.session_state = {}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        mock_factory.create.side_effect = RuntimeError("No provider")

        _run_evaluation(
            settings=_make_settings(),
            golden_path="tests/fixtures/golden_test_set.json",
            top_k=10,
            collection=None,
        )

        error_calls = [str(c) for c in mock_st.error.call_args_list]
        assert any("evaluator" in c.lower() for c in error_calls)
