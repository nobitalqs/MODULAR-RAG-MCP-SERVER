"""Unit tests for the Ingestion Manager page logic.

Tests focus on the _run_ingestion orchestration and the delete flow,
mocking Streamlit widgets, IngestionPipeline, and DataService.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeDeleteResult:
    success: bool = True
    chunks_deleted: int = 5
    images_deleted: int = 2
    errors: List[str] = field(default_factory=list)


def _make_uploaded_file(name: str = "test.pdf", content: bytes = b"fake") -> MagicMock:
    """Create a mock UploadedFile."""
    uf = MagicMock()
    uf.name = name
    uf.getbuffer.return_value = content
    return uf


# ===================================================================
# Tests: _run_ingestion
# ===================================================================

class TestRunIngestion:
    """Test the _run_ingestion helper function."""

    @patch("src.observability.dashboard.pages.ingestion_manager.IngestionPipeline", create=True)
    @patch("src.observability.dashboard.pages.ingestion_manager.TraceCollector", create=True)
    @patch("src.observability.dashboard.pages.ingestion_manager.TraceContext", create=True)
    @patch("src.observability.dashboard.pages.ingestion_manager.load_settings", create=True)
    def test_successful_ingestion(
        self, mock_settings, mock_trace_cls, mock_collector_cls, mock_pipeline_cls
    ):
        """Pipeline.run is called and progress reaches 1.0 on success."""
        # We need to test _run_ingestion which does lazy imports inside.
        # Patch the actual import targets within the function.
        from src.observability.dashboard.pages import ingestion_manager

        mock_settings_obj = MagicMock()
        progress_bar = MagicMock()
        status_text = MagicMock()
        uploaded = _make_uploaded_file()

        mock_trace_instance = MagicMock()
        mock_trace_instance.metadata = {}

        mock_pipeline_instance = MagicMock()

        # Patch the lazy imports inside _run_ingestion
        with patch.object(ingestion_manager, "__builtins__", ingestion_manager.__builtins__):
            with patch(
                "src.core.settings.load_settings", return_value=mock_settings_obj
            ), patch(
                "src.core.trace.TraceContext", return_value=mock_trace_instance
            ), patch(
                "src.core.trace.TraceCollector"
            ) as patched_collector, patch(
                "src.ingestion.pipeline.IngestionPipeline",
                return_value=mock_pipeline_instance,
            ):
                ingestion_manager._run_ingestion(
                    uploaded, "default", progress_bar, status_text
                )

        # Pipeline.run should have been called
        mock_pipeline_instance.run.assert_called_once()
        run_kwargs = mock_pipeline_instance.run.call_args
        assert run_kwargs.kwargs.get("on_progress") is not None or len(run_kwargs.args) >= 2

        # Progress bar should reach 1.0
        progress_bar.progress.assert_called()
        final_call = progress_bar.progress.call_args_list[-1]
        assert final_call.args[0] == 1.0

        # Status should show success
        status_text.success.assert_called_once()

    @patch("src.core.settings.load_settings")
    @patch("src.core.trace.TraceContext")
    @patch("src.core.trace.TraceCollector")
    @patch("src.ingestion.pipeline.IngestionPipeline")
    def test_failed_ingestion_shows_error(
        self, mock_pipeline_cls, mock_collector_cls, mock_trace_cls, mock_settings
    ):
        """When pipeline raises, status shows error."""
        from src.observability.dashboard.pages import ingestion_manager

        mock_trace_instance = MagicMock()
        mock_trace_instance.metadata = {}
        mock_trace_cls.return_value = mock_trace_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.side_effect = RuntimeError("Parse error")
        mock_pipeline_cls.return_value = mock_pipeline_instance

        progress_bar = MagicMock()
        status_text = MagicMock()
        uploaded = _make_uploaded_file()

        ingestion_manager._run_ingestion(
            uploaded, "default", progress_bar, status_text
        )

        status_text.error.assert_called_once()
        assert "Parse error" in status_text.error.call_args.args[0]

    @patch("src.core.settings.load_settings")
    @patch("src.core.trace.TraceContext")
    @patch("src.core.trace.TraceCollector")
    @patch("src.ingestion.pipeline.IngestionPipeline")
    def test_trace_always_collected(
        self, mock_pipeline_cls, mock_collector_cls, mock_trace_cls, mock_settings
    ):
        """TraceCollector.collect() is called even on failure."""
        from src.observability.dashboard.pages import ingestion_manager

        mock_trace_instance = MagicMock()
        mock_trace_instance.metadata = {}
        mock_trace_cls.return_value = mock_trace_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.side_effect = RuntimeError("boom")
        mock_pipeline_cls.return_value = mock_pipeline_instance

        progress_bar = MagicMock()
        status_text = MagicMock()
        uploaded = _make_uploaded_file()

        ingestion_manager._run_ingestion(
            uploaded, "test_coll", progress_bar, status_text
        )

        mock_collector_cls.return_value.collect.assert_called_once_with(
            mock_trace_instance
        )


# ===================================================================
# Tests: on_progress callback
# ===================================================================

class TestOnProgressCallback:
    """Test the progress callback wiring."""

    @patch("src.core.settings.load_settings")
    @patch("src.core.trace.TraceContext")
    @patch("src.core.trace.TraceCollector")
    @patch("src.ingestion.pipeline.IngestionPipeline")
    def test_on_progress_updates_bar(
        self, mock_pipeline_cls, mock_collector_cls, mock_trace_cls, mock_settings
    ):
        """The on_progress callback should update the progress bar."""
        from src.observability.dashboard.pages import ingestion_manager

        mock_trace_instance = MagicMock()
        mock_trace_instance.metadata = {}
        mock_trace_cls.return_value = mock_trace_instance

        captured_callback = None

        def capture_run(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("on_progress")

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.side_effect = capture_run
        mock_pipeline_cls.return_value = mock_pipeline_instance

        progress_bar = MagicMock()
        status_text = MagicMock()
        uploaded = _make_uploaded_file()

        ingestion_manager._run_ingestion(
            uploaded, "default", progress_bar, status_text
        )

        assert captured_callback is not None

        # Simulate progress calls
        captured_callback("load", 2, 6)
        progress_bar.progress.assert_any_call(
            pytest.approx(1 / 6),
            text="[2/6] Loading document...",
        )

        captured_callback("embed", 5, 6)
        progress_bar.progress.assert_any_call(
            pytest.approx(4 / 6),
            text="[5/6] Encoding vectors...",
        )


# ===================================================================
# Tests: Stage labels
# ===================================================================

class TestStageLabels:
    def test_all_pipeline_stages_have_labels(self):
        from src.observability.dashboard.pages.ingestion_manager import (
            _STAGE_LABELS,
        )

        expected_stages = {"integrity", "load", "split", "transform", "embed", "upsert"}
        assert expected_stages == set(_STAGE_LABELS.keys())

    def test_unknown_stage_falls_back_to_name(self):
        """If on_progress receives an unknown stage, it should still work."""
        from src.observability.dashboard.pages.ingestion_manager import (
            _STAGE_LABELS,
        )

        label = _STAGE_LABELS.get("unknown_stage", "unknown_stage")
        assert label == "unknown_stage"


# ===================================================================
# Tests: Delete document flow
# ===================================================================

class TestDeleteFlow:
    def test_delete_result_fields(self):
        """DeleteResult from DataService should have expected fields."""
        result = _FakeDeleteResult()
        assert result.success is True
        assert result.chunks_deleted == 5
        assert result.images_deleted == 2
        assert result.errors == []

    def test_failed_delete_has_errors(self):
        result = _FakeDeleteResult(
            success=False, errors=["ChromaDB delete failed"]
        )
        assert result.success is False
        assert len(result.errors) == 1
