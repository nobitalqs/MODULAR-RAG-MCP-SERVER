"""Tests for C14 – Pipeline orchestration and on_progress callback.

Verifies that IngestionPipeline.run() fires the optional on_progress
callback at each pipeline stage with (stage_name, current, total),
and that the pipeline correctly orchestrates all 6 stages.

Uses a duck-typed fake pipeline object to avoid real settings/factories.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, Document
from src.ingestion.pipeline import STAGE_NAMES, TOTAL_STAGES, IngestionPipeline


# ── Helpers ──────────────────────────────────────────────────────────


def _make_fake_pipeline() -> object:
    """Build a fake IngestionPipeline that bypasses __init__."""

    class FP:
        collection = "test"
        force = False

    fp = FP()

    # Stage 1: integrity
    fp.integrity_checker = MagicMock()
    fp.integrity_checker.compute_sha256.return_value = "abc123"
    fp.integrity_checker.should_skip.return_value = False
    fp.integrity_checker.lookup_by_path.return_value = None

    # Document lifecycle manager
    fp.document_manager = MagicMock()

    # Stage 2: loader factory
    mock_loader = MagicMock()
    mock_loader.load.return_value = Document(
        id="doc1",
        text="Hello world. " * 50,
        metadata={"source_path": "test.pdf", "images": []},
    )
    fp.loader_factory = MagicMock()
    fp.loader_factory.create_for_file.return_value = mock_loader
    # Keep reference for tests that check loader calls
    fp._mock_loader = mock_loader
    # Pipeline reads these for kwargs
    fp._table_extraction = None
    fp._formula_extraction = None

    # Stage 3: chunker
    chunks = [
        Chunk(
            id=f"c{i}",
            text=f"Chunk {i} text. " * 5,
            metadata={
                "source_path": "test.pdf",
                "chunk_index": i,
            },
        )
        for i in range(3)
    ]
    fp.chunker = MagicMock()
    fp.chunker.split_document.return_value = chunks

    # Stage 4: transforms (pass-through)
    fp.chunk_refiner = MagicMock()
    fp.chunk_refiner.transform.return_value = chunks
    fp.metadata_enricher = MagicMock()
    fp.metadata_enricher.transform.return_value = chunks
    fp.image_captioner = MagicMock()
    fp.image_captioner.transform.return_value = chunks
    fp.retrieval_text_generator = MagicMock()
    fp.retrieval_text_generator.transform.return_value = chunks

    # Stage 5: encoding
    batch_result = MagicMock()
    batch_result.dense_vectors = [[0.1, 0.2]] * 3
    batch_result.sparse_stats = [
        {
            "chunk_id": f"c{i}",
            "term_frequencies": {"chunk": 1, "text": 5},
            "doc_length": 6,
            "unique_terms": 2,
        }
        for i in range(3)
    ]
    batch_result.batch_count = 1
    batch_result.successful_chunks = 3
    batch_result.failed_chunks = 0
    fp.batch_processor = MagicMock()
    fp.batch_processor.process.return_value = batch_result

    # Stage 6: storage
    fp.vector_upserter = MagicMock()
    fp.vector_upserter.upsert.return_value = ["v0", "v1", "v2"]
    fp.bm25_indexer = MagicMock()
    fp.image_storage = MagicMock()

    return fp


def _collect_progress(fp) -> list[tuple[str, int, int]]:
    """Run pipeline with a callback and return collected calls."""
    calls: list[tuple[str, int, int]] = []

    def on_progress(stage: str, current: int, total: int) -> None:
        calls.append((stage, current, total))

    IngestionPipeline.run(fp, "test.pdf", on_progress=on_progress)
    return calls


# ── Progress callback tests ──────────────────────────────────────────


class TestPipelineProgressCallback:
    """Verify on_progress is called correctly."""

    def test_callback_called_for_all_stages(self) -> None:
        fp = _make_fake_pipeline()
        calls = _collect_progress(fp)
        stage_names = [c[0] for c in calls]
        for name in STAGE_NAMES:
            assert name in stage_names

    def test_total_is_six(self) -> None:
        fp = _make_fake_pipeline()
        calls = _collect_progress(fp)
        for _, _, total in calls:
            assert total == TOTAL_STAGES

    def test_current_is_monotonically_increasing(self) -> None:
        fp = _make_fake_pipeline()
        calls = _collect_progress(fp)
        currents = [c[1] for c in calls]
        assert currents == list(range(1, TOTAL_STAGES + 1))

    def test_no_callback_no_crash(self) -> None:
        """on_progress=None should not break anything."""
        fp = _make_fake_pipeline()
        result = IngestionPipeline.run(fp, "test.pdf", on_progress=None)
        assert result.success

    def test_callback_with_trace(self) -> None:
        """on_progress and trace both work simultaneously."""
        fp = _make_fake_pipeline()
        calls: list[tuple[str, int, int]] = []
        trace = TraceContext(trace_type="ingestion")

        def on_progress(stage: str, current: int, total: int) -> None:
            calls.append((stage, current, total))

        IngestionPipeline.run(fp, "test.pdf", trace=trace, on_progress=on_progress)
        assert len(calls) == TOTAL_STAGES
        # Trace should have recorded stages too
        assert len(trace.stages) >= 5

    def test_ordering(self) -> None:
        fp = _make_fake_pipeline()
        calls = _collect_progress(fp)
        stage_names = [c[0] for c in calls]
        expected = list(STAGE_NAMES)
        assert stage_names == expected


# ── Pipeline result tests ────────────────────────────────────────────


class TestPipelineResult:
    """Verify PipelineResult structure and success/skip behavior."""

    def test_success_result_has_all_fields(self) -> None:
        fp = _make_fake_pipeline()
        result = IngestionPipeline.run(fp, "test.pdf")
        assert result.success is True
        assert result.chunk_count == 3
        assert result.vector_ids == ["v0", "v1", "v2"]
        assert result.error is None
        assert "integrity" in result.stages
        assert "load" in result.stages
        assert "split" in result.stages
        assert "transform" in result.stages
        assert "embed" in result.stages
        assert "upsert" in result.stages

    def test_to_dict_roundtrip(self) -> None:
        fp = _make_fake_pipeline()
        result = IngestionPipeline.run(fp, "test.pdf")
        d = result.to_dict()
        assert d["success"] is True
        assert d["chunk_count"] == 3
        assert isinstance(d["stages"], dict)

    def test_skip_when_already_processed(self) -> None:
        fp = _make_fake_pipeline()
        fp.integrity_checker.should_skip.return_value = True
        result = IngestionPipeline.run(fp, "test.pdf")
        assert result.success is True
        assert result.chunk_count == 0
        assert result.stages["integrity"]["skipped"] is True
        # Loader factory should NOT have been called
        fp.loader_factory.create_for_file.assert_not_called()

    def test_force_bypasses_skip(self) -> None:
        fp = _make_fake_pipeline()
        fp.force = True
        fp.integrity_checker.should_skip.return_value = True
        result = IngestionPipeline.run(fp, "test.pdf")
        assert result.success is True
        assert result.chunk_count == 3
        assert result.stages["integrity"]["skipped"] is False

    def test_failure_returns_error_result(self) -> None:
        fp = _make_fake_pipeline()
        fp._mock_loader.load.side_effect = RuntimeError("load failed")
        result = IngestionPipeline.run(fp, "test.pdf")
        assert result.success is False
        assert "load failed" in result.error

    def test_failure_marks_integrity_failed(self) -> None:
        fp = _make_fake_pipeline()
        fp.chunker.split_document.side_effect = ValueError("split failed")
        IngestionPipeline.run(fp, "test.pdf")
        fp.integrity_checker.mark_failed.assert_called_once()

    def test_success_marks_integrity_success(self) -> None:
        fp = _make_fake_pipeline()
        IngestionPipeline.run(fp, "test.pdf")
        fp.integrity_checker.mark_success.assert_called_once()


# ── Stage orchestration tests ────────────────────────────────────────


class TestPipelineOrchestration:
    """Verify correct wiring of pipeline stages."""

    def test_transforms_called_in_order(self) -> None:
        fp = _make_fake_pipeline()
        call_order = []
        fp.chunk_refiner.transform.side_effect = lambda c, **kw: (
            call_order.append("refiner") or c
        )
        fp.metadata_enricher.transform.side_effect = lambda c, **kw: (
            call_order.append("enricher") or c
        )
        fp.image_captioner.transform.side_effect = lambda c, **kw: (
            call_order.append("captioner") or c
        )
        fp.retrieval_text_generator.transform.side_effect = lambda c, **kw: (
            call_order.append("retrieval_text_generator") or c
        )
        IngestionPipeline.run(fp, "test.pdf")
        assert call_order == ["refiner", "enricher", "captioner", "retrieval_text_generator"]

    def test_bm25_receives_vector_ids(self) -> None:
        """BM25 term_stats should use vector IDs, not chunk IDs."""
        fp = _make_fake_pipeline()
        IngestionPipeline.run(fp, "test.pdf")
        call_args = fp.bm25_indexer.build.call_args
        term_stats = call_args[0][0]
        chunk_ids = [t["chunk_id"] for t in term_stats]
        assert chunk_ids == ["v0", "v1", "v2"]

    def test_bm25_term_stats_structure(self) -> None:
        fp = _make_fake_pipeline()
        IngestionPipeline.run(fp, "test.pdf")
        call_args = fp.bm25_indexer.build.call_args
        term_stats = call_args[0][0]
        for ts in term_stats:
            assert "chunk_id" in ts
            assert "term_frequencies" in ts
            assert "doc_length" in ts

    def test_image_registration_with_existing_files(self, tmp_path) -> None:
        fp = _make_fake_pipeline()
        # Create a temporary image file
        img_path = tmp_path / "img_001.png"
        img_path.write_bytes(b"\x89PNG")
        fp._mock_loader.load.return_value = Document(
            id="doc1",
            text="Hello",
            metadata={
                "source_path": "test.pdf",
                "images": [
                    {"id": "img_001", "path": str(img_path), "page_num": 1},
                ],
            },
        )
        result = IngestionPipeline.run(fp, "test.pdf")
        assert result.image_count == 1
        fp.image_storage.register_image.assert_called_once()

    def test_image_registration_skips_missing_files(self) -> None:
        fp = _make_fake_pipeline()
        fp._mock_loader.load.return_value = Document(
            id="doc1",
            text="Hello",
            metadata={
                "source_path": "test.pdf",
                "images": [
                    {"id": "img_001", "path": "/nonexistent/img.png", "page_num": 1},
                ],
            },
        )
        result = IngestionPipeline.run(fp, "test.pdf")
        assert result.image_count == 0
        fp.image_storage.register_image.assert_not_called()

    def test_stages_have_elapsed_ms(self) -> None:
        fp = _make_fake_pipeline()
        result = IngestionPipeline.run(fp, "test.pdf")
        for stage_name in STAGE_NAMES:
            assert "elapsed_ms" in result.stages[stage_name]
            assert result.stages[stage_name]["elapsed_ms"] >= 0
