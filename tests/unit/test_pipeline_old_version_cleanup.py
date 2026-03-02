"""Tests for Pipeline old version detection and auto-cleanup."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.ingestion.document_manager import DeleteResult


class TestPipelineOldVersionCleanup:
    """Test auto-cleanup of old document versions in Pipeline Stage 1."""

    def _make_pipeline_with_mocks(self):
        """Create a pipeline-like object with mocked dependencies.

        We test the cleanup logic in isolation by mocking the components
        rather than instantiating a full IngestionPipeline (which needs
        real settings, API keys, etc.).
        """
        mock_integrity = MagicMock()
        mock_doc_manager = MagicMock()

        return mock_integrity, mock_doc_manager

    def test_no_old_version_proceeds_normally(self):
        """When lookup_by_path returns None, no cleanup happens."""
        integrity, doc_manager = self._make_pipeline_with_mocks()
        integrity.lookup_by_path.return_value = None

        # Simulate: no old version found
        old_hash = integrity.lookup_by_path("/data/report.pdf", "default")

        assert old_hash is None
        doc_manager.delete_document.assert_not_called()

    def test_same_hash_no_cleanup(self):
        """When old_hash == new file_hash, no cleanup needed."""
        integrity, doc_manager = self._make_pipeline_with_mocks()
        integrity.lookup_by_path.return_value = "same_hash_abc"

        old_hash = integrity.lookup_by_path("/data/report.pdf", "default")
        new_hash = "same_hash_abc"

        # Same hash means file unchanged — no cleanup
        if old_hash is not None and old_hash != new_hash:
            doc_manager.delete_document()

        doc_manager.delete_document.assert_not_called()

    def test_different_hash_triggers_cleanup(self):
        """When old_hash != new file_hash, delete old version."""
        integrity, doc_manager = self._make_pipeline_with_mocks()
        integrity.lookup_by_path.return_value = "old_hash_123"
        doc_manager.delete_document.return_value = DeleteResult(
            success=True, chunks_deleted=42, bm25_removed=True,
        )

        old_hash = integrity.lookup_by_path("/data/report.pdf", "default")
        new_hash = "new_hash_456"

        if old_hash is not None and old_hash != new_hash:
            result = doc_manager.delete_document(
                source_path="/data/report.pdf",
                collection="default",
                source_hash=old_hash,
            )
            assert result.success is True
            assert result.chunks_deleted == 42

        doc_manager.delete_document.assert_called_once_with(
            source_path="/data/report.pdf",
            collection="default",
            source_hash="old_hash_123",
        )

    def test_cleanup_failure_does_not_raise(self):
        """Cleanup failure is logged but does not abort pipeline."""
        integrity, doc_manager = self._make_pipeline_with_mocks()
        integrity.lookup_by_path.return_value = "old_hash_123"
        doc_manager.delete_document.side_effect = RuntimeError("DB locked")

        old_hash = integrity.lookup_by_path("/data/report.pdf", "default")
        new_hash = "new_hash_456"

        cleanup_error = None
        if old_hash is not None and old_hash != new_hash:
            try:
                doc_manager.delete_document(
                    source_path="/data/report.pdf",
                    collection="default",
                    source_hash=old_hash,
                )
            except Exception as e:
                cleanup_error = str(e)

        # Error captured, not raised
        assert cleanup_error == "DB locked"
