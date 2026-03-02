"""Tests for MCP delete_document tool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.ingestion.document_manager import (
    DeleteResult,
    DocumentDetail,
)


class TestDeleteDocumentTool:
    """Tests for the delete_document MCP tool."""

    def _make_mock_doc_manager(self):
        """Create a mock DocumentManager with default integrity behavior.

        By default, integrity.compute_sha256 returns a hash string so
        _resolve_source_hash succeeds. Override in individual tests as needed.
        """
        mgr = MagicMock()
        mgr.integrity.compute_sha256.return_value = "resolved_hash"
        return mgr

    @pytest.mark.asyncio
    async def test_preview_mode_returns_stats(self):
        """confirm_delete_data=False returns preview with chunk/image counts."""
        from src.mcp_server.tools.delete_document import DeleteDocumentTool

        mgr = self._make_mock_doc_manager()
        mgr.get_document_detail.return_value = DocumentDetail(
            source_path="/data/report.pdf",
            source_hash="abc123",
            collection="default",
            chunk_count=42,
            image_count=3,
            chunk_ids=["c1", "c2"],
            image_ids=["i1"],
        )

        tool = DeleteDocumentTool(document_manager=mgr)
        result = await tool.execute(
            source_path="/data/report.pdf",
            collection="default",
            confirm_delete_data=False,
        )

        assert result.isError is False
        response_text = result.content[0].text
        parsed = json.loads(response_text)
        assert parsed["status"] == "confirmation_required"
        assert parsed["associated_data"]["chunks"] == 42
        assert parsed["associated_data"]["images"] == 3
        mgr.delete_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirm_mode_executes_deletion(self):
        """confirm_delete_data=True calls DocumentManager.delete_document."""
        from src.mcp_server.tools.delete_document import DeleteDocumentTool

        mgr = self._make_mock_doc_manager()
        mgr.get_document_detail.return_value = DocumentDetail(
            source_path="/data/report.pdf",
            source_hash="abc123",
            collection="default",
            chunk_count=42,
            image_count=3,
        )
        mgr.delete_document.return_value = DeleteResult(
            success=True,
            chunks_deleted=42,
            bm25_removed=True,
            images_deleted=3,
            integrity_removed=True,
        )

        tool = DeleteDocumentTool(document_manager=mgr)
        result = await tool.execute(
            source_path="/data/report.pdf",
            collection="default",
            confirm_delete_data=True,
        )

        assert result.isError is False
        response_text = result.content[0].text
        parsed = json.loads(response_text)
        assert parsed["status"] == "deleted"
        assert parsed["result"]["chunks_deleted"] == 42
        mgr.delete_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_document_returns_error(self):
        """Requesting preview for unknown document returns not_found."""
        from src.mcp_server.tools.delete_document import DeleteDocumentTool

        mgr = self._make_mock_doc_manager()
        # File doesn't exist on disk → compute_sha256 raises
        mgr.integrity.compute_sha256.side_effect = FileNotFoundError("No such file")
        # No ingestion record either → lookup_by_path returns None
        mgr.integrity.lookup_by_path.return_value = None

        tool = DeleteDocumentTool(document_manager=mgr)
        result = await tool.execute(
            source_path="/nonexistent/file.pdf",
            collection="default",
            confirm_delete_data=False,
        )

        assert result.isError is True
        assert "not found" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_deletion_failure_reported(self):
        """Deletion errors are reported in response."""
        from src.mcp_server.tools.delete_document import DeleteDocumentTool

        mgr = self._make_mock_doc_manager()
        mgr.get_document_detail.return_value = DocumentDetail(
            source_path="/data/report.pdf",
            source_hash="abc123",
            collection="default",
            chunk_count=10,
            image_count=0,
        )
        mgr.delete_document.return_value = DeleteResult(
            success=False,
            chunks_deleted=10,
            errors=["BM25 remove failed: file locked"],
        )

        tool = DeleteDocumentTool(document_manager=mgr)
        result = await tool.execute(
            source_path="/data/report.pdf",
            collection="default",
            confirm_delete_data=True,
        )

        response_text = result.content[0].text
        parsed = json.loads(response_text)
        assert parsed["status"] == "partial_failure"
        assert len(parsed["errors"]) > 0
