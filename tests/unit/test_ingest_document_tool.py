"""Tests for ingest_document MCP tool — handler, schema, registration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_server.tools.ingest_document import (
    TOOL_INPUT_SCHEMA,
    TOOL_NAME,
    ingest_document_handler,
    register_tool,
)


# ── Schema tests ───────────────────────────────────────────────────


class TestIngestDocumentSchema:
    """Tool metadata and JSON schema validation."""

    def test_tool_name(self):
        assert TOOL_NAME == "ingest_document"

    def test_schema_has_file_path_required(self):
        assert "file_path" in TOOL_INPUT_SCHEMA["properties"]
        assert "file_path" in TOOL_INPUT_SCHEMA["required"]

    def test_schema_has_collection_optional(self):
        assert "collection" in TOOL_INPUT_SCHEMA["properties"]
        assert "collection" not in TOOL_INPUT_SCHEMA["required"]

    def test_collection_default(self):
        assert TOOL_INPUT_SCHEMA["properties"]["collection"]["default"] == "default"


# ── Handler tests ──────────────────────────────────────────────────


class TestIngestDocumentHandler:
    """Handler success and error paths."""

    @pytest.mark.asyncio
    async def test_file_not_found_returns_error(self, tmp_path):
        """Non-existent file returns isError=True."""
        result = await ingest_document_handler(
            file_path=str(tmp_path / "nonexistent.pdf"),
        )
        assert result.isError is True
        assert "not found" in result.content[0].text.lower() or \
               "不存在" in result.content[0].text

    @pytest.mark.asyncio
    async def test_success_returns_doc_info(self, tmp_path):
        """Successful ingestion returns doc_id, chunk_count, collection."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.doc_id = "doc_abc123"
        mock_result.chunk_count = 42

        with patch(
            "src.mcp_server.tools.ingest_document._run_pipeline",
            return_value=mock_result,
        ):
            result = await ingest_document_handler(
                file_path=str(test_file),
                collection="test_col",
            )

        assert result.isError is False
        text = result.content[0].text
        assert "doc_abc123" in text
        assert "42" in text
        assert "test_col" in text

    @pytest.mark.asyncio
    async def test_pipeline_failure_returns_error(self, tmp_path):
        """Pipeline exception returns isError=True with message."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy")

        with patch(
            "src.mcp_server.tools.ingest_document._run_pipeline",
            side_effect=RuntimeError("Pipeline exploded"),
        ):
            result = await ingest_document_handler(file_path=str(test_file))

        assert result.isError is True
        assert "Pipeline exploded" in result.content[0].text

    @pytest.mark.asyncio
    async def test_pipeline_result_not_success(self, tmp_path):
        """Pipeline returning success=False is reported as error."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy")

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Parsing failed"

        with patch(
            "src.mcp_server.tools.ingest_document._run_pipeline",
            return_value=mock_result,
        ):
            result = await ingest_document_handler(file_path=str(test_file))

        assert result.isError is True
        assert "Parsing failed" in result.content[0].text

    @pytest.mark.asyncio
    async def test_default_collection(self, tmp_path):
        """When collection not specified, defaults to 'default'."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.doc_id = "doc_x"
        mock_result.chunk_count = 1

        with patch(
            "src.mcp_server.tools.ingest_document._run_pipeline",
            return_value=mock_result,
        ) as mock_run:
            await ingest_document_handler(file_path=str(test_file))
            # Verify pipeline was called with default collection
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs[1]["collection"] == "default" or \
                   call_kwargs[0][1] == "default"


# ── Registration test ──────────────────────────────────────────────


class TestRegisterTool:
    """register_tool integrates with ProtocolHandler."""

    def test_register_tool_calls_handler(self):
        mock_handler = MagicMock()
        register_tool(mock_handler)
        mock_handler.register_tool.assert_called_once()
        call_kwargs = mock_handler.register_tool.call_args
        assert call_kwargs.kwargs["name"] == "ingest_document" or \
               call_kwargs[1]["name"] == "ingest_document"
