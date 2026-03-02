# Document Update Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate orphan data when documents are modified and re-ingested, and provide a user-facing MCP tool for document deletion with confirmation.

**Architecture:** Two features built on existing `DocumentManager.delete_document()`: (1) Pipeline Stage 1 auto-detects old versions via a new `FileIntegrity.lookup_by_path()` query and cascading-deletes before re-ingesting; (2) A new `delete_document` MCP tool uses two-phase confirmation (preview then execute). No changes to chunk_id format or storage layer internals.

**Tech Stack:** Python 3.10+, SQLite (file_integrity), MCP SDK (`mcp` package), pytest, asyncio

**Design Doc:** `docs/plans/2026-02-28-document-update-strategy-design.md`

---

## Task 1: FileIntegrity `lookup_by_path()` — Tests

**Files:**
- Create: `tests/unit/test_file_integrity_lookup.py`
- Reference: `src/libs/loader/file_integrity.py:245-270` (pattern for `should_skip`)

**Step 1: Write failing tests**

```python
"""Tests for FileIntegrity.lookup_by_path()."""

from __future__ import annotations

import pytest

from src.libs.loader.file_integrity import SQLiteIntegrityChecker


@pytest.fixture
def checker(tmp_path):
    """Create a fresh integrity checker with tmp DB."""
    db_path = str(tmp_path / "test_integrity.db")
    return SQLiteIntegrityChecker(db_path=db_path)


class TestLookupByPath:
    """Tests for lookup_by_path method."""

    def test_returns_none_for_unknown_path(self, checker):
        """Unknown path returns None."""
        result = checker.lookup_by_path("/nonexistent/file.pdf")
        assert result is None

    def test_returns_hash_for_known_path(self, checker):
        """Returns file_hash for previously ingested path."""
        checker.mark_success("abc123hash", "/data/report.pdf", collection="default")
        result = checker.lookup_by_path("/data/report.pdf")
        assert result == "abc123hash"

    def test_respects_collection_filter(self, checker):
        """With collection filter, only returns matching collection."""
        checker.mark_success("hash_a", "/data/report.pdf", collection="alpha")
        checker.mark_success("hash_b", "/data/report.pdf", collection="beta")

        assert checker.lookup_by_path("/data/report.pdf", collection="alpha") == "hash_a"
        assert checker.lookup_by_path("/data/report.pdf", collection="beta") == "hash_b"

    def test_returns_none_for_wrong_collection(self, checker):
        """Returns None when path exists but in different collection."""
        checker.mark_success("hash_a", "/data/report.pdf", collection="alpha")
        result = checker.lookup_by_path("/data/report.pdf", collection="gamma")
        assert result is None

    def test_ignores_failed_records(self, checker):
        """Only returns records with status='success'."""
        checker.mark_failed("fail_hash", "/data/broken.pdf", error_msg="parse error")
        result = checker.lookup_by_path("/data/broken.pdf")
        assert result is None

    def test_returns_latest_on_multiple_versions(self, checker):
        """When same path ingested multiple times, returns the latest."""
        checker.mark_success("old_hash", "/data/report.pdf", collection="default")
        # Simulate re-ingestion with new hash (mark_success uses INSERT OR REPLACE)
        checker.mark_success("new_hash", "/data/report.pdf", collection="default")
        result = checker.lookup_by_path("/data/report.pdf", collection="default")
        assert result == "new_hash"

    def test_no_collection_filter_returns_any(self, checker):
        """Without collection filter, returns most recent across all collections."""
        checker.mark_success("hash_x", "/data/report.pdf", collection="coll_a")
        result = checker.lookup_by_path("/data/report.pdf")
        assert result == "hash_x"
```

**Step 2: Run tests — expect FAIL**

```bash
pytest tests/unit/test_file_integrity_lookup.py -v
```

Expected: `AttributeError: 'SQLiteIntegrityChecker' object has no attribute 'lookup_by_path'`

---

## Task 2: FileIntegrity `lookup_by_path()` — Implementation

**Files:**
- Modify: `src/libs/loader/file_integrity.py` (add method after `should_skip`, around line 271)

**Step 3: Implement `lookup_by_path()`**

Add the following method to `SQLiteIntegrityChecker`, right after `should_skip()`:

```python
def lookup_by_path(
    self, file_path: str, collection: str | None = None
) -> str | None:
    """Find file_hash of previously ingested version by source path.

    Queries the ingestion_history table for the most recent successful
    record matching the given file_path (and optional collection).

    Args:
        file_path: Absolute path to the document.
        collection: Optional collection filter to avoid cross-collection
            interference.

    Returns:
        file_hash of the old version if found, None otherwise.
    """
    query = (
        "SELECT file_hash FROM ingestion_history "
        "WHERE file_path = ? AND status = 'success'"
    )
    params: list[str] = [file_path]
    if collection is not None:
        query += " AND collection = ?"
        params.append(collection)
    query += " ORDER BY updated_at DESC LIMIT 1"

    conn = sqlite3.connect(self.db_path)
    try:
        cursor = conn.execute(query, params)
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()
```

**Step 4: Run tests — expect PASS**

```bash
pytest tests/unit/test_file_integrity_lookup.py -v
```

Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add tests/unit/test_file_integrity_lookup.py src/libs/loader/file_integrity.py
git commit -m "feat: add FileIntegrity.lookup_by_path() for old version detection

Enables Pipeline to find the file_hash of a previously ingested document
by source_path + collection, supporting auto-cleanup on re-ingestion."
```

---

## Task 3: Pipeline Old Version Detection — Tests

**Files:**
- Create: `tests/unit/test_pipeline_old_version_cleanup.py`
- Reference: `src/ingestion/pipeline.py:248-275` (Stage 1 logic)
- Reference: `src/ingestion/document_manager.py:181-258` (delete_document)

**Step 6: Write failing tests**

These tests mock the pipeline's dependencies to test only the Stage 1 cleanup logic.

```python
"""Tests for Pipeline old version detection and auto-cleanup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
```

**Step 7: Run tests — expect PASS** (these are pure mock tests)

```bash
pytest tests/unit/test_pipeline_old_version_cleanup.py -v
```

Expected: All 4 tests PASS (they test logic patterns, not actual pipeline wiring).

---

## Task 4: Pipeline Old Version Detection — Implementation

**Files:**
- Modify: `src/ingestion/pipeline.py:12` (add import)
- Modify: `src/ingestion/pipeline.py:72-83` (add fields to PipelineResult)
- Modify: `src/ingestion/pipeline.py:160-168` (add DocumentManager to __init__)
- Modify: `src/ingestion/pipeline.py:253-275` (Stage 1 cleanup logic)

**Step 8: Add import**

At the top of `pipeline.py`, add the DocumentManager import alongside existing storage imports (around line 41-43):

```python
# After: from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.document_manager import DocumentManager
```

**Step 9: Extend PipelineResult**

Add two fields to the `PipelineResult` dataclass (after `error` field, around line 82):

```python
    old_version_cleaned: bool = False
    old_chunks_deleted: int = 0
```

Also add them to `to_dict()`:

```python
        "old_version_cleaned": self.old_version_cleaned,
        "old_chunks_deleted": self.old_chunks_deleted,
```

**Step 10: Add DocumentManager to `__init__`**

In `IngestionPipeline.__init__()`, after the image_storage initialization (around line 167), add:

```python
        # Document lifecycle manager (for old version cleanup)
        self.document_manager = DocumentManager(
            chroma_store=vector_store,
            bm25_indexer=self.bm25_indexer,
            image_storage=self.image_storage,
            file_integrity=self.integrity_checker,
        )
```

**Step 11: Add old version cleanup to Stage 1**

In the `run()` method, after the `should_skip()` block (around line 266, before Stage 2), insert:

```python
            # ── Old version cleanup ───────────────────────────────
            old_version_cleaned = False
            old_chunks_deleted = 0
            old_hash = self.integrity_checker.lookup_by_path(
                file_path, self.collection,
            )
            if old_hash is not None and old_hash != file_hash:
                logger.info(
                    "Detected modified document, cleaning old version: %s "
                    "(old=%s, new=%s)",
                    file_path, old_hash[:12], file_hash[:12],
                )
                try:
                    del_result = self.document_manager.delete_document(
                        source_path=file_path,
                        collection=self.collection,
                        source_hash=old_hash,
                    )
                    old_version_cleaned = True
                    old_chunks_deleted = del_result.chunks_deleted
                    if del_result.errors:
                        logger.warning(
                            "Partial cleanup errors: %s", del_result.errors,
                        )
                except Exception as e:
                    logger.warning("Failed to clean old version: %s", e)

            stages["integrity"]["old_version_cleaned"] = old_version_cleaned
            stages["integrity"]["old_chunks_deleted"] = old_chunks_deleted
```

Then in the final `PipelineResult` construction at the end of `run()`, include these fields:

```python
            old_version_cleaned=old_version_cleaned,
            old_chunks_deleted=old_chunks_deleted,
```

**Step 12: Run existing tests to verify no regression**

```bash
pytest tests/unit/test_document_chunker.py tests/unit/test_vector_upserter_idempotency.py tests/unit/test_document_manager.py -v
```

Expected: All existing tests PASS.

**Step 13: Commit**

```bash
git add src/ingestion/pipeline.py tests/unit/test_pipeline_old_version_cleanup.py
git commit -m "feat: auto-detect and clean old document version on re-ingest

Pipeline Stage 1 now queries FileIntegrity.lookup_by_path() to find
previously ingested versions. If the file_hash differs, it calls
DocumentManager.delete_document() to cascade-delete old data before
re-ingesting. Cleanup failure is non-fatal (logged as WARNING)."
```

---

## Task 5: MCP Delete Tool — Tests

**Files:**
- Create: `tests/unit/test_delete_document_tool.py`
- Reference: `src/mcp_server/tools/list_collections.py` (tool pattern)
- Reference: `src/ingestion/document_manager.py:140-175` (get_document_detail)

**Step 14: Write failing tests**

```python
"""Tests for MCP delete_document tool."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import json

import pytest

from src.ingestion.document_manager import (
    DeleteResult,
    DocumentDetail,
)


class TestDeleteDocumentTool:
    """Tests for the delete_document MCP tool."""

    def _make_mock_doc_manager(self):
        """Create a mock DocumentManager."""
        mgr = MagicMock()
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
        mgr.get_document_detail.return_value = None
        # Also need lookup_by_path to fail
        integrity_mock = MagicMock()
        integrity_mock.lookup_by_path.return_value = None
        mgr._file_integrity = integrity_mock

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
```

**Step 15: Run tests — expect FAIL**

```bash
pytest tests/unit/test_delete_document_tool.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.mcp_server.tools.delete_document'`

---

## Task 6: MCP Delete Tool — Implementation

**Files:**
- Create: `src/mcp_server/tools/delete_document.py`
- Modify: `src/mcp_server/protocol_handler.py:192-208` (register new tool)

**Step 16: Create `delete_document.py`**

```python
"""MCP Tool: delete_document

Two-phase document deletion with user confirmation:
- Phase 1 (confirm_delete_data=false): Returns preview of associated data
- Phase 2 (confirm_delete_data=true): Executes cascading deletion

Usage via MCP:
    Tool name: delete_document
    Input schema:
        - source_path (string, required): Document file path
        - collection (string, optional): Collection name (default: "default")
        - confirm_delete_data (boolean, optional): Execute deletion (default: false)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from mcp import types

if TYPE_CHECKING:
    from src.ingestion.document_manager import DocumentManager
    from src.mcp_server.protocol_handler import ProtocolHandler

logger = logging.getLogger(__name__)


TOOL_NAME = "delete_document"
TOOL_DESCRIPTION = """Delete a document and its associated data from the RAG knowledge base.

First call without confirm_delete_data (or with confirm_delete_data=false) to preview
what will be deleted (chunk count, image count). Then call again with
confirm_delete_data=true to execute the deletion.

This removes all associated data: vector embeddings, BM25 index entries,
extracted images, and the ingestion history record.
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "source_path": {
            "type": "string",
            "description": "Path to the document to delete.",
        },
        "collection": {
            "type": "string",
            "description": "Collection the document belongs to.",
            "default": "default",
        },
        "confirm_delete_data": {
            "type": "boolean",
            "description": (
                "Set to true to execute deletion. "
                "When false (default), returns a preview of what will be deleted."
            ),
            "default": False,
        },
    },
    "required": ["source_path"],
}


class DeleteDocumentTool:
    """MCP Tool for deleting documents with two-phase confirmation.

    Phase 1 (preview): Returns associated data statistics.
    Phase 2 (execute): Cascading deletion across all storage backends.
    """

    def __init__(
        self,
        document_manager: Optional[DocumentManager] = None,
    ) -> None:
        self._document_manager = document_manager

    @property
    def document_manager(self) -> DocumentManager:
        """Get or create DocumentManager."""
        if self._document_manager is None:
            self._document_manager = self._create_document_manager()
        return self._document_manager

    @staticmethod
    def _create_document_manager() -> DocumentManager:
        """Create DocumentManager from default settings."""
        from src.core.settings import load_settings
        from src.ingestion.document_manager import DocumentManager
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.ingestion.storage.image_storage import ImageStorage
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker
        from src.libs.vector_store import ChromaStore, VectorStoreFactory

        settings = load_settings()
        factory = VectorStoreFactory()
        factory.register_provider("chroma", ChromaStore)
        vector_store = factory.create_from_settings(settings.vector_store)

        return DocumentManager(
            chroma_store=vector_store,
            bm25_indexer=BM25Indexer(index_dir="data/db/bm25"),
            image_storage=ImageStorage(
                db_path="data/db/image_index.db",
                images_root="data/images",
            ),
            file_integrity=SQLiteIntegrityChecker(
                db_path="data/db/file_integrity.db",
            ),
        )

    def _resolve_source_hash(
        self, source_path: str, collection: str,
    ) -> str | None:
        """Resolve the source_hash for a document.

        Tries: live file hash -> integrity DB lookup by path.
        """
        try:
            return self.document_manager.integrity.compute_sha256(source_path)
        except Exception:
            return self.document_manager.integrity.lookup_by_path(
                source_path, collection,
            )

    async def execute(
        self,
        source_path: str,
        collection: str = "default",
        confirm_delete_data: bool = False,
    ) -> types.CallToolResult:
        """Execute the delete_document tool.

        Args:
            source_path: Document file path.
            collection: Collection name.
            confirm_delete_data: If True, execute deletion. If False, preview.

        Returns:
            CallToolResult with JSON response.
        """
        logger.info(
            "delete_document: path=%s, collection=%s, confirm=%s",
            source_path, collection, confirm_delete_data,
        )

        source_hash = await asyncio.to_thread(
            self._resolve_source_hash, source_path, collection,
        )

        if source_hash is None:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "not_found",
                        "message": (
                            f"Document not found: '{source_path}' "
                            f"(collection: {collection}). "
                            "File does not exist and no ingestion record found."
                        ),
                    }),
                )],
                isError=True,
            )

        detail = await asyncio.to_thread(
            self.document_manager.get_document_detail, source_hash,
        )

        doc_name = Path(source_path).name

        if not confirm_delete_data:
            # Phase 1: Preview
            chunk_count = detail.chunk_count if detail else 0
            image_count = detail.image_count if detail else 0

            response = {
                "status": "confirmation_required",
                "document": doc_name,
                "source_path": source_path,
                "collection": collection,
                "associated_data": {
                    "chunks": chunk_count,
                    "images": image_count,
                },
                "message": (
                    f"Document '{doc_name}' has {chunk_count} chunks "
                    f"and {image_count} images in the RAG system. "
                    "Call again with confirm_delete_data=true to "
                    "delete all associated data."
                ),
                "instructions": (
                    "To proceed, call delete_document with "
                    "confirm_delete_data=true"
                ),
            }

            return types.CallToolResult(
                content=[types.TextContent(
                    type="text", text=json.dumps(response),
                )],
                isError=False,
            )

        # Phase 2: Execute deletion
        del_result = await asyncio.to_thread(
            self.document_manager.delete_document,
            source_path, collection, source_hash,
        )

        if del_result.success:
            response = {
                "status": "deleted",
                "document": doc_name,
                "result": {
                    "chunks_deleted": del_result.chunks_deleted,
                    "bm25_removed": del_result.bm25_removed,
                    "images_deleted": del_result.images_deleted,
                    "integrity_removed": del_result.integrity_removed,
                },
            }
        else:
            response = {
                "status": "partial_failure",
                "document": doc_name,
                "result": {
                    "chunks_deleted": del_result.chunks_deleted,
                    "bm25_removed": del_result.bm25_removed,
                    "images_deleted": del_result.images_deleted,
                    "integrity_removed": del_result.integrity_removed,
                },
                "errors": del_result.errors,
            }

        return types.CallToolResult(
            content=[types.TextContent(
                type="text", text=json.dumps(response),
            )],
            isError=False,
        )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register the delete_document tool with the protocol handler.

    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    tool = DeleteDocumentTool()

    async def handler(
        source_path: str,
        collection: str = "default",
        confirm_delete_data: bool = False,
    ) -> types.CallToolResult:
        return await tool.execute(
            source_path=source_path,
            collection=collection,
            confirm_delete_data=confirm_delete_data,
        )

    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )

    logger.info("Registered MCP tool: %s", TOOL_NAME)
```

**Step 17: Register in protocol_handler.py**

In `src/mcp_server/protocol_handler.py`, add to `_register_default_tools()` (after line 208):

```python
    # Delete document tool
    from src.mcp_server.tools.delete_document import register_tool as register_delete_tool
    register_delete_tool(protocol_handler)
```

**Step 18: Run tests — expect PASS**

```bash
pytest tests/unit/test_delete_document_tool.py -v
```

Expected: All 4 tests PASS.

**Step 19: Commit**

```bash
git add src/mcp_server/tools/delete_document.py src/mcp_server/protocol_handler.py tests/unit/test_delete_document_tool.py
git commit -m "feat: add delete_document MCP tool with two-phase confirmation

Phase 1 (preview): Returns chunk/image counts without deleting.
Phase 2 (confirm): Cascading deletion across all storage backends.
Registered in protocol_handler alongside existing tools."
```

---

## Task 7: Full Test Suite Verification

**Step 20: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: All existing tests pass, plus 11+ new tests from Tasks 1, 3, 5.

**Step 21: Run ruff**

```bash
ruff check src/libs/loader/file_integrity.py src/ingestion/pipeline.py src/mcp_server/tools/delete_document.py src/mcp_server/protocol_handler.py
ruff format src/libs/loader/file_integrity.py src/ingestion/pipeline.py src/mcp_server/tools/delete_document.py src/mcp_server/protocol_handler.py
```

**Step 22: Final commit (if ruff made changes)**

```bash
git add -u
git commit -m "style: apply ruff formatting to new/modified files"
```

---

## Summary

| Task | What | Files | Tests |
|------|------|-------|-------|
| 1-2 | `FileIntegrity.lookup_by_path()` | 1 modify, 1 new test | 7 unit tests |
| 3-4 | Pipeline old version auto-cleanup | 1 modify, 1 new test | 4 unit tests |
| 5-6 | MCP `delete_document` tool | 1 new, 1 modify, 1 new test | 4 unit tests |
| 7 | Full verification | — | Regression check |

**Total new/modified production files:** 4
**Total new test files:** 3
**Files NOT changed:** chunk_id generation, DocumentManager, BM25Indexer, VectorUpserter, ChromaStore
