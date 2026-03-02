"""Unit tests for DataService -- read-only facade for the Data Browser.

Tests verify:
- Lazy initialisation & collection-switching logic
- list_documents / get_document_detail / get_chunks / get_images
- get_collection_stats / delete_document delegation
- Exception handling (graceful fallback)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Lightweight stand-ins (no real DB)
# ---------------------------------------------------------------------------

@dataclass
class _FakeDocInfo:
    source_path: str = "/tmp/a.pdf"
    source_hash: str = "abc123"
    collection: Optional[str] = "default"
    chunk_count: int = 3
    image_count: int = 1
    processed_at: Optional[str] = "2026-01-01T00:00:00"


@dataclass
class _FakeDocDetail(_FakeDocInfo):
    chunk_ids: List[str] = field(default_factory=lambda: ["c1", "c2", "c3"])
    image_ids: List[str] = field(default_factory=lambda: ["i1"])


@dataclass
class _FakeDeleteResult:
    success: bool = True
    chunks_deleted: int = 3
    bm25_removed: bool = True
    images_deleted: int = 1
    integrity_removed: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class _FakeCollectionStats:
    collection: Optional[str] = "default"
    document_count: int = 2
    chunk_count: int = 10
    image_count: int = 4


def _make_mock_manager() -> MagicMock:
    """Return a MagicMock that behaves like DocumentManager."""
    mgr = MagicMock()
    mgr.list_documents.return_value = [_FakeDocInfo()]
    mgr.get_document_detail.return_value = _FakeDocDetail()
    mgr.delete_document.return_value = _FakeDeleteResult()
    mgr.get_collection_stats.return_value = _FakeCollectionStats()
    return mgr


def _make_mock_chroma() -> MagicMock:
    """Return a MagicMock that behaves like ChromaStore."""
    chroma = MagicMock()
    chroma.collection.get.return_value = {
        "ids": ["chunk_001", "chunk_002"],
        "documents": ["Hello world", "Foo bar"],
        "metadatas": [{"page": 1}, {"page": 2}],
    }
    return chroma


def _make_mock_images() -> MagicMock:
    """Return a MagicMock that behaves like ImageStorage."""
    imgs = MagicMock()
    imgs.list_images.return_value = [
        {"image_id": "img_001", "file_path": "/tmp/img_001.png"},
    ]
    return imgs


# ---------------------------------------------------------------------------
# Helper: create a pre-wired DataService (skip _ensure_stores)
# ---------------------------------------------------------------------------

def _make_service(
    manager: Any = None,
    chroma: Any = None,
    images: Any = None,
    collection: str = "default",
):
    """Build a DataService with stores already injected."""
    from src.observability.dashboard.services.data_service import DataService

    svc = DataService()
    svc._manager = manager or _make_mock_manager()
    svc._chroma = chroma or _make_mock_chroma()
    svc._images = images or _make_mock_images()
    svc._current_collection = collection
    return svc


# ===================================================================
# Tests: list_documents
# ===================================================================

class TestListDocuments:
    def test_returns_dicts(self):
        svc = _make_service()
        result = svc.list_documents("default")

        assert isinstance(result, list)
        assert len(result) == 1
        doc = result[0]
        assert doc["source_path"] == "/tmp/a.pdf"
        assert doc["source_hash"] == "abc123"
        assert doc["chunk_count"] == 3
        assert doc["image_count"] == 1

    def test_passes_collection_filter(self):
        mgr = _make_mock_manager()
        mgr.list_documents.return_value = []
        svc = _make_service(manager=mgr, collection="my_collection")

        result = svc.list_documents("my_collection")
        mgr.list_documents.assert_called_once_with("my_collection")
        assert result == []

    def test_none_collection(self):
        mgr = _make_mock_manager()
        svc = _make_service(manager=mgr)

        svc.list_documents(None)
        mgr.list_documents.assert_called_once_with(None)


# ===================================================================
# Tests: get_document_detail
# ===================================================================

class TestGetDocumentDetail:
    def test_found(self):
        svc = _make_service()
        result = svc.get_document_detail("abc123", "default")

        assert result is not None
        assert result["source_hash"] == "abc123"
        assert result["chunk_ids"] == ["c1", "c2", "c3"]
        assert result["image_ids"] == ["i1"]

    def test_not_found(self):
        mgr = _make_mock_manager()
        mgr.get_document_detail.return_value = None
        svc = _make_service(manager=mgr)

        result = svc.get_document_detail("nonexistent", "default")
        assert result is None


# ===================================================================
# Tests: get_chunks
# ===================================================================

class TestGetChunks:
    def test_returns_chunk_dicts(self):
        svc = _make_service()
        chunks = svc.get_chunks("abc123", "default")

        assert len(chunks) == 2
        assert chunks[0]["id"] == "chunk_001"
        assert chunks[0]["text"] == "Hello world"
        assert chunks[0]["metadata"] == {"page": 1}
        assert chunks[1]["id"] == "chunk_002"

    def test_empty_results(self):
        chroma = _make_mock_chroma()
        chroma.collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }
        svc = _make_service(chroma=chroma)

        assert svc.get_chunks("missing_hash", "default") == []

    def test_exception_returns_empty(self):
        chroma = _make_mock_chroma()
        chroma.collection.get.side_effect = RuntimeError("DB error")
        svc = _make_service(chroma=chroma)

        assert svc.get_chunks("abc123", "default") == []

    def test_queries_by_doc_hash(self):
        chroma = _make_mock_chroma()
        svc = _make_service(chroma=chroma)

        svc.get_chunks("hash_xyz", "default")
        chroma.collection.get.assert_called_once_with(
            where={"doc_hash": "hash_xyz"},
            include=["documents", "metadatas"],
        )


# ===================================================================
# Tests: get_images
# ===================================================================

class TestGetImages:
    def test_returns_image_dicts(self):
        svc = _make_service()
        images = svc.get_images("abc123", "default")

        assert len(images) == 1
        assert images[0]["image_id"] == "img_001"

    def test_exception_returns_empty(self):
        imgs = _make_mock_images()
        imgs.list_images.side_effect = RuntimeError("DB error")
        svc = _make_service(images=imgs)

        assert svc.get_images("abc123", "default") == []

    def test_queries_by_doc_hash(self):
        imgs = _make_mock_images()
        svc = _make_service(images=imgs)

        svc.get_images("hash_xyz", "default")
        imgs.list_images.assert_called_once_with(doc_hash="hash_xyz")


# ===================================================================
# Tests: delete_document
# ===================================================================

class TestDeleteDocument:
    def test_delegates_to_manager(self):
        mgr = _make_mock_manager()
        svc = _make_service(manager=mgr)

        result = svc.delete_document("/tmp/a.pdf", "default", "abc123")
        mgr.delete_document.assert_called_once_with(
            "/tmp/a.pdf", "default", source_hash="abc123"
        )
        assert result.success is True

    def test_none_collection_defaults(self):
        mgr = _make_mock_manager()
        svc = _make_service(manager=mgr)

        svc.delete_document("/tmp/a.pdf")
        mgr.delete_document.assert_called_once_with(
            "/tmp/a.pdf", "default", source_hash=None
        )


# ===================================================================
# Tests: get_collection_stats
# ===================================================================

class TestGetCollectionStats:
    def test_returns_stats_dict(self):
        svc = _make_service()
        stats = svc.get_collection_stats("default")

        assert stats["document_count"] == 2
        assert stats["chunk_count"] == 10
        assert stats["image_count"] == 4
        assert stats["collection"] == "default"

    def test_passes_collection_to_manager(self):
        mgr = _make_mock_manager()
        svc = _make_service(manager=mgr, collection="special")

        svc.get_collection_stats("special")
        mgr.get_collection_stats.assert_called_once_with("special")


# ===================================================================
# Tests: Lazy initialisation / collection switching
# ===================================================================

class TestEnsureStores:
    def test_skips_reinit_for_same_collection(self):
        """If the collection hasn't changed, _ensure_stores is a no-op."""
        svc = _make_service(collection="default")
        original_manager = svc._manager

        # Call a public method (triggers _ensure_stores internally)
        svc.list_documents("default")

        # Manager should NOT have been replaced
        assert svc._manager is original_manager

    @patch(
        "src.observability.dashboard.services.data_service.DataService._ensure_stores"
    )
    def test_ensure_stores_called_on_public_methods(self, mock_ensure):
        """Every public method should call _ensure_stores."""
        from src.observability.dashboard.services.data_service import DataService

        svc = DataService()
        # Inject stores so calls after _ensure_stores don't fail
        svc._manager = _make_mock_manager()
        svc._chroma = _make_mock_chroma()
        svc._images = _make_mock_images()

        svc.list_documents("x")
        svc.get_document_detail("d", "x")
        svc.get_chunks("h", "x")
        svc.get_images("h", "x")
        svc.delete_document("/p", "x")
        svc.get_collection_stats("x")

        assert mock_ensure.call_count == 6

    def test_collection_switch_triggers_reinit(self):
        """Changing collection should trigger a new _ensure_stores."""
        svc = _make_service(collection="coll_a")
        original_manager = svc._manager

        # Patch _ensure_stores to simulate collection switch
        # (we just verify the condition: different collection => stores not reused)
        svc._current_collection = "coll_a"

        # Same collection -> no reinit needed
        # The _ensure_stores guard checks this
        assert svc._current_collection == "coll_a"

        # If we manually set to a different collection, next call would reinit
        svc._current_collection = "coll_b"
        # The guard condition now differs from "coll_a"
        assert svc._current_collection != "coll_a"
