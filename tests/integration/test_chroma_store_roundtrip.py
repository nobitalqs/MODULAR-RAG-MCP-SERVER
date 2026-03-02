"""Integration tests for ChromaStore roundtrip operations.

Tests upsert, query, delete, clear, get_by_ids with real ChromaDB instances.
Each test uses a temporary directory to ensure isolation.
"""

from __future__ import annotations

import pytest

from src.libs.vector_store.chroma_store import ChromaStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class TestChromaStoreFactory:
    """Test factory creation and registration."""

    def test_factory_can_create(self, tmp_path):
        """Factory can register and create ChromaStore instances."""
        factory = VectorStoreFactory()
        factory.register_provider("chroma", ChromaStore)

        store = factory.create(
            provider="chroma",
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        assert isinstance(store, ChromaStore)


class TestChromaStoreUpsert:
    """Test upsert operations."""

    def test_upsert_single_record(self, tmp_path):
        """Upsert a single record and query returns it."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        record = {
            "id": "doc1",
            "vector": [1.0, 0.0, 0.0],
            "text": "Hello world",
            "metadata": {"source": "test"},
        }

        store.upsert([record])

        results = store.query(vector=[1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["text"] == "Hello world"
        assert results[0]["metadata"]["source"] == "test"
        # Score should be close to 1.0 (identical vectors)
        assert results[0]["score"] > 0.99

    def test_upsert_multiple_records(self, tmp_path):
        """Upsert multiple records with different vectors."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        records = [
            {
                "id": "doc1",
                "vector": [1.0, 0.0, 0.0],
                "text": "First document",
                "metadata": {"index": 1},
            },
            {
                "id": "doc2",
                "vector": [0.0, 1.0, 0.0],
                "text": "Second document",
                "metadata": {"index": 2},
            },
            {
                "id": "doc3",
                "vector": [0.0, 0.0, 1.0],
                "text": "Third document",
                "metadata": {"index": 3},
            },
        ]

        store.upsert(records)

        # Query with first vector should return doc1 first
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        assert results[0]["id"] == "doc1"

    def test_upsert_idempotent(self, tmp_path):
        """Upserting the same ID twice updates the record."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        # First upsert
        record1 = {
            "id": "doc1",
            "vector": [1.0, 0.0, 0.0],
            "text": "Original text",
            "metadata": {"version": 1},
        }
        store.upsert([record1])

        # Second upsert with same ID
        record2 = {
            "id": "doc1",
            "vector": [0.0, 1.0, 0.0],
            "text": "Updated text",
            "metadata": {"version": 2},
        }
        store.upsert([record2])

        # Query should return updated version
        results = store.query(vector=[0.0, 1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["text"] == "Updated text"
        assert results[0]["metadata"]["version"] == 2


class TestChromaStoreQuery:
    """Test query operations."""

    def test_query_returns_sorted_by_similarity(self, tmp_path):
        """Query returns results sorted by similarity score."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        records = [
            {
                "id": "doc1",
                "vector": [1.0, 0.0, 0.0],  # Will be closest
                "text": "Document 1",
            },
            {
                "id": "doc2",
                "vector": [0.707, 0.707, 0.0],  # 45 degrees away
                "text": "Document 2",
            },
            {
                "id": "doc3",
                "vector": [0.0, 1.0, 0.0],  # 90 degrees away
                "text": "Document 3",
            },
        ]

        store.upsert(records)

        # Query with [1.0, 0.0, 0.0] should return doc1, doc2, doc3 in that order
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        assert results[0]["id"] == "doc1"
        assert results[1]["id"] == "doc2"
        assert results[2]["id"] == "doc3"
        # Verify scores are descending
        assert results[0]["score"] > results[1]["score"]
        assert results[1]["score"] > results[2]["score"]

    def test_query_respects_top_k(self, tmp_path):
        """Query respects top_k parameter."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        records = [
            {"id": f"doc{i}", "vector": [float(i), 0.0, 0.0], "text": f"Doc {i}"}
            for i in range(5)
        ]

        store.upsert(records)

        # Query with top_k=2 should return only 2 results
        results = store.query(vector=[4.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_query_with_metadata_filters(self, tmp_path):
        """Query with metadata filters returns only matching records."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        records = [
            {
                "id": "doc1",
                "vector": [1.0, 0.0, 0.0],
                "text": "Document 1",
                "metadata": {"source": "A"},
            },
            {
                "id": "doc2",
                "vector": [1.0, 0.0, 0.0],
                "text": "Document 2",
                "metadata": {"source": "B"},
            },
            {
                "id": "doc3",
                "vector": [1.0, 0.0, 0.0],
                "text": "Document 3",
                "metadata": {"source": "A"},
            },
        ]

        store.upsert(records)

        # Query with filter for source=A
        results = store.query(
            vector=[1.0, 0.0, 0.0],
            top_k=10,
            filters={"source": "A"},
        )

        assert len(results) == 2
        assert all(r["metadata"]["source"] == "A" for r in results)


class TestChromaStoreDelete:
    """Test delete operations."""

    def test_delete_records(self, tmp_path):
        """Delete removes records from the store."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        records = [
            {"id": "doc1", "vector": [1.0, 0.0, 0.0], "text": "Document 1"},
            {"id": "doc2", "vector": [0.0, 1.0, 0.0], "text": "Document 2"},
            {"id": "doc3", "vector": [0.0, 0.0, 1.0], "text": "Document 3"},
        ]

        store.upsert(records)

        # Delete doc2
        store.delete(["doc2"])

        # Query should only return doc1 and doc3
        results = store.query(vector=[0.0, 1.0, 0.0], top_k=10)
        assert len(results) == 2
        assert all(r["id"] != "doc2" for r in results)


class TestChromaStoreClear:
    """Test clear operations."""

    def test_clear_collection(self, tmp_path):
        """Clear removes all records from the collection."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        records = [
            {"id": "doc1", "vector": [1.0, 0.0, 0.0], "text": "Document 1"},
            {"id": "doc2", "vector": [0.0, 1.0, 0.0], "text": "Document 2"},
        ]

        store.upsert(records)

        # Clear the collection
        store.clear()

        # Query should return no results
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=10)
        assert len(results) == 0


class TestChromaStoreGetByIds:
    """Test get_by_ids operations."""

    def test_get_by_ids(self, tmp_path):
        """get_by_ids returns records in the correct order."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        records = [
            {"id": "doc1", "vector": [1.0, 0.0, 0.0], "text": "Document 1"},
            {"id": "doc2", "vector": [0.0, 1.0, 0.0], "text": "Document 2"},
            {"id": "doc3", "vector": [0.0, 0.0, 1.0], "text": "Document 3"},
        ]

        store.upsert(records)

        # Get records by IDs in specific order
        results = store.get_by_ids(["doc3", "doc1"])
        assert len(results) == 2
        assert results[0]["id"] == "doc3"
        assert results[0]["text"] == "Document 3"
        assert results[1]["id"] == "doc1"
        assert results[1]["text"] == "Document 1"


class TestChromaStoreValidation:
    """Test validation and error handling."""

    def test_validates_records(self, tmp_path):
        """Upsert validates records and raises ValueError for empty list."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        with pytest.raises(ValueError, match="Records list cannot be empty"):
            store.upsert([])

    def test_validates_query_vector(self, tmp_path):
        """Query validates vector and raises ValueError for empty vector."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        with pytest.raises(ValueError, match="Query vector cannot be empty"):
            store.query(vector=[], top_k=1)


class TestChromaStoreMetadataSanitization:
    """Test metadata sanitization."""

    def test_sanitizes_metadata(self, tmp_path):
        """ChromaStore sanitizes metadata to supported types."""
        store = ChromaStore(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        record = {
            "id": "doc1",
            "vector": [1.0, 0.0, 0.0],
            "text": "Test document",
            "metadata": {
                "string_field": "value",
                "int_field": 42,
                "float_field": 3.14,
                "bool_field": True,
                "list_field": ["a", "b", "c"],
                "none_field": None,
                "dict_field": {"nested": "value"},
            },
        }

        store.upsert([record])

        results = store.query(vector=[1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        metadata = results[0]["metadata"]

        # Supported types should be preserved
        assert metadata["string_field"] == "value"
        assert metadata["int_field"] == 42
        assert metadata["float_field"] == 3.14
        assert metadata["bool_field"] is True

        # Lists should be converted to comma-separated strings
        assert metadata["list_field"] == "a,b,c"

        # None and dict should be skipped
        assert "none_field" not in metadata
        assert "dict_field" not in metadata
