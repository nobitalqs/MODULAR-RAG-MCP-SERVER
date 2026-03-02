"""Tests for VectorStore abstraction: BaseVectorStore, VectorStoreFactory.

Contract tests that verify record shape (id + vector), query params,
and optional lifecycle methods.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.libs.vector_store.base_vector_store import BaseVectorStore


# ---------------------------------------------------------------------------
# Fake provider for testing
# ---------------------------------------------------------------------------

class FakeVectorStore(BaseVectorStore):
    """In-memory VectorStore stub for contract testing."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs
        self.storage: dict[str, dict[str, Any]] = {}
        self.upsert_count = 0
        self.query_count = 0

    def upsert(
        self,
        records: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> None:
        self.validate_records(records)
        self.upsert_count += 1
        for record in records:
            self.storage[record["id"]] = record

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.validate_query_vector(vector, top_k)
        self.query_count += 1
        results: list[dict[str, Any]] = []
        for i, (rid, record) in enumerate(self.storage.items()):
            if i >= top_k:
                break
            if filters:
                meta = record.get("metadata", {})
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue
            results.append({
                "id": rid,
                "score": 1.0 - (i * 0.1),
                "metadata": record.get("metadata", {}),
            })
        return results


# ===========================================================================
# BaseVectorStore.validate_records
# ===========================================================================

class TestValidateRecords:
    """Tests for record validation."""

    def setup_method(self) -> None:
        self.store = FakeVectorStore()

    def test_valid_records(self) -> None:
        self.store.validate_records([
            {"id": "d1", "vector": [0.1, 0.2]},
            {"id": "d2", "vector": [0.3], "metadata": {"src": "a.pdf"}},
        ])

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self.store.validate_records([])

    def test_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="not a dict"):
            self.store.validate_records([{"id": "ok", "vector": [0.1]}, "bad"])  # type: ignore[list-item]

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required field: 'id'"):
            self.store.validate_records([{"vector": [0.1]}])

    def test_missing_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required field: 'vector'"):
            self.store.validate_records([{"id": "d1"}])

    def test_invalid_vector_type_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid vector type"):
            self.store.validate_records([{"id": "d1", "vector": "bad"}])

    def test_empty_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="empty vector"):
            self.store.validate_records([{"id": "d1", "vector": []}])

    def test_tuple_vector_accepted(self) -> None:
        self.store.validate_records([{"id": "d1", "vector": (0.1, 0.2)}])


# ===========================================================================
# BaseVectorStore.validate_query_vector
# ===========================================================================

class TestValidateQueryVector:
    """Tests for query vector validation."""

    def setup_method(self) -> None:
        self.store = FakeVectorStore()

    def test_valid_query(self) -> None:
        self.store.validate_query_vector([0.1, 0.2], top_k=5)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a list or tuple"):
            self.store.validate_query_vector("bad", top_k=5)  # type: ignore[arg-type]

    def test_empty_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self.store.validate_query_vector([], top_k=5)

    def test_zero_top_k_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a positive integer"):
            self.store.validate_query_vector([0.1], top_k=0)

    def test_negative_top_k_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a positive integer"):
            self.store.validate_query_vector([0.1], top_k=-3)


# ===========================================================================
# Optional lifecycle methods
# ===========================================================================

class TestOptionalMethods:
    """Tests for optional lifecycle methods (default NotImplementedError)."""

    def setup_method(self) -> None:
        self.store = FakeVectorStore()

    def test_delete_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="does not implement delete"):
            self.store.delete(["d1"])

    def test_clear_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="does not implement clear"):
            self.store.clear()

    def test_get_by_ids_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="does not implement get_by_ids"):
            self.store.get_by_ids(["d1"])


# ===========================================================================
# FakeVectorStore behavior (contract tests)
# ===========================================================================

class TestFakeVectorStore:
    """Contract tests for FakeVectorStore upsert/query."""

    def test_upsert_single(self) -> None:
        store = FakeVectorStore()
        store.upsert([{"id": "d1", "vector": [0.1, 0.2]}])
        assert store.upsert_count == 1
        assert "d1" in store.storage

    def test_upsert_multiple(self) -> None:
        store = FakeVectorStore()
        store.upsert([
            {"id": "d1", "vector": [0.1]},
            {"id": "d2", "vector": [0.2], "metadata": {"src": "t.pdf"}},
        ])
        assert len(store.storage) == 2
        assert store.storage["d2"]["metadata"]["src"] == "t.pdf"

    def test_upsert_idempotent(self) -> None:
        store = FakeVectorStore()
        rec = [{"id": "d1", "vector": [0.1]}]
        store.upsert(rec)
        store.upsert(rec)
        assert len(store.storage) == 1
        assert store.upsert_count == 2

    def test_upsert_validates_input(self) -> None:
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="cannot be empty"):
            store.upsert([])

    def test_query_returns_results(self) -> None:
        store = FakeVectorStore()
        store.upsert([
            {"id": "d1", "vector": [0.1]},
            {"id": "d2", "vector": [0.2]},
        ])
        results = store.query(vector=[0.5], top_k=10)
        assert len(results) == 2
        assert results[0]["id"] == "d1"
        assert results[0]["score"] == 1.0

    def test_query_respects_top_k(self) -> None:
        store = FakeVectorStore()
        store.upsert([{"id": f"d{i}", "vector": [float(i)]} for i in range(10)])
        results = store.query(vector=[0.0], top_k=3)
        assert len(results) == 3

    def test_query_with_filters(self) -> None:
        store = FakeVectorStore()
        store.upsert([
            {"id": "d1", "vector": [0.1], "metadata": {"src": "a.pdf"}},
            {"id": "d2", "vector": [0.2], "metadata": {"src": "b.pdf"}},
            {"id": "d3", "vector": [0.3], "metadata": {"src": "a.pdf"}},
        ])
        results = store.query(vector=[0.0], top_k=10, filters={"src": "a.pdf"})
        ids = [r["id"] for r in results]
        assert "d1" in ids
        assert "d3" in ids
        assert "d2" not in ids

    def test_query_validates_input(self) -> None:
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="cannot be empty"):
            store.query(vector=[], top_k=5)

    def test_query_increments_count(self) -> None:
        store = FakeVectorStore()
        store.query(vector=[0.1], top_k=5)
        store.query(vector=[0.2], top_k=5)
        assert store.query_count == 2


# ===========================================================================
# VectorStoreFactory
# ===========================================================================

class TestVectorStoreFactory:
    """Tests for the VectorStore factory routing logic."""

    def setup_method(self) -> None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        self.factory = VectorStoreFactory()

    def test_register_and_create(self) -> None:
        self.factory.register_provider("fake", FakeVectorStore)
        store = self.factory.create("fake")
        assert isinstance(store, FakeVectorStore)

    def test_case_insensitive_registration(self) -> None:
        self.factory.register_provider("Chroma", FakeVectorStore)
        store = self.factory.create("chroma")
        assert isinstance(store, FakeVectorStore)

    def test_case_insensitive_create(self) -> None:
        self.factory.register_provider("fake", FakeVectorStore)
        store = self.factory.create("FAKE")
        assert isinstance(store, FakeVectorStore)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown VectorStore provider"):
            self.factory.create("nonexistent")

    def test_error_lists_available_providers(self) -> None:
        self.factory.register_provider("chroma", FakeVectorStore)
        self.factory.register_provider("qdrant", FakeVectorStore)
        with pytest.raises(ValueError, match="chroma, qdrant"):
            self.factory.create("missing")

    def test_register_non_subclass_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a subclass"):
            self.factory.register_provider("bad", dict)  # type: ignore[arg-type]

    def test_list_providers(self) -> None:
        self.factory.register_provider("qdrant", FakeVectorStore)
        self.factory.register_provider("chroma", FakeVectorStore)
        assert self.factory.list_providers() == ["chroma", "qdrant"]

    def test_create_with_kwargs(self) -> None:
        self.factory.register_provider("fake", FakeVectorStore)
        store = self.factory.create("fake", persist_directory="./data")
        assert store.config["persist_directory"] == "./data"

    def test_create_from_settings(self) -> None:
        from src.core.settings import VectorStoreSettings

        self.factory.register_provider("fake", FakeVectorStore)

        settings = VectorStoreSettings(
            provider="fake",
            persist_directory="./data/db/chroma",
            collection_name="knowledge_hub",
        )
        store = self.factory.create_from_settings(settings)
        assert isinstance(store, FakeVectorStore)

    def test_create_from_settings_forwards_fields(self) -> None:
        from src.core.settings import VectorStoreSettings

        self.factory.register_provider("fake", FakeVectorStore)

        settings = VectorStoreSettings(
            provider="fake",
            persist_directory="./data/db/chroma",
            collection_name="kb",
            host="localhost",
            port=8000,
        )
        store = self.factory.create_from_settings(settings)
        assert store.config["persist_directory"] == "./data/db/chroma"
        assert store.config["collection_name"] == "kb"
        assert store.config["host"] == "localhost"
        assert store.config["port"] == 8000

    def test_empty_registry_lists_none(self) -> None:
        with pytest.raises(ValueError, match="\\(none\\)"):
            self.factory.create("anything")
