"""Tests for C1: Core data types (Document, Chunk, ChunkRecord, ProcessedQuery, RetrievalResult).

Tests cover:
- Construction and field access
- Required metadata validation (source_path)
- Serialization (to_dict / from_dict roundtrip)
- ChunkRecord.from_chunk conversion
- Type aliases existence
"""

from __future__ import annotations

import pytest

from src.core.types import (
    Chunk,
    ChunkRecord,
    Document,
    Metadata,
    ProcessedQuery,
    RetrievalResult,
    SparseVector,
    Vector,
)


# ── Document Tests ──────────────────────────────────────────────────────


class TestDocument:
    """Tests for Document dataclass."""

    def test_basic_creation(self) -> None:
        doc = Document(id="doc1", text="Hello", metadata={"source_path": "a.pdf"})
        assert doc.id == "doc1"
        assert doc.text == "Hello"
        assert doc.metadata["source_path"] == "a.pdf"

    def test_missing_source_path_raises(self) -> None:
        with pytest.raises(ValueError, match="source_path"):
            Document(id="doc1", text="Hello", metadata={})

    def test_extra_metadata_allowed(self) -> None:
        doc = Document(
            id="d1",
            text="x",
            metadata={"source_path": "a.pdf", "title": "Report", "pages": 10},
        )
        assert doc.metadata["title"] == "Report"
        assert doc.metadata["pages"] == 10

    def test_to_dict(self) -> None:
        doc = Document(id="d1", text="x", metadata={"source_path": "a.pdf"})
        d = doc.to_dict()
        assert d == {"id": "d1", "text": "x", "metadata": {"source_path": "a.pdf"}}

    def test_from_dict_roundtrip(self) -> None:
        original = Document(id="d1", text="content", metadata={"source_path": "f.pdf"})
        restored = Document.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata

    def test_images_metadata(self) -> None:
        images = [
            {
                "id": "hash_1_0",
                "path": "data/images/coll/hash_1_0.png",
                "page": 1,
                "text_offset": 50,
                "text_length": 20,
                "position": {},
            }
        ]
        doc = Document(
            id="d1",
            text="Before [IMAGE: hash_1_0] After",
            metadata={"source_path": "a.pdf", "images": images},
        )
        assert len(doc.metadata["images"]) == 1
        assert doc.metadata["images"][0]["id"] == "hash_1_0"


# ── Chunk Tests ─────────────────────────────────────────────────────────


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_basic_creation(self) -> None:
        chunk = Chunk(id="c1", text="Part", metadata={"source_path": "a.pdf"})
        assert chunk.id == "c1"
        assert chunk.start_offset is None
        assert chunk.source_ref is None

    def test_missing_source_path_raises(self) -> None:
        with pytest.raises(ValueError, match="source_path"):
            Chunk(id="c1", text="x", metadata={})

    def test_with_offsets_and_source_ref(self) -> None:
        chunk = Chunk(
            id="c1",
            text="x",
            metadata={"source_path": "a.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=100,
            source_ref="doc1",
        )
        assert chunk.start_offset == 0
        assert chunk.end_offset == 100
        assert chunk.source_ref == "doc1"

    def test_to_dict(self) -> None:
        chunk = Chunk(
            id="c1",
            text="x",
            metadata={"source_path": "a.pdf"},
            source_ref="d1",
        )
        d = chunk.to_dict()
        assert d["source_ref"] == "d1"
        assert d["start_offset"] is None

    def test_from_dict_roundtrip(self) -> None:
        original = Chunk(
            id="c1",
            text="text",
            metadata={"source_path": "a.pdf", "chunk_index": 3},
            start_offset=10,
            end_offset=20,
            source_ref="d1",
        )
        restored = Chunk.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.source_ref == original.source_ref
        assert restored.start_offset == original.start_offset


# ── ChunkRecord Tests ──────────────────────────────────────────────────


class TestChunkRecord:
    """Tests for ChunkRecord dataclass."""

    def test_basic_creation(self) -> None:
        rec = ChunkRecord(id="r1", text="x", metadata={"source_path": "a.pdf"})
        assert rec.dense_vector is None
        assert rec.sparse_vector is None

    def test_missing_source_path_raises(self) -> None:
        with pytest.raises(ValueError, match="source_path"):
            ChunkRecord(id="r1", text="x", metadata={})

    def test_with_vectors(self) -> None:
        rec = ChunkRecord(
            id="r1",
            text="x",
            metadata={"source_path": "a.pdf"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"hello": 0.5, "world": 0.3},
        )
        assert len(rec.dense_vector) == 3
        assert rec.sparse_vector["hello"] == 0.5

    def test_from_chunk(self) -> None:
        chunk = Chunk(
            id="c1",
            text="content",
            metadata={"source_path": "a.pdf", "chunk_index": 0},
            source_ref="d1",
        )
        vec = [0.1, 0.2]
        sparse = {"term": 1.0}
        rec = ChunkRecord.from_chunk(chunk, dense_vector=vec, sparse_vector=sparse)
        assert rec.id == "c1"
        assert rec.text == "content"
        assert rec.metadata["source_path"] == "a.pdf"
        assert rec.dense_vector == vec
        assert rec.sparse_vector == sparse

    def test_from_chunk_copies_metadata(self) -> None:
        """Metadata should be a copy, not a reference."""
        chunk = Chunk(id="c1", text="x", metadata={"source_path": "a.pdf"})
        rec = ChunkRecord.from_chunk(chunk)
        rec.metadata["extra"] = "added"
        assert "extra" not in chunk.metadata

    def test_to_dict_roundtrip(self) -> None:
        original = ChunkRecord(
            id="r1",
            text="x",
            metadata={"source_path": "a.pdf"},
            dense_vector=[0.1],
            sparse_vector={"a": 1.0},
        )
        restored = ChunkRecord.from_dict(original.to_dict())
        assert restored.dense_vector == original.dense_vector
        assert restored.sparse_vector == original.sparse_vector


# ── ProcessedQuery Tests ────────────────────────────────────────────────


class TestProcessedQuery:
    """Tests for ProcessedQuery dataclass."""

    def test_basic_creation(self) -> None:
        pq = ProcessedQuery(original_query="test query")
        assert pq.original_query == "test query"
        assert pq.keywords == []
        assert pq.filters == {}

    def test_with_keywords_and_filters(self) -> None:
        pq = ProcessedQuery(
            original_query="Azure config",
            keywords=["Azure", "config"],
            filters={"collection": "docs"},
        )
        assert len(pq.keywords) == 2
        assert pq.filters["collection"] == "docs"

    def test_roundtrip(self) -> None:
        original = ProcessedQuery(
            original_query="test",
            keywords=["a", "b"],
            filters={"k": "v"},
            expanded_terms=["c"],
        )
        restored = ProcessedQuery.from_dict(original.to_dict())
        assert restored.original_query == original.original_query
        assert restored.expanded_terms == ["c"]


# ── RetrievalResult Tests ──────────────────────────────────────────────


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_basic_creation(self) -> None:
        rr = RetrievalResult(chunk_id="c1", score=0.95, text="result text")
        assert rr.chunk_id == "c1"
        assert rr.score == 0.95

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_id"):
            RetrievalResult(chunk_id="", score=0.5, text="x")

    def test_non_numeric_score_raises(self) -> None:
        with pytest.raises(ValueError, match="score"):
            RetrievalResult(chunk_id="c1", score="high", text="x")  # type: ignore[arg-type]

    def test_int_score_accepted(self) -> None:
        rr = RetrievalResult(chunk_id="c1", score=1, text="x")
        assert rr.score == 1

    def test_with_metadata(self) -> None:
        rr = RetrievalResult(
            chunk_id="c1",
            score=0.8,
            text="x",
            metadata={"source_path": "a.pdf", "chunk_index": 3},
        )
        assert rr.metadata["chunk_index"] == 3

    def test_roundtrip(self) -> None:
        original = RetrievalResult(
            chunk_id="c1", score=0.9, text="content", metadata={"k": "v"},
        )
        restored = RetrievalResult.from_dict(original.to_dict())
        assert restored.chunk_id == original.chunk_id
        assert restored.score == original.score


# ── Type Aliases ────────────────────────────────────────────────────────


class TestTypeAliases:
    """Verify type aliases are accessible."""

    def test_metadata_alias(self) -> None:
        m: Metadata = {"source_path": "a.pdf"}
        assert isinstance(m, dict)

    def test_vector_alias(self) -> None:
        v: Vector = [0.1, 0.2, 0.3]
        assert isinstance(v, list)

    def test_sparse_vector_alias(self) -> None:
        sv: SparseVector = {"term": 0.5}
        assert isinstance(sv, dict)
