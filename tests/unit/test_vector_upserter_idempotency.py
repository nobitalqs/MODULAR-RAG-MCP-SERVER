"""Unit tests for VectorUpserter idempotency and correctness.

Test Coverage:
1. Idempotency: Same chunk produces same ID on repeated upserts
2. Determinism: Chunk ID generation is stable and reproducible
3. Content sensitivity: Different content produces different IDs
4. Batch operations: Ordering and correctness with multiple chunks
5. Error handling: Validation and failure scenarios
"""

import pytest
from unittest.mock import Mock

from src.core.types import Chunk
from src.ingestion.storage.vector_upserter import VectorUpserter


# ============================================================================
# Fixtures
# ============================================================================


class FakeVectorStore:
    """Fake VectorStore that records upsert calls."""

    def __init__(self) -> None:
        self.upserted_records: list[dict] = []
        self.upsert_call_count = 0

    def upsert(self, records, trace=None, **kwargs):
        self.upsert_call_count += 1
        self.upserted_records.extend(records)


@pytest.fixture
def fake_store():
    return FakeVectorStore()


@pytest.fixture
def upserter(fake_store):
    return VectorUpserter(vector_store=fake_store)


@pytest.fixture
def sample_chunk():
    return Chunk(
        id="temp_id",
        text="This is a test chunk for vector storage.",
        metadata={
            "source_path": "data/documents/test.pdf",
            "chunk_index": 0,
            "source_ref": "doc_abc123",
        },
    )


@pytest.fixture
def sample_vector():
    return [0.1, 0.2, 0.3, 0.4, 0.5]


# ============================================================================
# Chunk ID Generation Tests
# ============================================================================


def test_chunk_id_deterministic(upserter, sample_chunk):
    """Test that same chunk produces same ID every time."""
    id1 = upserter._generate_chunk_id(sample_chunk)
    id2 = upserter._generate_chunk_id(sample_chunk)
    id3 = upserter._generate_chunk_id(sample_chunk)

    assert id1 == id2 == id3, "Chunk ID must be deterministic"


def test_chunk_id_format(upserter, sample_chunk):
    """Test that chunk ID follows {source_hash}_{index:04d}_{content_hash}."""
    chunk_id = upserter._generate_chunk_id(sample_chunk)

    parts = chunk_id.split("_")

    assert len(parts) == 3, "Chunk ID must have 3 parts"
    assert len(parts[0]) == 8, "Source hash must be 8 characters"
    assert parts[1] == "0000", "Index must be zero-padded to 4 digits"
    assert len(parts[2]) == 8, "Content hash must be 8 characters"


def test_chunk_id_changes_with_content(upserter, sample_chunk):
    """Test that different content produces different IDs."""
    id1 = upserter._generate_chunk_id(sample_chunk)

    modified = Chunk(
        id=sample_chunk.id,
        text="Different content",
        metadata=sample_chunk.metadata.copy(),
    )
    id2 = upserter._generate_chunk_id(modified)

    assert id1 != id2, "Different content must produce different IDs"
    # Source hash and index stay same
    assert id1.split("_")[0] == id2.split("_")[0]
    assert id1.split("_")[1] == id2.split("_")[1]


def test_chunk_id_changes_with_index(upserter, sample_chunk):
    """Test that different index produces different IDs."""
    id1 = upserter._generate_chunk_id(sample_chunk)

    modified = Chunk(
        id=sample_chunk.id,
        text=sample_chunk.text,
        metadata={**sample_chunk.metadata, "chunk_index": 5},
    )
    id2 = upserter._generate_chunk_id(modified)

    assert id1 != id2, "Different index must produce different IDs"
    assert id1.split("_")[1] == "0000"
    assert id2.split("_")[1] == "0005"


def test_chunk_id_changes_with_source_path(upserter, sample_chunk):
    """Test that different source path produces different IDs."""
    id1 = upserter._generate_chunk_id(sample_chunk)

    modified = Chunk(
        id=sample_chunk.id,
        text=sample_chunk.text,
        metadata={**sample_chunk.metadata, "source_path": "other.pdf"},
    )
    id2 = upserter._generate_chunk_id(modified)

    assert id1 != id2
    assert id1.split("_")[0] != id2.split("_")[0], "Source hash should differ"


def test_chunk_id_generation_missing_source_path():
    """Test that missing source_path raises ValueError at Chunk creation."""
    with pytest.raises(ValueError, match="source_path"):
        Chunk(id="temp", text="Test", metadata={"chunk_index": 0})


def test_chunk_id_generation_missing_chunk_index(upserter):
    """Test that missing chunk_index raises ValueError."""
    chunk = Chunk(id="temp", text="Test", metadata={"source_path": "test.pdf"})

    with pytest.raises(ValueError, match="chunk_index"):
        upserter._generate_chunk_id(chunk)


# ============================================================================
# Upsert Tests
# ============================================================================


def test_upsert_single_chunk(upserter, fake_store, sample_chunk, sample_vector):
    """Test upserting a single chunk with vector."""
    chunk_ids = upserter.upsert([sample_chunk], [sample_vector])

    assert len(chunk_ids) == 1
    assert isinstance(chunk_ids[0], str)

    assert fake_store.upsert_call_count == 1
    records = fake_store.upserted_records

    assert len(records) == 1
    assert records[0]["id"] == chunk_ids[0]
    assert records[0]["vector"] == sample_vector
    assert records[0]["metadata"]["text"] == sample_chunk.text


def test_upsert_multiple_chunks(upserter, fake_store):
    """Test upserting multiple chunks maintains order."""
    chunks = [
        Chunk(
            id=f"temp{i}",
            text=f"Chunk {i}",
            metadata={"source_path": "test.pdf", "chunk_index": i},
        )
        for i in range(5)
    ]
    vectors = [[float(i)] * 5 for i in range(5)]

    chunk_ids = upserter.upsert(chunks, vectors)

    assert len(chunk_ids) == 5
    records = fake_store.upserted_records
    assert len(records) == 5

    for i, record in enumerate(records):
        assert record["id"] == chunk_ids[i]
        assert record["vector"] == vectors[i]
        assert record["metadata"]["text"] == f"Chunk {i}"


def test_upsert_idempotency(upserter, sample_chunk, sample_vector):
    """Test that repeated upserts produce same IDs."""
    ids1 = upserter.upsert([sample_chunk], [sample_vector])
    ids2 = upserter.upsert([sample_chunk], [sample_vector])

    assert ids1 == ids2, "Idempotent upsert must produce same IDs"


def test_upsert_preserves_metadata(upserter, fake_store, sample_chunk, sample_vector):
    """Test that all metadata is preserved in storage."""
    sample_chunk.metadata["custom_field"] = "custom_value"
    sample_chunk.metadata["tags"] = ["tag1", "tag2"]

    upserter.upsert([sample_chunk], [sample_vector])

    metadata = fake_store.upserted_records[0]["metadata"]

    assert metadata["source_path"] == "data/documents/test.pdf"
    assert metadata["chunk_index"] == 0
    assert metadata["source_ref"] == "doc_abc123"
    assert metadata["custom_field"] == "custom_value"
    assert metadata["tags"] == ["tag1", "tag2"]
    assert metadata["text"] == sample_chunk.text


def test_upsert_empty_chunks_raises_error(upserter):
    """Test that empty chunks list raises ValueError."""
    with pytest.raises(ValueError, match="empty chunks"):
        upserter.upsert([], [])


def test_upsert_mismatched_lengths_raises_error(upserter, sample_chunk):
    """Test that mismatched chunks and vectors raises error."""
    with pytest.raises(ValueError, match="must match"):
        upserter.upsert([sample_chunk, sample_chunk], [[0.1, 0.2]])


def test_upsert_vector_store_failure(sample_chunk, sample_vector):
    """Test that vector store failures are wrapped in RuntimeError."""
    mock_store = Mock()
    mock_store.upsert.side_effect = Exception("Connection failed")

    upserter = VectorUpserter(vector_store=mock_store)

    with pytest.raises(RuntimeError, match="Vector store upsert failed"):
        upserter.upsert([sample_chunk], [sample_vector])


def test_upsert_with_trace_context(fake_store, sample_chunk, sample_vector):
    """Test that trace context is passed to vector store."""
    mock_store = Mock()
    upserter = VectorUpserter(vector_store=mock_store)
    mock_trace = Mock()

    upserter.upsert([sample_chunk], [sample_vector], trace=mock_trace)

    call_kwargs = mock_store.upsert.call_args[1]
    assert call_kwargs["trace"] == mock_trace


# ============================================================================
# Batch Upsert Tests
# ============================================================================


def test_upsert_batch_single_batch(upserter, fake_store):
    """Test upserting a single batch."""
    chunks = [
        Chunk(id="t1", text="C1", metadata={"source_path": "test.pdf", "chunk_index": 0}),
        Chunk(id="t2", text="C2", metadata={"source_path": "test.pdf", "chunk_index": 1}),
    ]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    chunk_ids = upserter.upsert_batch([(chunks, vectors)])

    assert len(chunk_ids) == 2
    assert fake_store.upsert_call_count == 1


def test_upsert_batch_multiple_batches(upserter, fake_store):
    """Test upserting multiple batches flattens into single upsert."""
    batch1 = (
        [
            Chunk(id="t1", text="C1", metadata={"source_path": "test.pdf", "chunk_index": 0}),
            Chunk(id="t2", text="C2", metadata={"source_path": "test.pdf", "chunk_index": 1}),
        ],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    batch2 = (
        [
            Chunk(id="t3", text="C3", metadata={"source_path": "test.pdf", "chunk_index": 2}),
        ],
        [[0.5, 0.6]],
    )

    chunk_ids = upserter.upsert_batch([batch1, batch2])

    assert len(chunk_ids) == 3
    assert fake_store.upsert_call_count == 1
    assert len(fake_store.upserted_records) == 3


def test_upsert_batch_empty_raises(upserter):
    """Test that empty batches list raises error."""
    with pytest.raises(ValueError, match="empty"):
        upserter.upsert_batch([])


# ============================================================================
# Edge Cases
# ============================================================================


def test_chunk_with_unicode_text(upserter):
    """Test handling of unicode characters in chunk text."""
    chunk = Chunk(
        id="temp",
        text="测试中文 émojis αβγ",
        metadata={"source_path": "test.pdf", "chunk_index": 0},
    )

    chunk_id = upserter._generate_chunk_id(chunk)

    assert isinstance(chunk_id, str)
    assert chunk_id.isascii()


def test_chunk_with_long_source_path(upserter):
    """Test handling of very long source paths."""
    chunk = Chunk(
        id="temp",
        text="Test",
        metadata={"source_path": "a" * 500 + ".pdf", "chunk_index": 0},
    )

    chunk_id = upserter._generate_chunk_id(chunk)

    assert len(chunk_id) < 50


def test_chunk_with_large_index(upserter):
    """Test handling of large chunk indices."""
    chunk = Chunk(
        id="temp",
        text="Test",
        metadata={"source_path": "test.pdf", "chunk_index": 9999},
    )

    chunk_id = upserter._generate_chunk_id(chunk)

    assert "_9999_" in chunk_id
