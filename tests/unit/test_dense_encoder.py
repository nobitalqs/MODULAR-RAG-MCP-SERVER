"""Unit tests for DenseEncoder.

Tests the DenseEncoder class in isolation using mocked BaseEmbedding providers.
Validates batch processing, error handling, and output correctness.

Tests cover:
- Constructor validation (batch_size)
- Basic encoding (single, multiple, order preservation)
- Batch processing (respects batch_size, exact boundaries, large batch)
- Input validation (empty list, empty text, whitespace-only)
- Error handling (provider failure, batch range, vector count/dim mismatch)
- Utility methods (get_batch_count)
- Integration-like scenarios (100 chunks, trace passthrough)
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.core.types import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding


# ── Fake embedding provider ─────────────────────────────────────────


class FakeEmbedding(BaseEmbedding):
    """Deterministic embedding provider for testing.

    Returns vectors derived from text length, making assertions easy.
    """

    def __init__(self, dimension: int = 1536, fail_on_call: bool = False) -> None:
        self.dimension = dimension
        self.fail_on_call = fail_on_call
        self.call_count = 0
        self.call_history: list[list[str]] = []

    def embed(
        self, texts: list[str], trace: object = None, **kwargs: object
    ) -> list[list[float]]:
        self.call_count += 1
        self.call_history.append(texts)

        if self.fail_on_call:
            raise RuntimeError("Simulated embedding failure")

        self.validate_texts(texts)

        vectors = []
        for text in texts:
            base_value = len(text) / 1000.0
            vector = [base_value + (i * 0.001) for i in range(self.dimension)]
            vectors.append(vector)
        return vectors


# ── Constructor ──────────────────────────────────────────────────────


class TestConstructor:
    def test_valid_init(self):
        embedding = FakeEmbedding()
        encoder = DenseEncoder(embedding, batch_size=32)
        assert encoder.embedding is embedding
        assert encoder.batch_size == 32

    def test_default_batch_size(self):
        encoder = DenseEncoder(FakeEmbedding())
        assert encoder.batch_size == 100

    def test_rejects_zero_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DenseEncoder(FakeEmbedding(), batch_size=0)

    def test_rejects_negative_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DenseEncoder(FakeEmbedding(), batch_size=-1)


# ── Basic encoding ───────────────────────────────────────────────────


class TestBasicEncoding:
    def test_single_chunk(self):
        embedding = FakeEmbedding(dimension=4)
        encoder = DenseEncoder(embedding, batch_size=10)

        chunks = [Chunk(id="1", text="Hello world", metadata={"source_path": "t.pdf"})]
        vectors = encoder.encode(chunks)

        assert len(vectors) == 1
        assert len(vectors[0]) == 4
        assert embedding.call_count == 1
        assert embedding.call_history[0] == ["Hello world"]

    def test_multiple_chunks(self):
        embedding = FakeEmbedding(dimension=8)
        encoder = DenseEncoder(embedding, batch_size=10)

        chunks = [
            Chunk(id="1", text="First chunk", metadata={"source_path": "t.pdf"}),
            Chunk(id="2", text="Second chunk", metadata={"source_path": "t.pdf"}),
            Chunk(id="3", text="Third chunk", metadata={"source_path": "t.pdf"}),
        ]
        vectors = encoder.encode(chunks)

        assert len(vectors) == 3
        assert all(len(v) == 8 for v in vectors)
        assert embedding.call_count == 1  # all in one batch

    def test_preserves_chunk_order(self):
        embedding = FakeEmbedding(dimension=4)
        encoder = DenseEncoder(embedding, batch_size=10)

        chunks = [
            Chunk(id="1", text="A" * 10, metadata={"source_path": "t.pdf"}),
            Chunk(id="2", text="B" * 20, metadata={"source_path": "t.pdf"}),
            Chunk(id="3", text="C" * 30, metadata={"source_path": "t.pdf"}),
        ]
        vectors = encoder.encode(chunks)

        # FakeEmbedding: base_value = len(text)/1000 → longer text → larger first element
        assert vectors[0][0] < vectors[1][0] < vectors[2][0]


# ── Batch processing ─────────────────────────────────────────────────


class TestBatchProcessing:
    def test_respects_batch_size(self):
        embedding = FakeEmbedding(dimension=4)
        encoder = DenseEncoder(embedding, batch_size=2)

        chunks = [
            Chunk(id=str(i), text=f"Chunk {i}", metadata={"source_path": "t.pdf"})
            for i in range(5)
        ]
        vectors = encoder.encode(chunks)

        # 5 chunks / batch_size 2 → 3 calls: [0:2], [2:4], [4:5]
        assert embedding.call_count == 3
        assert len(embedding.call_history[0]) == 2
        assert len(embedding.call_history[1]) == 2
        assert len(embedding.call_history[2]) == 1
        assert len(vectors) == 5

    def test_exact_batch_boundary(self):
        embedding = FakeEmbedding(dimension=4)
        encoder = DenseEncoder(embedding, batch_size=3)

        chunks = [
            Chunk(id=str(i), text=f"Chunk {i}", metadata={"source_path": "t.pdf"})
            for i in range(6)
        ]
        vectors = encoder.encode(chunks)

        assert embedding.call_count == 2  # exactly 2 batches
        assert len(vectors) == 6

    def test_large_batch(self):
        embedding = FakeEmbedding(dimension=4)
        encoder = DenseEncoder(embedding, batch_size=100)

        chunks = [
            Chunk(id=str(i), text=f"Chunk {i}", metadata={"source_path": "t.pdf"})
            for i in range(10)
        ]
        vectors = encoder.encode(chunks)

        assert embedding.call_count == 1  # single batch
        assert len(vectors) == 10


# ── Input validation ─────────────────────────────────────────────────


class TestInputValidation:
    def test_rejects_empty_list(self):
        encoder = DenseEncoder(FakeEmbedding())
        with pytest.raises(ValueError, match="Cannot encode empty chunks list"):
            encoder.encode([])

    def test_rejects_empty_text(self):
        encoder = DenseEncoder(FakeEmbedding())
        chunks = [
            Chunk(id="1", text="Valid text", metadata={"source_path": "t.pdf"}),
            Chunk(id="2", text="", metadata={"source_path": "t.pdf"}),
        ]
        with pytest.raises(ValueError, match="Chunk at index 1.*has empty"):
            encoder.encode(chunks)

    def test_rejects_whitespace_only_text(self):
        encoder = DenseEncoder(FakeEmbedding())
        chunks = [
            Chunk(id="1", text="   \n\t  ", metadata={"source_path": "t.pdf"}),
        ]
        with pytest.raises(ValueError, match="has empty or whitespace-only text"):
            encoder.encode(chunks)


# ── Error handling ───────────────────────────────────────────────────


class TestErrorHandling:
    def test_provider_failure_surfaced(self):
        embedding = FakeEmbedding(fail_on_call=True)
        encoder = DenseEncoder(embedding, batch_size=10)

        chunks = [Chunk(id="1", text="Test", metadata={"source_path": "t.pdf"})]
        with pytest.raises(RuntimeError, match="Failed to encode batch.*Simulated embedding failure"):
            encoder.encode(chunks)

    def test_failure_includes_batch_range(self):
        embedding = FakeEmbedding(fail_on_call=True)
        encoder = DenseEncoder(embedding, batch_size=2)

        chunks = [
            Chunk(id=str(i), text=f"Chunk {i}", metadata={"source_path": "t.pdf"})
            for i in range(3)
        ]
        with pytest.raises(RuntimeError, match=r"Failed to encode batch 0-2"):
            encoder.encode(chunks)

    def test_validates_vector_count(self):
        embedding = Mock(spec=BaseEmbedding)
        embedding.embed.return_value = [[0.1, 0.2]]  # only 1 vector

        encoder = DenseEncoder(embedding, batch_size=10)
        chunks = [
            Chunk(id="1", text="Chunk 1", metadata={"source_path": "t.pdf"}),
            Chunk(id="2", text="Chunk 2", metadata={"source_path": "t.pdf"}),
        ]
        with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
            encoder.encode(chunks)

    def test_validates_vector_dimensions(self):
        embedding = Mock(spec=BaseEmbedding)
        embedding.embed.return_value = [
            [0.1, 0.2, 0.3],  # 3 dims
            [0.4, 0.5],       # 2 dims — inconsistent
        ]

        encoder = DenseEncoder(embedding, batch_size=10)
        chunks = [
            Chunk(id="1", text="Chunk 1", metadata={"source_path": "t.pdf"}),
            Chunk(id="2", text="Chunk 2", metadata={"source_path": "t.pdf"}),
        ]
        with pytest.raises(RuntimeError, match="Inconsistent vector dimensions"):
            encoder.encode(chunks)


# ── Utility ──────────────────────────────────────────────────────────


class TestGetBatchCount:
    def test_single_batch(self):
        encoder = DenseEncoder(FakeEmbedding(), batch_size=10)
        assert encoder.get_batch_count(5) == 1
        assert encoder.get_batch_count(10) == 1

    def test_multiple_batches(self):
        encoder = DenseEncoder(FakeEmbedding(), batch_size=10)
        assert encoder.get_batch_count(11) == 2
        assert encoder.get_batch_count(20) == 2
        assert encoder.get_batch_count(21) == 3

    def test_zero_chunks(self):
        encoder = DenseEncoder(FakeEmbedding(), batch_size=10)
        assert encoder.get_batch_count(0) == 0

    def test_varies_with_batch_size(self):
        embedding = FakeEmbedding()
        assert DenseEncoder(embedding, batch_size=2).get_batch_count(10) == 5
        assert DenseEncoder(embedding, batch_size=100).get_batch_count(10) == 1


# ── Integration-like ─────────────────────────────────────────────────


class TestIntegration:
    def test_realistic_100_chunks(self):
        embedding = FakeEmbedding(dimension=1536)
        encoder = DenseEncoder(embedding, batch_size=32)

        chunks = [
            Chunk(
                id=f"chunk_{i}",
                text=f"This is chunk number {i} with some content",
                metadata={"source_path": "test.pdf", "page": i // 10},
            )
            for i in range(100)
        ]
        vectors = encoder.encode(chunks)

        assert len(vectors) == 100
        assert all(len(v) == 1536 for v in vectors)
        # 100 / 32 = 4 batches (32+32+32+4)
        assert embedding.call_count == 4
        total_processed = sum(len(b) for b in embedding.call_history)
        assert total_processed == 100

    def test_trace_context_passed_through(self):
        embedding = Mock(spec=BaseEmbedding)
        embedding.embed.return_value = [[0.1, 0.2, 0.3]]

        encoder = DenseEncoder(embedding, batch_size=10)
        chunks = [Chunk(id="1", text="Test", metadata={"source_path": "t.pdf"})]

        mock_trace = {"trace_id": "test_trace"}
        encoder.encode(chunks, trace=mock_trace)

        embedding.embed.assert_called_once()
        call_kwargs = embedding.embed.call_args.kwargs
        assert call_kwargs["trace"] == mock_trace


# ── Retrieval text ────────────────────────────────────────────────────


class TestRetrievalText:
    """DenseEncoder should prefer metadata['retrieval_text'] over chunk.text."""

    def test_uses_retrieval_text_when_present(self):
        embedding = FakeEmbedding(dimension=4)
        encoder = DenseEncoder(embedding, batch_size=10)

        chunk = Chunk(
            id="rt_001",
            text="def foo(): pass",
            metadata={
                "source_path": "test.py",
                "retrieval_text": "A function that does nothing.",
            },
        )
        encoder.encode([chunk])

        # FakeEmbedding records call_history — check the text sent
        assert embedding.call_history[0] == ["A function that does nothing."]

    def test_falls_back_to_text_without_retrieval_text(self):
        embedding = FakeEmbedding(dimension=4)
        encoder = DenseEncoder(embedding, batch_size=10)

        chunk = Chunk(
            id="rt_002",
            text="def bar(): return 42",
            metadata={"source_path": "test.py"},
        )
        encoder.encode([chunk])

        assert embedding.call_history[0] == ["def bar(): return 42"]

    def test_falls_back_when_retrieval_text_empty(self):
        embedding = FakeEmbedding(dimension=4)
        encoder = DenseEncoder(embedding, batch_size=10)

        chunk = Chunk(
            id="rt_003",
            text="def baz(): return 0",
            metadata={
                "source_path": "test.py",
                "retrieval_text": "",
            },
        )
        encoder.encode([chunk])

        # Empty retrieval_text should fall back to chunk.text
        assert embedding.call_history[0] == ["def baz(): return 0"]
