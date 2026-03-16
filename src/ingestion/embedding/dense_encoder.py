"""Dense Encoder: batch text-to-vector conversion via BaseEmbedding provider.

Bridges the ingestion pipeline and the pluggable embedding layer.
Extracts text from Chunk objects, processes in configurable batches,
and validates output shape (vector count + dimension consistency).

Design Principles:
    - Dependency Injection: receives a pre-built BaseEmbedding instance
    - Batch-First: configurable batch_size to control API call frequency
    - Stateless: no internal state between encode() calls
    - Observable: passes TraceContext through to embedding provider
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.types import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class DenseEncoder:
    """Encode text chunks into dense vectors using a BaseEmbedding provider.

    Args:
        embedding: Embedding provider instance (from EmbeddingFactory).
        batch_size: Number of chunks per API call (default 100).

    Raises:
        ValueError: If *batch_size* is not positive.
    """

    def __init__(self, embedding: BaseEmbedding, batch_size: int = 100) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.embedding = embedding
        self.batch_size = batch_size

    # ── public interface ──────────────────────────────────────────────

    def encode(
        self,
        chunks: list[Chunk],
        trace: Any = None,
    ) -> list[list[float]]:
        """Encode chunks into dense vectors.

        Pipeline:
            1. Extract text from each chunk
            2. Validate non-empty content
            3. Process in batches via ``embedding.embed()``
            4. Validate output shape

        Args:
            chunks: Non-empty list of Chunk objects.
            trace: Optional TraceContext for observability.

        Returns:
            Dense vectors in the same order as input chunks.

        Raises:
            ValueError: If *chunks* is empty or contains blank text.
            RuntimeError: If embedding provider fails or output shape is wrong.
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")

        texts = [c.metadata.get("retrieval_text") or c.text for c in chunks]
        self._validate_texts(texts, chunks)

        all_vectors: list[list[float]] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            try:
                batch_vectors = self.embedding.embed(texts=batch_texts, trace=trace)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to encode batch {batch_start}-{batch_end}: {exc}"
                ) from exc

            if len(batch_vectors) != len(batch_texts):
                raise RuntimeError(
                    f"Embedding provider returned {len(batch_vectors)} vectors "
                    f"for {len(batch_texts)} texts in batch {batch_start}-{batch_end}"
                )

            all_vectors.extend(batch_vectors)

        if len(all_vectors) != len(chunks):
            raise RuntimeError(
                f"Vector count mismatch: got {len(all_vectors)} vectors "
                f"for {len(chunks)} chunks"
            )

        self._validate_dimensions(all_vectors)

        return all_vectors

    # ── utilities ─────────────────────────────────────────────────────

    def get_batch_count(self, num_chunks: int) -> int:
        """Calculate the number of batches needed for *num_chunks*."""
        if num_chunks <= 0:
            return 0
        return (num_chunks + self.batch_size - 1) // self.batch_size

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _validate_texts(texts: list[str], chunks: list[Chunk]) -> None:
        """Reject empty / whitespace-only texts."""
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunks[i].id}) "
                    f"has empty or whitespace-only text"
                )

    @staticmethod
    def _validate_dimensions(vectors: list[list[float]]) -> None:
        """Ensure all vectors share the same dimensionality."""
        if not vectors:
            return
        expected_dim = len(vectors[0])
        for i, vec in enumerate(vectors):
            if len(vec) != expected_dim:
                raise RuntimeError(
                    f"Inconsistent vector dimensions: vector {i} has "
                    f"{len(vec)} dimensions, expected {expected_dim}"
                )
