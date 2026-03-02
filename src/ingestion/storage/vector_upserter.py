"""Vector Upserter for idempotent writes to vector database.

Generates deterministic chunk IDs from content, transforms chunks
and vectors into storage records, and calls VectorStore.upsert().

Chunk ID Format::

    {source_hash}_{chunk_index:04d}_{content_hash}

    source_hash  = SHA256(source_path)[:8]
    content_hash = SHA256(chunk.text)[:8]

This ensures same content produces same ID (idempotent) and
content changes produce different IDs (versioning).

Design Principles:
    - Idempotent: Same content → same ID, repeated writes safe
    - Deterministic: Stable SHA256-based ID generation
    - Observable: Accepts TraceContext for future integration
    - Type-Safe: Full validation of inputs
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.core.types import Chunk
from src.libs.vector_store.base_vector_store import BaseVectorStore


class VectorUpserter:
    """Write chunks and vectors to vector database with idempotent guarantees.

    Args:
        vector_store: Pre-built VectorStore instance (dependency injection).
    """

    def __init__(self, vector_store: BaseVectorStore) -> None:
        self.vector_store = vector_store

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        trace: Any | None = None,
    ) -> list[str]:
        """Upsert chunks with their vectors to vector store.

        Args:
            chunks: Chunk objects to store.
            vectors: Embedding vectors (same order and length as chunks).
            trace: Optional TraceContext for observability.

        Returns:
            Generated chunk IDs in input order.

        Raises:
            ValueError: If lengths mismatch or chunks is empty.
            RuntimeError: If vector store upsert fails.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match "
                f"vector count ({len(vectors)})"
            )

        if not chunks:
            raise ValueError("Cannot upsert empty chunks list")

        records: list[dict[str, Any]] = []
        chunk_ids: list[str] = []

        for chunk, vector in zip(chunks, vectors):
            chunk_id = self._generate_chunk_id(chunk)
            chunk_ids.append(chunk_id)

            records.append({
                "id": chunk_id,
                "vector": vector,
                "metadata": {
                    **chunk.metadata,
                    "text": chunk.text,
                    "chunk_id": chunk_id,
                },
            })

        try:
            self.vector_store.upsert(records, trace=trace)
        except Exception as e:
            raise RuntimeError(f"Vector store upsert failed: {e}") from e

        return chunk_ids

    def upsert_batch(
        self,
        batches: list[tuple[list[Chunk], list[list[float]]]],
        trace: Any | None = None,
    ) -> list[str]:
        """Upsert multiple batches in a single operation.

        Flattens all batches and delegates to :meth:`upsert` to reduce
        vector store round trips.

        Args:
            batches: List of ``(chunks, vectors)`` tuples.
            trace: Optional TraceContext for observability.

        Returns:
            All generated chunk IDs in order.
        """
        if not batches:
            raise ValueError("Cannot upsert empty batches list")

        all_chunks: list[Chunk] = []
        all_vectors: list[list[float]] = []

        for chunks, vectors in batches:
            all_chunks.extend(chunks)
            all_vectors.extend(vectors)

        return self.upsert(all_chunks, all_vectors, trace=trace)

    @staticmethod
    def _generate_chunk_id(chunk: Chunk) -> str:
        """Generate deterministic chunk ID from content.

        Format: ``{source_hash}_{index:04d}_{content_hash}``

        Raises:
            ValueError: If ``chunk_index`` is missing from metadata.
        """
        if "chunk_index" not in chunk.metadata:
            raise ValueError("Chunk metadata must contain 'chunk_index'")

        source_path = chunk.metadata["source_path"]
        chunk_index = chunk.metadata["chunk_index"]

        source_hash = hashlib.sha256(
            source_path.encode("utf-8")
        ).hexdigest()[:8]
        content_hash = hashlib.sha256(
            chunk.text.encode("utf-8")
        ).hexdigest()[:8]

        return f"{source_hash}_{chunk_index:04d}_{content_hash}"
