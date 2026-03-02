"""Batch Processor for orchestrating dense and sparse encoding.

Coordinates DenseEncoder and SparseEncoder in a unified batch workflow.

Design Principles:
- Orchestration: Coordinates both encoders per batch, not per chunk
- Config-Driven: Batch size from settings, not hardcoded
- Observable: Records batch timing and statistics via TraceContext
- Error Isolation: Individual batch failures don't crash entire pipeline
- Deterministic: Same inputs produce same batching and results
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


@dataclass
class BatchResult:
    """Result of batch processing operation.

    Attributes:
        dense_vectors: Dense embeddings (one per chunk).
        sparse_stats: Term statistics (one per chunk).
        batch_count: Number of batches processed.
        total_time: Total processing time in seconds.
        successful_chunks: Number of successfully processed chunks.
        failed_chunks: Number of chunks that failed processing.
    """

    dense_vectors: list[list[float]]
    sparse_stats: list[dict[str, Any]]
    batch_count: int
    total_time: float
    successful_chunks: int
    failed_chunks: int


class BatchProcessor:
    """Orchestrates batch processing of chunks through encoding pipeline.

    Divides chunks into batches, drives both dense and sparse encoders
    per batch, and collects timing metrics. Each batch failure is
    isolated — remaining batches continue processing.

    Args:
        dense_encoder: DenseEncoder instance for embedding generation.
        sparse_encoder: SparseEncoder instance for term statistics.
        batch_size: Number of chunks per batch (default 100).

    Raises:
        ValueError: If *batch_size* <= 0.
    """

    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: int = 100,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def process(
        self,
        chunks: list[Chunk],
        trace: Any | None = None,
    ) -> BatchResult:
        """Process chunks through dense and sparse encoding pipeline.

        Args:
            chunks: Non-empty list of Chunk objects to process.
            trace: Optional TraceContext for observability.

        Returns:
            BatchResult containing vectors, statistics, and metrics.

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("Cannot process empty chunks list")

        start_time = time.time()

        batches = self._create_batches(chunks)
        batch_count = len(batches)

        dense_vectors: list[list[float]] = []
        sparse_stats: list[dict[str, Any]] = []
        successful_chunks = 0
        failed_chunks = 0

        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()

            try:
                batch_dense = self.dense_encoder.encode(batch, trace=trace)
                dense_vectors.extend(batch_dense)

                batch_sparse = self.sparse_encoder.encode(batch, trace=trace)
                sparse_stats.extend(batch_sparse)

                successful_chunks += len(batch)

            except Exception as e:
                failed_chunks += len(batch)
                if trace:
                    trace.record_stage(
                        f"batch_{batch_idx}_error",
                        {"error": str(e), "batch_size": len(batch)},
                    )

            batch_duration = time.time() - batch_start

            if trace:
                trace.record_stage(
                    f"batch_{batch_idx}",
                    {
                        "batch_size": len(batch),
                        "duration_seconds": batch_duration,
                        "chunks_processed": len(batch),
                    },
                )

        total_time = time.time() - start_time

        if trace:
            trace.record_stage(
                "batch_processing",
                {
                    "total_chunks": len(chunks),
                    "batch_count": batch_count,
                    "batch_size": self.batch_size,
                    "successful_chunks": successful_chunks,
                    "failed_chunks": failed_chunks,
                    "total_time_seconds": total_time,
                },
            )

        return BatchResult(
            dense_vectors=dense_vectors,
            sparse_stats=sparse_stats,
            batch_count=batch_count,
            total_time=total_time,
            successful_chunks=successful_chunks,
            failed_chunks=failed_chunks,
        )

    def _create_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Divide chunks into batches of specified size, preserving order."""
        return [
            chunks[i : i + self.batch_size]
            for i in range(0, len(chunks), self.batch_size)
        ]

    def get_batch_count(self, total_chunks: int) -> int:
        """Calculate number of batches needed for *total_chunks*."""
        if total_chunks <= 0:
            return 0
        return (total_chunks + self.batch_size - 1) // self.batch_size
