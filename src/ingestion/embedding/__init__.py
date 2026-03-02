"""
Embedding Module - Dense and Sparse vector encoding.

Components:
- DenseEncoder: Via libs.embedding batch encoding
- SparseEncoder: BM25 term frequency and weights
- BatchProcessor: Batch processing orchestration
"""

from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor, BatchResult

__all__: list[str] = ["DenseEncoder", "SparseEncoder", "BatchProcessor", "BatchResult"]
