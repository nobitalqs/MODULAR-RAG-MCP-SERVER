"""
Storage Module - Persistence layer for ingestion outputs.

Components:
- VectorUpserter: Idempotent vector upsert to ChromaDB
- BM25Indexer: Inverted index + IDF persistence
- ImageStorage: File storage + SQLite index
"""

from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter

__all__: list[str] = ["BM25Indexer", "ImageStorage", "VectorUpserter"]
