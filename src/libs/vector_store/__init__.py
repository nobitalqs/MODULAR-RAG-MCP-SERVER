"""
VectorStore - Vector database abstraction.

Components:
- BaseVectorStore: Abstract base class
- VectorStoreFactory: Backend routing factory
- ChromaStore: ChromaDB implementation
"""

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.chroma_store import ChromaStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

__all__: list[str] = [
    "BaseVectorStore",
    "ChromaStore",
    "VectorStoreFactory",
]
