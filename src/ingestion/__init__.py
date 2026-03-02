"""
Ingestion Pipeline - Offline data ingestion.

Pipeline stages: load -> split -> transform -> embed -> upsert

Components:
- Pipeline: Orchestration with progress callback
- DocumentManager: Document lifecycle management
- Chunking: Document -> Chunks conversion
- Transform: ChunkRefiner / MetadataEnricher / ImageCaptioner
- Embedding: Dense + Sparse encoding
- Storage: VectorUpserter / BM25Indexer / ImageStorage
"""

__all__: list[str] = []
