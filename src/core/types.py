"""Core data types and contracts for the entire pipeline.

Defines the fundamental data structures shared across all stages:
- Ingestion: loaders, transforms, embedding, storage
- Retrieval: query engine, search, reranking
- MCP Server: tools, response formatting

Design Principles:
- Centralized contracts avoid coupling between stages.
- Serializable via to_dict / from_dict.
- Extensible metadata with required minimum fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ── Type Aliases ────────────────────────────────────────────────────────

Metadata = dict[str, Any]
Vector = list[float]
SparseVector = dict[str, float]


# ── Document ────────────────────────────────────────────────────────────


@dataclass
class Document:
    """Raw document loaded from source, before splitting.

    Attributes:
        id: Unique document identifier (e.g., file hash).
        text: Content in standardised Markdown. Images use ``[IMAGE: {id}]``.
        metadata: Must contain ``source_path``; extensible with extra keys.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "source_path" not in self.metadata:
            raise ValueError("Document metadata must contain 'source_path'")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Deserialise from dict."""
        return cls(**data)


# ── Chunk ───────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """Text chunk produced by splitting a Document.

    Attributes:
        id: Unique chunk identifier.
        text: Chunk content (subset of Document.text).
        metadata: Inherited from Document + chunk-specific fields.
        start_offset: Character offset in original document (optional).
        end_offset: Character offset end (optional).
        source_ref: Parent Document.id for traceability (optional).
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_offset: int | None = None
    end_offset: int | None = None
    source_ref: str | None = None

    def __post_init__(self) -> None:
        if "source_path" not in self.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(**data)


# ── ChunkRecord ─────────────────────────────────────────────────────────


@dataclass
class ChunkRecord:
    """Fully processed chunk ready for storage and retrieval.

    Extends Chunk with dense and sparse vector representations.

    Attributes:
        id: Stable identifier for idempotent upsert.
        text: Same as Chunk.text.
        metadata: Extended with enrichment (title, summary, tags, etc.).
        dense_vector: Dense embedding vector (optional).
        sparse_vector: Term-weight map for BM25 (optional).
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if "source_path" not in self.metadata:
            raise ValueError("ChunkRecord metadata must contain 'source_path'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkRecord:
        return cls(**data)

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        dense_vector: list[float] | None = None,
        sparse_vector: dict[str, float] | None = None,
    ) -> ChunkRecord:
        """Create from a Chunk, copying metadata (not sharing reference)."""
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )


# ── ProcessedQuery ──────────────────────────────────────────────────────


@dataclass
class ProcessedQuery:
    """Processed query ready for retrieval.

    Attributes:
        original_query: Raw user query string.
        keywords: Extracted keywords after stopword removal.
        filters: Filter conditions (e.g., ``{"collection": "docs"}``).
        expanded_terms: Optional synonym/expansion terms (future use).
    """

    original_query: str
    keywords: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    expanded_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessedQuery:
        return cls(**data)


# ── RetrievalResult ─────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """Single retrieval result from Dense/Sparse/Hybrid search.

    Attributes:
        chunk_id: Retrieved chunk identifier.
        score: Relevance score (higher = more relevant).
        text: Chunk text content.
        metadata: Associated metadata.
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not isinstance(self.score, (int, float)):
            raise ValueError(
                f"score must be numeric, got {type(self.score).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalResult:
        return cls(**data)
