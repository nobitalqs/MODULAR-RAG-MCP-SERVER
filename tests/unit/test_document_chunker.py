"""Tests for C4: Document Chunker (Splitter Integration).

Tests cover:
- Chunk ID generation (deterministic, unique)
- Metadata inheritance (copy all fields)
- chunk_index addition (sequential 0-based)
- source_ref traceability
- Type conversion (Document -> List[Chunk])
- Error handling (empty text, no chunks from splitter)
"""

from __future__ import annotations

import pytest

from src.core.types import Chunk, Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.splitter.base_splitter import BaseSplitter


# ── Test Fixtures ───────────────────────────────────────────────────────


class FakeSplitter(BaseSplitter):
    """Fake splitter for testing isolation.

    Splits on double newlines to create predictable chunks.
    """

    def __init__(self, chunks: list[str] | None = None):
        """Initialize with optional predetermined chunks.

        Args:
            chunks: If provided, split_text returns these chunks.
                   Otherwise, splits on double newlines.
        """
        self._predetermined_chunks = chunks

    def split_text(self, text: str, trace=None, **kwargs) -> list[str]:
        """Split text on double newlines or return predetermined chunks."""
        if self._predetermined_chunks is not None:
            return self._predetermined_chunks

        # Split on double newlines, filter empty chunks
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        return chunks if chunks else [text]


# ── DocumentChunker Tests ──────────────────────────────────────────────


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    def test_basic_split(self) -> None:
        """Test basic document splitting."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="First paragraph.\n\nSecond paragraph.",
            metadata={"source_path": "test.pdf"}
        )

        chunks = chunker.split_document(doc)

        assert len(chunks) == 2
        assert chunks[0].text == "First paragraph."
        assert chunks[1].text == "Second paragraph."

    def test_chunk_id_deterministic(self) -> None:
        """Test that same document produces same chunk IDs."""
        splitter = FakeSplitter(chunks=["chunk1", "chunk2"])
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="original text",
            metadata={"source_path": "test.pdf"}
        )

        # Split twice
        chunks1 = chunker.split_document(doc)
        chunks2 = chunker.split_document(doc)

        assert chunks1[0].id == chunks2[0].id
        assert chunks1[1].id == chunks2[1].id

    def test_chunk_id_format(self) -> None:
        """Test chunk ID has correct format: {doc_id}_{index:04d}_{hash8}."""
        splitter = FakeSplitter(chunks=["first", "second"])
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc_abc",
            text="text",
            metadata={"source_path": "test.pdf"}
        )

        chunks = chunker.split_document(doc)

        # Check format
        assert chunks[0].id.startswith("doc_abc_0000_")
        assert chunks[1].id.startswith("doc_abc_0001_")

        # Check hash length (8 chars)
        hash_part_0 = chunks[0].id.split("_")[-1]
        hash_part_1 = chunks[1].id.split("_")[-1]
        assert len(hash_part_0) == 8
        assert len(hash_part_1) == 8

    def test_chunk_id_uniqueness(self) -> None:
        """Test different text produces different chunk IDs."""
        splitter = FakeSplitter(chunks=["chunk1", "chunk2"])
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="text",
            metadata={"source_path": "test.pdf"}
        )

        chunks = chunker.split_document(doc)

        # Different text -> different IDs
        assert chunks[0].id != chunks[1].id

    def test_metadata_inheritance(self) -> None:
        """Test all document metadata is copied to chunks."""
        splitter = FakeSplitter(chunks=["chunk1"])
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="text",
            metadata={
                "source_path": "test.pdf",
                "title": "Test Document",
                "author": "Alice",
                "pages": 5
            }
        )

        chunks = chunker.split_document(doc)
        chunk = chunks[0]

        # Check all fields inherited
        assert chunk.metadata["source_path"] == "test.pdf"
        assert chunk.metadata["title"] == "Test Document"
        assert chunk.metadata["author"] == "Alice"
        assert chunk.metadata["pages"] == 5

    def test_chunk_index_sequential(self) -> None:
        """Test chunk_index is sequential 0-based."""
        splitter = FakeSplitter(chunks=["a", "b", "c", "d"])
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="text",
            metadata={"source_path": "test.pdf"}
        )

        chunks = chunker.split_document(doc)

        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[1].metadata["chunk_index"] == 1
        assert chunks[2].metadata["chunk_index"] == 2
        assert chunks[3].metadata["chunk_index"] == 3

    def test_source_ref_traceability(self) -> None:
        """Test source_ref points to parent document ID."""
        splitter = FakeSplitter(chunks=["chunk1", "chunk2"])
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc_xyz",
            text="text",
            metadata={"source_path": "test.pdf"}
        )

        chunks = chunker.split_document(doc)

        # All chunks should reference parent doc
        assert chunks[0].metadata["source_ref"] == "doc_xyz"
        assert chunks[1].metadata["source_ref"] == "doc_xyz"

    def test_type_conversion(self) -> None:
        """Test output is list of Chunk objects."""
        splitter = FakeSplitter(chunks=["chunk1"])
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="text",
            metadata={"source_path": "test.pdf"}
        )

        chunks = chunker.split_document(doc)

        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert hasattr(chunks[0], 'id')
        assert hasattr(chunks[0], 'text')
        assert hasattr(chunks[0], 'metadata')

    def test_metadata_not_shared(self) -> None:
        """Test chunk metadata is a copy, not a reference."""
        splitter = FakeSplitter(chunks=["chunk1", "chunk2"])
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="text",
            metadata={"source_path": "test.pdf", "shared": "value"}
        )

        chunks = chunker.split_document(doc)

        # Modify first chunk's metadata
        chunks[0].metadata["shared"] = "modified"

        # Second chunk should be unaffected
        assert chunks[1].metadata["shared"] == "value"

        # Document metadata should be unaffected
        assert doc.metadata["shared"] == "value"

    def test_empty_text_raises(self) -> None:
        """Test that empty text raises ValueError."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="",
            metadata={"source_path": "test.pdf"}
        )

        with pytest.raises(ValueError, match="no text content"):
            chunker.split_document(doc)

    def test_whitespace_only_text_raises(self) -> None:
        """Test that whitespace-only text raises ValueError."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="   \n\n  \t  ",
            metadata={"source_path": "test.pdf"}
        )

        with pytest.raises(ValueError, match="no text content"):
            chunker.split_document(doc)

    def test_no_chunks_from_splitter_raises(self) -> None:
        """Test that no chunks from splitter raises ValueError."""
        splitter = FakeSplitter(chunks=[])
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="some text",
            metadata={"source_path": "test.pdf"}
        )

        with pytest.raises(ValueError, match="no chunks"):
            chunker.split_document(doc)


# ── Internal Methods Tests ─────────────────────────────────────────────


class TestChunkIDGeneration:
    """Tests for _generate_chunk_id method."""

    def test_format(self) -> None:
        """Test chunk ID format."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        chunk_id = chunker._generate_chunk_id("doc_123", 0, "Hello world")

        parts = chunk_id.split("_")
        assert parts[0] == "doc"
        assert parts[1] == "123"
        assert parts[2] == "0000"
        assert len(parts[3]) == 8  # hash length

    def test_index_padding(self) -> None:
        """Test index is zero-padded to 4 digits."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        id_0 = chunker._generate_chunk_id("doc1", 0, "text")
        id_5 = chunker._generate_chunk_id("doc1", 5, "text")
        id_99 = chunker._generate_chunk_id("doc1", 99, "text")
        id_1234 = chunker._generate_chunk_id("doc1", 1234, "text")

        assert "_0000_" in id_0
        assert "_0005_" in id_5
        assert "_0099_" in id_99
        assert "_1234_" in id_1234

    def test_deterministic_hash(self) -> None:
        """Test same text produces same hash."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        id1 = chunker._generate_chunk_id("doc1", 0, "Hello world")
        id2 = chunker._generate_chunk_id("doc1", 0, "Hello world")

        assert id1 == id2

    def test_different_text_different_hash(self) -> None:
        """Test different text produces different hash."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        id1 = chunker._generate_chunk_id("doc1", 0, "Hello world")
        id2 = chunker._generate_chunk_id("doc1", 0, "Goodbye world")

        assert id1 != id2


class TestMetadataInheritance:
    """Tests for _inherit_metadata method."""

    def test_basic_inheritance(self) -> None:
        """Test basic metadata inheritance."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="text",
            metadata={
                "source_path": "test.pdf",
                "title": "Test",
                "custom_field": 123
            }
        )

        metadata = chunker._inherit_metadata(doc, 0)

        assert metadata["source_path"] == "test.pdf"
        assert metadata["title"] == "Test"
        assert metadata["custom_field"] == 123

    def test_adds_chunk_index(self) -> None:
        """Test chunk_index is added."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="text",
            metadata={"source_path": "test.pdf"}
        )

        metadata_0 = chunker._inherit_metadata(doc, 0)
        metadata_5 = chunker._inherit_metadata(doc, 5)

        assert metadata_0["chunk_index"] == 0
        assert metadata_5["chunk_index"] == 5

    def test_adds_source_ref(self) -> None:
        """Test source_ref is added."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc_xyz",
            text="text",
            metadata={"source_path": "test.pdf"}
        )

        metadata = chunker._inherit_metadata(doc, 0)

        assert metadata["source_ref"] == "doc_xyz"

    def test_returns_copy(self) -> None:
        """Test returned metadata is a copy."""
        splitter = FakeSplitter()
        chunker = DocumentChunker(splitter)

        doc = Document(
            id="doc1",
            text="text",
            metadata={"source_path": "test.pdf", "mutable": "original"}
        )

        metadata = chunker._inherit_metadata(doc, 0)
        metadata["mutable"] = "modified"

        # Original should be unchanged
        assert doc.metadata["mutable"] == "original"
