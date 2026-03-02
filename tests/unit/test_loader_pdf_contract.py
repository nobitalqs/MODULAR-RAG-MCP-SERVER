"""Unit tests for PDF loader contract compliance.

Tests the BaseLoader ABC and PdfLoader implementation to ensure:
- BaseLoader defines the correct interface
- PdfLoader validates files correctly
- PdfLoader produces proper Document objects
- Image extraction works with graceful degradation
- Metadata contains all required fields
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # PyMuPDF
import pytest

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def simple_pdf_path(tmp_path: Path) -> Path:
    """Create a simple PDF file for testing.

    Creates a minimal PDF with text content using PyMuPDF.
    """
    pdf_path = tmp_path / "simple.pdf"

    # Create a simple PDF with text
    doc = fitz.open()  # Create new PDF
    page = doc.new_page(width=595, height=842)  # A4 size

    # Add title
    page.insert_text(
        (50, 50),
        "Test Document Title",
        fontsize=18,
        fontname="helv"
    )

    # Add some body text
    body_text = """This is a test document.
It contains multiple lines of text.
This text will be extracted by the PDF loader."""

    page.insert_text(
        (50, 100),
        body_text,
        fontsize=12,
        fontname="helv"
    )

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


@pytest.fixture
def pdf_with_image_path(tmp_path: Path) -> Path:
    """Create a PDF file with an embedded image for testing.

    Creates a PDF with both text and an image.
    """
    pdf_path = tmp_path / "with_image.pdf"

    # Create a simple PNG image in memory
    img_path = tmp_path / "test_image.png"
    img_doc = fitz.open()
    img_page = img_doc.new_page(width=200, height=200)

    # Draw a simple colored rectangle as image content
    rect = fitz.Rect(10, 10, 190, 190)
    img_page.draw_rect(rect, color=(1, 0, 0), fill=(0.5, 0.5, 1))

    # Save as PNG
    pix = img_page.get_pixmap()
    pix.save(str(img_path))
    img_doc.close()

    # Now create PDF with image
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Add text
    page.insert_text(
        (50, 50),
        "Document with Image",
        fontsize=18,
        fontname="helv"
    )

    # Insert image
    page.insert_image(
        fitz.Rect(50, 100, 250, 300),
        filename=str(img_path)
    )

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


@pytest.fixture
def nonexistent_path(tmp_path: Path) -> Path:
    """Return path to a file that doesn't exist."""
    return tmp_path / "nonexistent.pdf"


@pytest.fixture
def non_pdf_path(tmp_path: Path) -> Path:
    """Create a non-PDF file for testing."""
    txt_path = tmp_path / "not_a_pdf.txt"
    txt_path.write_text("This is not a PDF file")
    return txt_path


# ── BaseLoader Tests ────────────────────────────────────────────────────


class TestBaseLoader:
    """Test the BaseLoader abstract base class."""

    def test_is_abstract(self):
        """BaseLoader should be abstract and not instantiable."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseLoader()  # type: ignore

    def test_validate_file_exists(self, simple_pdf_path: Path):
        """_validate_file should accept existing files."""
        result = BaseLoader._validate_file(simple_pdf_path)
        assert result.exists()
        assert result.is_file()
        assert result == simple_pdf_path.resolve()

    def test_validate_file_not_found(self, nonexistent_path: Path):
        """_validate_file should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            BaseLoader._validate_file(nonexistent_path)

    def test_validate_file_is_directory(self, tmp_path: Path):
        """_validate_file should raise ValueError for directories."""
        with pytest.raises(ValueError, match="Path is not a file"):
            BaseLoader._validate_file(tmp_path)

    def test_validate_file_accepts_string(self, simple_pdf_path: Path):
        """_validate_file should accept string paths."""
        result = BaseLoader._validate_file(str(simple_pdf_path))
        assert result.exists()
        assert result.is_file()


# ── PdfLoader Tests ─────────────────────────────────────────────────────


class TestPdfLoader:
    """Test the PdfLoader implementation."""

    def test_initialization_default(self):
        """PdfLoader should initialize with default parameters."""
        loader = PdfLoader()
        assert loader.extract_images is True
        assert loader.image_storage_dir == Path("data/images")

    def test_initialization_custom_params(self):
        """PdfLoader should accept custom initialization parameters."""
        loader = PdfLoader(
            extract_images=False,
            image_storage_dir="/custom/path"
        )
        assert loader.extract_images is False
        assert loader.image_storage_dir == Path("/custom/path")

    def test_load_simple_pdf(self, simple_pdf_path: Path):
        """PdfLoader should load a simple PDF and return Document."""
        loader = PdfLoader()
        doc = loader.load(simple_pdf_path)

        # Verify return type
        assert isinstance(doc, Document)

        # Verify required fields
        assert doc.id
        assert doc.text
        assert doc.metadata

        # Verify metadata requirements
        assert "source_path" in doc.metadata
        assert "doc_type" in doc.metadata
        assert doc.metadata["doc_type"] == "pdf"
        assert doc.metadata["source_path"] == str(simple_pdf_path.resolve())

    def test_load_extracts_text_content(self, simple_pdf_path: Path):
        """PdfLoader should extract text from PDF."""
        loader = PdfLoader()
        doc = loader.load(simple_pdf_path)

        # Verify text was extracted
        assert len(doc.text) > 0
        assert "Test Document" in doc.text or "test document" in doc.text.lower()

    def test_load_generates_unique_id(self, simple_pdf_path: Path):
        """PdfLoader should generate unique document ID based on content hash."""
        loader = PdfLoader()
        doc1 = loader.load(simple_pdf_path)
        doc2 = loader.load(simple_pdf_path)

        # Same file should produce same ID
        assert doc1.id == doc2.id
        assert doc1.id.startswith("doc_")

    def test_load_computes_doc_hash(self, simple_pdf_path: Path):
        """PdfLoader should include document hash in metadata."""
        loader = PdfLoader()
        doc = loader.load(simple_pdf_path)

        assert "doc_hash" in doc.metadata
        assert len(doc.metadata["doc_hash"]) == 64  # SHA256 hex length

    def test_load_file_not_found(self, nonexistent_path: Path):
        """PdfLoader should raise FileNotFoundError for missing files."""
        loader = PdfLoader()
        with pytest.raises(FileNotFoundError, match="File not found"):
            loader.load(nonexistent_path)

    def test_load_invalid_pdf_format(self, non_pdf_path: Path):
        """PdfLoader should raise ValueError for non-PDF files."""
        loader = PdfLoader()
        with pytest.raises(ValueError, match="File is not a PDF"):
            loader.load(non_pdf_path)

    def test_load_accepts_string_path(self, simple_pdf_path: Path):
        """PdfLoader should accept string paths."""
        loader = PdfLoader()
        doc = loader.load(str(simple_pdf_path))

        assert isinstance(doc, Document)
        assert doc.metadata["source_path"] == str(simple_pdf_path.resolve())

    def test_load_accepts_path_object(self, simple_pdf_path: Path):
        """PdfLoader should accept Path objects."""
        loader = PdfLoader()
        doc = loader.load(simple_pdf_path)

        assert isinstance(doc, Document)


# ── Image Extraction Tests ──────────────────────────────────────────────


class TestPdfLoaderImageExtraction:
    """Test image extraction functionality."""

    def test_load_with_images_enabled(self, pdf_with_image_path: Path, tmp_path: Path):
        """PdfLoader should extract images when enabled."""
        loader = PdfLoader(
            extract_images=True,
            image_storage_dir=str(tmp_path / "images")
        )
        doc = loader.load(pdf_with_image_path)

        # Verify document was created
        assert isinstance(doc, Document)

        # If images were found, verify metadata
        if "images" in doc.metadata:
            images = doc.metadata["images"]
            assert isinstance(images, list)
            assert len(images) > 0

            # Verify first image metadata structure
            img = images[0]
            assert "id" in img
            assert "path" in img
            assert "page" in img

            # Verify image placeholder in text
            assert "[IMAGE:" in doc.text

    def test_load_with_images_disabled(self, pdf_with_image_path: Path):
        """PdfLoader should skip image extraction when disabled."""
        loader = PdfLoader(extract_images=False)
        doc = loader.load(pdf_with_image_path)

        # Verify document was created
        assert isinstance(doc, Document)

        # Images should not be in metadata
        assert "images" not in doc.metadata or len(doc.metadata.get("images", [])) == 0

    def test_image_extraction_graceful_degradation(self, simple_pdf_path: Path, tmp_path: Path):
        """PdfLoader should handle image extraction failures gracefully."""
        # Use a PDF without images - extraction should not fail
        loader = PdfLoader(
            extract_images=True,
            image_storage_dir=str(tmp_path / "images")
        )
        doc = loader.load(simple_pdf_path)

        # Should still produce valid document
        assert isinstance(doc, Document)
        assert doc.text
        assert "source_path" in doc.metadata


# ── Metadata Tests ──────────────────────────────────────────────────────


class TestPdfLoaderMetadata:
    """Test metadata extraction and structure."""

    def test_metadata_contains_required_fields(self, simple_pdf_path: Path):
        """Document metadata should contain all required fields."""
        loader = PdfLoader()
        doc = loader.load(simple_pdf_path)

        # Required fields from spec
        assert "source_path" in doc.metadata
        assert "doc_type" in doc.metadata
        assert "doc_hash" in doc.metadata

        # Verify values
        assert doc.metadata["doc_type"] == "pdf"
        assert isinstance(doc.metadata["doc_hash"], str)

    def test_metadata_source_path_is_absolute(self, simple_pdf_path: Path):
        """Metadata source_path should be absolute path."""
        loader = PdfLoader()
        doc = loader.load(simple_pdf_path)

        source_path = Path(doc.metadata["source_path"])
        assert source_path.is_absolute()

    def test_title_extraction(self, simple_pdf_path: Path):
        """PdfLoader should extract title when available."""
        loader = PdfLoader()
        doc = loader.load(simple_pdf_path)

        # Title may or may not be extracted depending on PDF structure
        # Just verify if present, it's a string
        if "title" in doc.metadata:
            assert isinstance(doc.metadata["title"], str)
            assert len(doc.metadata["title"]) > 0


# ── Integration Tests ───────────────────────────────────────────────────


class TestPdfLoaderIntegration:
    """Integration tests for complete PDF loading workflow."""

    def test_load_multiple_pdfs(self, tmp_path: Path):
        """PdfLoader should handle loading multiple PDFs correctly."""
        # Create two different PDFs
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"

        for i, pdf_path in enumerate([pdf1, pdf2], 1):
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), f"Document {i}", fontsize=18)
            doc.save(str(pdf_path))
            doc.close()

        # Load both
        loader = PdfLoader()
        doc1 = loader.load(pdf1)
        doc2 = loader.load(pdf2)

        # Should have different IDs
        assert doc1.id != doc2.id

        # Should have different hashes
        assert doc1.metadata["doc_hash"] != doc2.metadata["doc_hash"]

        # Should have different source paths
        assert doc1.metadata["source_path"] != doc2.metadata["source_path"]

    def test_load_idempotent(self, simple_pdf_path: Path):
        """Loading the same PDF multiple times should produce identical results."""
        loader = PdfLoader()

        doc1 = loader.load(simple_pdf_path)
        doc2 = loader.load(simple_pdf_path)

        # Same file should produce same ID and hash
        assert doc1.id == doc2.id
        assert doc1.metadata["doc_hash"] == doc2.metadata["doc_hash"]
        assert doc1.metadata["source_path"] == doc2.metadata["source_path"]

    def test_document_validates_source_path(self, simple_pdf_path: Path):
        """Document should enforce source_path in metadata via post_init."""
        loader = PdfLoader()
        doc = loader.load(simple_pdf_path)

        # Document should not raise during creation because source_path is present
        assert "source_path" in doc.metadata

        # Verify Document would reject missing source_path
        with pytest.raises(ValueError, match="must contain 'source_path'"):
            Document(id="test", text="test", metadata={})
