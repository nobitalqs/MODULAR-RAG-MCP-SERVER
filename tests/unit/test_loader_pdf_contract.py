"""Unit tests for PDF loader contract compliance.

Tests the BaseLoader ABC and PdfLoader implementation to ensure:
- BaseLoader defines the correct interface
- PdfLoader validates files correctly
- PdfLoader produces proper Document objects
- Image extraction works with graceful degradation
- Table extraction (K2) works and degrades gracefully
- Formula extraction (K3) works with graceful degradation
- Metadata contains all required fields
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF
import pytest

from src.core.settings import FormulaExtractionSettings, TableExtractionSettings
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


# ── Table Extraction Tests (K2) ───────────────────────────────────────


def _create_pdf_with_table(pdf_path: Path) -> Path:
    """Helper: create a minimal PDF containing a table drawn with lines.

    PyMuPDF find_tables() detects tables from line intersections,
    so we draw a 3x2 table with explicit horizontal/vertical lines
    and place text in each cell.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Add some text above the table
    page.insert_text((50, 40), "Results Section", fontsize=14, fontname="helv")

    # Draw a 3-column × 3-row table (header + 2 data rows)
    # Table area: x=50..500, y=60..180, row height=40, col width=150
    x0, y0 = 50, 60
    col_w, row_h = 150, 40
    cols, rows = 3, 3

    # Draw horizontal lines
    for r in range(rows + 1):
        y = y0 + r * row_h
        page.draw_line((x0, y), (x0 + cols * col_w, y))

    # Draw vertical lines
    for c in range(cols + 1):
        x = x0 + c * col_w
        page.draw_line((x, y0), (x, y0 + rows * row_h))

    # Insert cell text
    cells = [
        ["Model", "BLEU", "Params"],
        ["Transformer", "28.4", "65M"],
        ["RNN", "25.2", "80M"],
    ]
    for r, row_data in enumerate(cells):
        for c, cell_text in enumerate(row_data):
            x = x0 + c * col_w + 10
            y = y0 + r * row_h + 25
            page.insert_text((x, y), cell_text, fontsize=10, fontname="helv")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_with_table_path(tmp_path: Path) -> Path:
    """Create a PDF with a table for testing table extraction."""
    return _create_pdf_with_table(tmp_path / "table.pdf")


class TestPdfLoaderTableExtraction:
    """Test table extraction functionality (K2)."""

    def test_table_extraction_enabled_contains_marker(self, pdf_with_table_path: Path):
        """With table extraction enabled, output should contain [TABLE_ marker."""
        settings = TableExtractionSettings(enabled=True, format="markdown")
        loader = PdfLoader(
            extract_images=False,
            table_extraction=settings,
        )
        doc = loader.load(pdf_with_table_path)

        assert "[TABLE_" in doc.text

    def test_table_extraction_contains_markdown_pipe(self, pdf_with_table_path: Path):
        """Extracted table should contain Markdown pipe characters."""
        settings = TableExtractionSettings(enabled=True, format="markdown")
        loader = PdfLoader(
            extract_images=False,
            table_extraction=settings,
        )
        doc = loader.load(pdf_with_table_path)

        # Markdown tables use pipe characters
        assert "|" in doc.text

    def test_table_extraction_contains_cell_data(self, pdf_with_table_path: Path):
        """Extracted table should contain the cell text data."""
        settings = TableExtractionSettings(enabled=True, format="markdown")
        loader = PdfLoader(
            extract_images=False,
            table_extraction=settings,
        )
        doc = loader.load(pdf_with_table_path)

        # Check that cell data appears in the output
        for keyword in ("Model", "BLEU", "Transformer"):
            assert keyword in doc.text

    def test_table_extraction_disabled_no_marker(self, pdf_with_table_path: Path):
        """With table extraction disabled, output should NOT contain [TABLE_ marker."""
        settings = TableExtractionSettings(enabled=False, format="markdown")
        loader = PdfLoader(
            extract_images=False,
            table_extraction=settings,
        )
        doc = loader.load(pdf_with_table_path)

        assert "[TABLE_" not in doc.text

    def test_table_extraction_none_settings(self, pdf_with_table_path: Path):
        """With table_extraction=None, behavior identical to disabled."""
        loader = PdfLoader(extract_images=False, table_extraction=None)
        doc = loader.load(pdf_with_table_path)

        assert "[TABLE_" not in doc.text

    def test_no_table_pdf_produces_no_marker(self, simple_pdf_path: Path):
        """PDF without tables should not produce [TABLE_ markers."""
        settings = TableExtractionSettings(enabled=True, format="markdown")
        loader = PdfLoader(
            extract_images=False,
            table_extraction=settings,
        )
        doc = loader.load(simple_pdf_path)

        assert "[TABLE_" not in doc.text

    def test_table_extraction_preserves_existing_text(self, pdf_with_table_path: Path):
        """Table extraction should preserve existing page text."""
        settings = TableExtractionSettings(enabled=True, format="markdown")
        loader = PdfLoader(
            extract_images=False,
            table_extraction=settings,
        )
        doc = loader.load(pdf_with_table_path)

        assert "Results Section" in doc.text

    def test_table_extraction_backward_compatible_init(self):
        """PdfLoader without table_extraction param should work (backward compat)."""
        loader = PdfLoader()
        assert loader.table_extraction is None

    def test_settings_parsing_table_extraction(self):
        """Settings.from_dict() should parse table_extraction correctly."""
        from src.core.settings import Settings

        data = {
            "llm": {
                "provider": "ollama", "model": "m", "temperature": 0.0,
                "max_tokens": 100, "base_url": "http://localhost:11434",
            },
            "embedding": {
                "provider": "ollama", "model": "m", "dimensions": 768,
                "base_url": "http://localhost:11434",
            },
            "vector_store": {
                "provider": "chroma", "persist_directory": "./db",
                "collection_name": "c",
            },
            "retrieval": {
                "dense_top_k": 20, "sparse_top_k": 20,
                "fusion_top_k": 10, "rrf_k": 60,
            },
            "rerank": {
                "enabled": False, "provider": "none", "model": "none",
                "top_k": 5,
            },
            "evaluation": {
                "enabled": False, "provider": "custom",
                "metrics": ["faithfulness"],
            },
            "observability": {
                "log_level": "INFO", "trace_enabled": False,
                "trace_file": "./t.jsonl", "structured_logging": False,
            },
            "ingestion": {
                "chunk_size": 1000, "chunk_overlap": 200,
                "table_extraction": {
                    "enabled": True,
                    "format": "markdown",
                },
            },
        }

        settings = Settings.from_dict(data)
        assert settings.ingestion is not None
        assert settings.ingestion.table_extraction is not None
        assert settings.ingestion.table_extraction.enabled is True
        assert settings.ingestion.table_extraction.format == "markdown"

    def test_settings_parsing_no_table_extraction(self):
        """Settings without table_extraction should parse with None."""
        from src.core.settings import Settings

        data = {
            "llm": {
                "provider": "ollama", "model": "m", "temperature": 0.0,
                "max_tokens": 100, "base_url": "http://localhost:11434",
            },
            "embedding": {
                "provider": "ollama", "model": "m", "dimensions": 768,
                "base_url": "http://localhost:11434",
            },
            "vector_store": {
                "provider": "chroma", "persist_directory": "./db",
                "collection_name": "c",
            },
            "retrieval": {
                "dense_top_k": 20, "sparse_top_k": 20,
                "fusion_top_k": 10, "rrf_k": 60,
            },
            "rerank": {
                "enabled": False, "provider": "none", "model": "none",
                "top_k": 5,
            },
            "evaluation": {
                "enabled": False, "provider": "custom",
                "metrics": ["faithfulness"],
            },
            "observability": {
                "log_level": "INFO", "trace_enabled": False,
                "trace_file": "./t.jsonl", "structured_logging": False,
            },
            "ingestion": {
                "chunk_size": 1000, "chunk_overlap": 200,
            },
        }

        settings = Settings.from_dict(data)
        assert settings.ingestion is not None
        assert settings.ingestion.table_extraction is None


# ── Formula Extraction Helpers ─────────────────────────────────────


def _create_pdf_with_formula_image(pdf_path: Path, *, wide: bool = True) -> Path:
    """Helper: create a PDF containing an image that triggers formula heuristics.

    Args:
        pdf_path: Output path for the PDF.
        wide: If True, create a wide thin image (aspect ratio > 3 → inline formula).
              If False, create a small image (height < 5% page, width < 50% page).
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Add context keywords to boost formula detection
    page.insert_text((50, 40), "Given the equation below:", fontsize=12, fontname="helv")

    # Create a PNG image in memory
    if wide:
        img_w, img_h = 400, 30  # aspect ratio > 13 → inline
        rect = fitz.Rect(50, 60, 450, 90)  # rendered: 400x30
    else:
        img_w, img_h = 100, 30  # small image
        rect = fitz.Rect(50, 60, 150, 90)  # rendered: 100x30 (< 5% of 842)

    # Create a minimal image
    img_doc = fitz.open()
    img_page = img_doc.new_page(width=img_w, height=img_h)
    img_page.draw_rect(fitz.Rect(0, 0, img_w, img_h), fill=(0.9, 0.9, 0.9))
    pix = img_page.get_pixmap()
    img_bytes = pix.tobytes("png")
    img_doc.close()

    page.insert_image(rect, stream=img_bytes)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_with_formula_image_path(tmp_path: Path) -> Path:
    """Create a PDF with a wide thin image (formula candidate)."""
    return _create_pdf_with_formula_image(tmp_path / "formula.pdf", wide=True)


@pytest.fixture
def pdf_with_small_formula_image_path(tmp_path: Path) -> Path:
    """Create a PDF with a small image (formula symbol candidate)."""
    return _create_pdf_with_formula_image(tmp_path / "formula_small.pdf", wide=False)


# ── Formula Extraction Tests (K3) ─────────────────────────────────


class TestPdfLoaderFormulaExtraction:
    """Test formula extraction functionality (K3)."""

    def test_formula_extraction_enabled_with_mock_pix2tex(
        self, pdf_with_formula_image_path: Path,
    ):
        """With formula extraction enabled and pix2tex mocked, output contains LaTeX."""
        settings = FormulaExtractionSettings(
            enabled=True, model="pix2tex", confidence_threshold=0.5,
        )
        loader = PdfLoader(
            extract_images=False,
            formula_extraction=settings,
        )

        mock_model = MagicMock(return_value="E = mc^2")
        with patch.object(loader, "_get_pix2tex_model", return_value=mock_model):
            doc = loader.load(pdf_with_formula_image_path)

        # Should contain LaTeX delimiters
        assert "$" in doc.text
        assert "E = mc^2" in doc.text

    def test_formula_extraction_inline_format(
        self, pdf_with_formula_image_path: Path,
    ):
        """Wide images (aspect ratio > 3) should produce inline $...$ format."""
        settings = FormulaExtractionSettings(
            enabled=True, model="pix2tex", confidence_threshold=0.5,
        )
        loader = PdfLoader(
            extract_images=False,
            formula_extraction=settings,
        )

        mock_model = MagicMock(return_value="x^2 + y^2 = r^2")
        with patch.object(loader, "_get_pix2tex_model", return_value=mock_model):
            doc = loader.load(pdf_with_formula_image_path)

        assert "$x^2 + y^2 = r^2$" in doc.text

    def test_formula_extraction_block_format(self, tmp_path: Path):
        """Block formula candidates should produce $$...$$ format."""
        # Create a PDF with a moderate-height image + context keywords
        pdf_path = tmp_path / "block_formula.pdf"
        doc_pdf = fitz.open()
        page = doc_pdf.new_page(width=595, height=842)

        # Add equation context keyword
        page.insert_text((50, 40), "The equation is defined as:", fontsize=12, fontname="helv")

        # Image: moderate height (< 15% of 842 ≈ 126), NOT wide aspect ratio,
        # NOT small (height >= 5% of 842 ≈ 42), so only Rule 3 matches
        img_w, img_h = 200, 80
        rect = fitz.Rect(50, 100, 250, 180)  # 200x80 rendered

        img_doc = fitz.open()
        img_page = img_doc.new_page(width=img_w, height=img_h)
        img_page.draw_rect(fitz.Rect(0, 0, img_w, img_h), fill=(0.8, 0.8, 0.8))
        pix = img_page.get_pixmap()
        img_bytes = pix.tobytes("png")
        img_doc.close()

        page.insert_image(rect, stream=img_bytes)
        doc_pdf.save(str(pdf_path))
        doc_pdf.close()

        settings = FormulaExtractionSettings(
            enabled=True, model="pix2tex", confidence_threshold=0.5,
        )
        loader = PdfLoader(extract_images=False, formula_extraction=settings)

        mock_model = MagicMock(return_value="\\int_0^\\infty e^{-x} dx = 1")
        with patch.object(loader, "_get_pix2tex_model", return_value=mock_model):
            result = loader.load(pdf_path)

        assert "$$\\int_0^\\infty e^{-x} dx = 1$$" in result.text

    def test_formula_extraction_disabled_no_markers(
        self, pdf_with_formula_image_path: Path,
    ):
        """With formula extraction disabled, no formula markers appear."""
        settings = FormulaExtractionSettings(enabled=False)
        loader = PdfLoader(
            extract_images=False,
            formula_extraction=settings,
        )
        doc = loader.load(pdf_with_formula_image_path)

        assert "$" not in doc.text or "E = mc^2" not in doc.text
        assert "[FORMULA:" not in doc.text

    def test_formula_extraction_none_settings(
        self, pdf_with_formula_image_path: Path,
    ):
        """With formula_extraction=None, behavior identical to disabled."""
        loader = PdfLoader(extract_images=False, formula_extraction=None)
        doc = loader.load(pdf_with_formula_image_path)

        assert "[FORMULA:" not in doc.text

    def test_formula_extraction_pix2tex_not_installed(
        self, pdf_with_formula_image_path: Path,
    ):
        """Without pix2tex, formula regions marked as [FORMULA: unrecognized]."""
        settings = FormulaExtractionSettings(enabled=True)
        loader = PdfLoader(
            extract_images=False,
            formula_extraction=settings,
        )

        with patch.object(loader, "_get_pix2tex_model", return_value=None):
            doc = loader.load(pdf_with_formula_image_path)

        assert "[FORMULA: unrecognized]" in doc.text

    def test_formula_extraction_low_confidence(
        self, pdf_with_formula_image_path: Path,
    ):
        """Low confidence results degrade to [FORMULA: unrecognized]."""
        settings = FormulaExtractionSettings(
            enabled=True, confidence_threshold=0.8,
        )
        loader = PdfLoader(
            extract_images=False,
            formula_extraction=settings,
        )

        # Return dict with low confidence
        mock_model = MagicMock(
            return_value={"latex": "x", "confidence": 0.3},
        )
        with patch.object(loader, "_get_pix2tex_model", return_value=mock_model):
            doc = loader.load(pdf_with_formula_image_path)

        assert "[FORMULA: unrecognized]" in doc.text
        assert "$x$" not in doc.text

    def test_formula_extraction_high_confidence_dict_result(
        self, pdf_with_formula_image_path: Path,
    ):
        """High confidence dict results produce LaTeX output."""
        settings = FormulaExtractionSettings(
            enabled=True, confidence_threshold=0.5,
        )
        loader = PdfLoader(
            extract_images=False,
            formula_extraction=settings,
        )

        mock_model = MagicMock(
            return_value={"latex": "a^2 + b^2 = c^2", "confidence": 0.95},
        )
        with patch.object(loader, "_get_pix2tex_model", return_value=mock_model):
            doc = loader.load(pdf_with_formula_image_path)

        assert "a^2 + b^2 = c^2" in doc.text

    def test_formula_heuristic_no_match_skips_image(self, tmp_path: Path):
        """Image not matching any heuristic should not produce formula markers."""
        # Create a PDF with a large square image (no heuristic match)
        pdf_path = tmp_path / "large_image.pdf"
        doc_pdf = fitz.open()
        page = doc_pdf.new_page(width=595, height=842)
        page.insert_text((50, 40), "A photo:", fontsize=12, fontname="helv")

        # Large square image: aspect ≈ 1, height > 15% page
        img_w, img_h = 300, 300
        rect = fitz.Rect(50, 100, 350, 400)  # 300x300

        img_doc = fitz.open()
        img_page = img_doc.new_page(width=img_w, height=img_h)
        img_page.draw_rect(fitz.Rect(0, 0, img_w, img_h), fill=(0.5, 0.5, 0.5))
        pix = img_page.get_pixmap()
        img_bytes = pix.tobytes("png")
        img_doc.close()

        page.insert_image(rect, stream=img_bytes)
        doc_pdf.save(str(pdf_path))
        doc_pdf.close()

        settings = FormulaExtractionSettings(enabled=True)
        loader = PdfLoader(extract_images=False, formula_extraction=settings)

        mock_model = MagicMock(return_value="should not be called")
        with patch.object(loader, "_get_pix2tex_model", return_value=mock_model):
            doc = loader.load(pdf_path)

        # No formula markers — image didn't match heuristics
        assert "[FORMULA:" not in doc.text
        assert "$" not in doc.text
        mock_model.assert_not_called()

    def test_formula_extraction_preserves_existing_text(
        self, pdf_with_formula_image_path: Path,
    ):
        """Formula extraction should preserve existing page text."""
        settings = FormulaExtractionSettings(enabled=True)
        loader = PdfLoader(
            extract_images=False,
            formula_extraction=settings,
        )

        mock_model = MagicMock(return_value="E = mc^2")
        with patch.object(loader, "_get_pix2tex_model", return_value=mock_model):
            doc = loader.load(pdf_with_formula_image_path)

        assert "equation" in doc.text.lower()

    def test_formula_extraction_ocr_failure_degrades(
        self, pdf_with_formula_image_path: Path,
    ):
        """If pix2tex inference raises an exception, degrade to placeholder."""
        settings = FormulaExtractionSettings(enabled=True)
        loader = PdfLoader(
            extract_images=False,
            formula_extraction=settings,
        )

        mock_model = MagicMock(side_effect=RuntimeError("OCR failed"))
        with patch.object(loader, "_get_pix2tex_model", return_value=mock_model):
            doc = loader.load(pdf_with_formula_image_path)

        assert "[FORMULA: unrecognized]" in doc.text

    def test_formula_extraction_backward_compatible_init(self):
        """PdfLoader without formula_extraction param should work."""
        loader = PdfLoader()
        assert loader.formula_extraction is None

    def test_no_formula_pdf_produces_no_markers(self, simple_pdf_path: Path):
        """PDF without images should not produce formula markers."""
        settings = FormulaExtractionSettings(enabled=True)
        loader = PdfLoader(extract_images=False, formula_extraction=settings)
        doc = loader.load(simple_pdf_path)

        assert "[FORMULA:" not in doc.text

    def test_settings_parsing_formula_extraction(self):
        """Settings.from_dict() should parse formula_extraction correctly."""
        from src.core.settings import Settings

        data = {
            "llm": {
                "provider": "ollama", "model": "m", "temperature": 0.0,
                "max_tokens": 100, "base_url": "http://localhost:11434",
            },
            "embedding": {
                "provider": "ollama", "model": "m", "dimensions": 768,
                "base_url": "http://localhost:11434",
            },
            "vector_store": {
                "provider": "chroma", "persist_directory": "./db",
                "collection_name": "c",
            },
            "retrieval": {
                "dense_top_k": 20, "sparse_top_k": 20,
                "fusion_top_k": 10, "rrf_k": 60,
            },
            "rerank": {
                "enabled": False, "provider": "none", "model": "none",
                "top_k": 5,
            },
            "evaluation": {
                "enabled": False, "provider": "custom",
                "metrics": ["faithfulness"],
            },
            "observability": {
                "log_level": "INFO", "trace_enabled": False,
                "trace_file": "./t.jsonl", "structured_logging": False,
            },
            "ingestion": {
                "chunk_size": 1000, "chunk_overlap": 200,
                "formula_extraction": {
                    "enabled": True,
                    "model": "pix2tex",
                    "confidence_threshold": 0.7,
                },
            },
        }

        settings = Settings.from_dict(data)
        assert settings.ingestion is not None
        assert settings.ingestion.formula_extraction is not None
        assert settings.ingestion.formula_extraction.enabled is True
        assert settings.ingestion.formula_extraction.model == "pix2tex"
        assert settings.ingestion.formula_extraction.confidence_threshold == 0.7

    def test_settings_parsing_no_formula_extraction(self):
        """Settings without formula_extraction should parse with None."""
        from src.core.settings import Settings

        data = {
            "llm": {
                "provider": "ollama", "model": "m", "temperature": 0.0,
                "max_tokens": 100, "base_url": "http://localhost:11434",
            },
            "embedding": {
                "provider": "ollama", "model": "m", "dimensions": 768,
                "base_url": "http://localhost:11434",
            },
            "vector_store": {
                "provider": "chroma", "persist_directory": "./db",
                "collection_name": "c",
            },
            "retrieval": {
                "dense_top_k": 20, "sparse_top_k": 20,
                "fusion_top_k": 10, "rrf_k": 60,
            },
            "rerank": {
                "enabled": False, "provider": "none", "model": "none",
                "top_k": 5,
            },
            "evaluation": {
                "enabled": False, "provider": "custom",
                "metrics": ["faithfulness"],
            },
            "observability": {
                "log_level": "INFO", "trace_enabled": False,
                "trace_file": "./t.jsonl", "structured_logging": False,
            },
            "ingestion": {
                "chunk_size": 1000, "chunk_overlap": 200,
            },
        }

        settings = Settings.from_dict(data)
        assert settings.ingestion is not None
        assert settings.ingestion.formula_extraction is None
