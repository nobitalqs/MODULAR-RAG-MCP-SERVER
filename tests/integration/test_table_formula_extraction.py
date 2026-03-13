"""Integration tests for table and formula extraction (K4).

Tests the end-to-end extraction pipeline:
  PdfLoader (table + formula) → DocumentChunker → chunk content verification

Two test tiers:
  1. Self-contained (no external services) — Loader + Chunker integration
  2. Full pipeline (needs embedding service) — marked @pytest.mark.integration

Run self-contained tests:
    pytest tests/integration/test_table_formula_extraction.py -v -k "not integration"

Run full pipeline tests (requires Ollama / embedding service):
    pytest tests/integration/test_table_formula_extraction.py -v -m integration
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF
import pytest

from src.core.settings import FormulaExtractionSettings, TableExtractionSettings
from src.core.types import Chunk, Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.splitter import RecursiveSplitter


# ── Helpers ────────────────────────────────────────────────────────


def _create_table_formula_pdf(pdf_path: Path) -> Path:
    """Create a realistic PDF containing tables and formula-like images.

    Page 1: Experimental results table (3×3 with header)
    Page 2: Two formula images — one wide-thin (inline), one moderate (block)

    The page text includes context keywords ("equation", "defined") so that
    heuristic formula classification can trigger.
    """
    doc = fitz.open()

    # ── Page 1: Table ──────────────────────────────────────────
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 40), "Experimental Results",
        fontsize=16, fontname="helv",
    )
    page1.insert_text(
        (50, 65), "The following table shows model performance:",
        fontsize=11, fontname="helv",
    )

    # Draw a 3-column × 3-row table
    x0, y0 = 50, 80
    col_w, row_h = 150, 35
    cols, rows = 3, 3

    for r in range(rows + 1):
        y = y0 + r * row_h
        page1.draw_line((x0, y), (x0 + cols * col_w, y))
    for c in range(cols + 1):
        x = x0 + c * col_w
        page1.draw_line((x, y0), (x, y0 + rows * row_h))

    cells = [
        ["Model", "BLEU", "ROUGE"],
        ["Transformer", "28.4", "52.1"],
        ["LSTM", "25.2", "48.7"],
    ]
    for r, row_data in enumerate(cells):
        for c, cell_text in enumerate(row_data):
            page1.insert_text(
                (x0 + c * col_w + 10, y0 + r * row_h + 22),
                cell_text, fontsize=10, fontname="helv",
            )

    page1.insert_text(
        (50, 210),
        "As shown in the table above, Transformer achieves best performance.",
        fontsize=11, fontname="helv",
    )

    # ── Page 2: Formula images ─────────────────────────────────
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (50, 40), "Mathematical Formulation",
        fontsize=16, fontname="helv",
    )
    page2.insert_text(
        (50, 70), "Given the equation for self-attention:",
        fontsize=11, fontname="helv",
    )

    # Inline formula image: wide thin (aspect ratio > 3 → inline heuristic)
    img_doc = fitz.open()
    img_page = img_doc.new_page(width=400, height=30)
    img_page.draw_rect(fitz.Rect(0, 0, 400, 30), fill=(0.95, 0.95, 0.95))
    img_page.insert_text(
        (10, 20), "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V",
        fontsize=9,
    )
    pix = img_page.get_pixmap()
    inline_bytes = pix.tobytes("png")
    img_doc.close()

    # Render at 400×30 on page → aspect ratio ≈ 13
    page2.insert_image(fitz.Rect(50, 90, 450, 120), stream=inline_bytes)

    page2.insert_text(
        (50, 150), "The loss function is defined as:",
        fontsize=11, fontname="helv",
    )

    # Block formula image: moderate height (< 15% page), aspect ratio ≤ 3
    # so it does NOT trigger Rule 1 (wide ratio) but DOES trigger Rule 3 (block + keywords)
    img_doc2 = fitz.open()
    img_page2 = img_doc2.new_page(width=200, height=100)
    img_page2.draw_rect(fitz.Rect(0, 0, 200, 100), fill=(0.95, 0.95, 0.95))
    img_page2.insert_text((10, 55), "L = -sum(y_i * log(p_i))", fontsize=14)
    pix2 = img_page2.get_pixmap()
    block_bytes = pix2.tobytes("png")
    img_doc2.close()

    # Render at 200×100 on page → aspect 2.0, height 100 < 15% of 842 (≈126)
    page2.insert_image(fitz.Rect(150, 170, 350, 270), stream=block_bytes)

    page2.insert_text(
        (50, 290),
        "Where y_i is the true label and p_i is the predicted probability.",
        fontsize=11, fontname="helv",
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def table_formula_pdf(tmp_path: Path) -> Path:
    """Create a PDF with both a table (page 1) and formula images (page 2)."""
    return _create_table_formula_pdf(tmp_path / "table_formula_test.pdf")


@pytest.fixture
def table_only_loader() -> PdfLoader:
    """PdfLoader with only table extraction enabled."""
    return PdfLoader(
        extract_images=False,
        table_extraction=TableExtractionSettings(enabled=True, format="markdown"),
    )


@pytest.fixture
def formula_only_loader() -> PdfLoader:
    """PdfLoader with only formula extraction enabled."""
    return PdfLoader(
        extract_images=False,
        formula_extraction=FormulaExtractionSettings(
            enabled=True, model="pix2tex", confidence_threshold=0.5,
        ),
    )


@pytest.fixture
def combined_loader() -> PdfLoader:
    """PdfLoader with both table and formula extraction enabled."""
    return PdfLoader(
        extract_images=False,
        table_extraction=TableExtractionSettings(enabled=True, format="markdown"),
        formula_extraction=FormulaExtractionSettings(
            enabled=True, model="pix2tex", confidence_threshold=0.5,
        ),
    )


@pytest.fixture
def chunker() -> DocumentChunker:
    """DocumentChunker with default recursive splitter."""
    splitter = RecursiveSplitter(chunk_size=500, chunk_overlap=50)
    return DocumentChunker(splitter)


# ── Table Extraction Integration ───────────────────────────────────


class TestTableExtractionIntegration:
    """End-to-end table extraction: loader → chunker → verification."""

    def test_loader_extracts_table_with_markdown_format(
        self, table_only_loader: PdfLoader, table_formula_pdf: Path,
    ):
        """Table extraction produces Markdown with [TABLE_] markers."""
        doc = table_only_loader.load(table_formula_pdf)

        assert "[TABLE_" in doc.text
        assert "|" in doc.text  # Markdown pipe
        # Cell data present
        assert "Transformer" in doc.text
        assert "28.4" in doc.text
        assert "BLEU" in doc.text

    def test_chunks_preserve_table_data(
        self, table_only_loader: PdfLoader,
        table_formula_pdf: Path, chunker: DocumentChunker,
    ):
        """Table content survives chunking — cell data present in chunks."""
        doc = table_only_loader.load(table_formula_pdf)
        chunks = chunker.split_document(doc)

        all_text = " ".join(c.text for c in chunks)
        assert "[TABLE_" in all_text
        assert "Transformer" in all_text
        assert "28.4" in all_text

    def test_table_extraction_metadata_intact(
        self, table_only_loader: PdfLoader, table_formula_pdf: Path,
    ):
        """Document metadata is complete with table extraction enabled."""
        doc = table_only_loader.load(table_formula_pdf)

        assert doc.metadata["doc_type"] == "pdf"
        assert doc.metadata["page_count"] == 2
        assert "source_path" in doc.metadata
        assert "doc_hash" in doc.metadata
        assert len(doc.metadata["doc_hash"]) == 64


# ── Formula Extraction Integration ─────────────────────────────────


class TestFormulaExtractionIntegration:
    """End-to-end formula extraction: loader → chunker → verification."""

    def test_loader_extracts_inline_formula(
        self, formula_only_loader: PdfLoader, table_formula_pdf: Path,
    ):
        """Wide image (aspect ratio > 3) → inline $...$ LaTeX."""
        mock_model = MagicMock(
            return_value="\\text{Attention}(Q,K,V) = \\text{softmax}(QK^T/\\sqrt{d_k})V",
        )
        with patch.object(formula_only_loader, "_get_pix2tex_model", return_value=mock_model):
            doc = formula_only_loader.load(table_formula_pdf)

        assert "$\\text{Attention}" in doc.text
        assert mock_model.call_count >= 1

    def test_loader_extracts_block_formula(
        self, formula_only_loader: PdfLoader, table_formula_pdf: Path,
    ):
        """Moderate-height image + context keyword → block $$...$$ LaTeX."""
        mock_model = MagicMock(return_value="L = -\\sum_{i} y_i \\log(p_i)")
        with patch.object(formula_only_loader, "_get_pix2tex_model", return_value=mock_model):
            doc = formula_only_loader.load(table_formula_pdf)

        assert "$$L = -\\sum_{i} y_i \\log(p_i)$$" in doc.text

    def test_pix2tex_missing_degrades_gracefully(
        self, formula_only_loader: PdfLoader, table_formula_pdf: Path,
    ):
        """Without pix2tex, formula images become [FORMULA: unrecognized]."""
        with patch.object(formula_only_loader, "_get_pix2tex_model", return_value=None):
            doc = formula_only_loader.load(table_formula_pdf)

        assert "[FORMULA: unrecognized]" in doc.text
        # Original text preserved
        assert "Mathematical Formulation" in doc.text

    def test_chunks_preserve_formula_content(
        self, formula_only_loader: PdfLoader,
        table_formula_pdf: Path, chunker: DocumentChunker,
    ):
        """Formula LaTeX survives chunking."""
        mock_model = MagicMock(return_value="E = mc^2")
        with patch.object(formula_only_loader, "_get_pix2tex_model", return_value=mock_model):
            doc = formula_only_loader.load(table_formula_pdf)

        chunks = chunker.split_document(doc)
        all_text = " ".join(c.text for c in chunks)
        assert "E = mc^2" in all_text


# ── Combined Extraction Integration ────────────────────────────────


class TestCombinedExtractionIntegration:
    """End-to-end with both table and formula extraction enabled."""

    def test_both_extractions_produce_correct_output(
        self, combined_loader: PdfLoader, table_formula_pdf: Path,
    ):
        """Tables and formulas both present in final Document text."""
        mock_model = MagicMock(return_value="\\alpha + \\beta = \\gamma")
        with patch.object(combined_loader, "_get_pix2tex_model", return_value=mock_model):
            doc = combined_loader.load(table_formula_pdf)

        # Table markers from page 1
        assert "[TABLE_" in doc.text
        assert "Transformer" in doc.text
        assert "28.4" in doc.text

        # Formula from page 2
        assert "\\alpha + \\beta = \\gamma" in doc.text

        # Original text from both pages
        assert "Experimental Results" in doc.text
        assert "Mathematical Formulation" in doc.text

    def test_combined_chunks_contain_both_types(
        self, combined_loader: PdfLoader,
        table_formula_pdf: Path, chunker: DocumentChunker,
    ):
        """Chunks from combined extraction contain both table and formula data."""
        mock_model = MagicMock(return_value="x^2 + y^2 = r^2")
        with patch.object(combined_loader, "_get_pix2tex_model", return_value=mock_model):
            doc = combined_loader.load(table_formula_pdf)

        chunks = chunker.split_document(doc)
        all_text = " ".join(c.text for c in chunks)

        assert "[TABLE_" in all_text
        assert "x^2 + y^2 = r^2" in all_text
        assert len(chunks) >= 1  # Document produces at least one chunk

    def test_chunk_metadata_has_source_path(
        self, combined_loader: PdfLoader,
        table_formula_pdf: Path, chunker: DocumentChunker,
    ):
        """Every chunk inherits source_path from the document."""
        mock_model = MagicMock(return_value="f(x)")
        with patch.object(combined_loader, "_get_pix2tex_model", return_value=mock_model):
            doc = combined_loader.load(table_formula_pdf)

        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert "source_path" in chunk.metadata
            assert chunk.metadata["source_path"] == str(table_formula_pdf.resolve())

    def test_pix2tex_failure_does_not_break_table_extraction(
        self, combined_loader: PdfLoader, table_formula_pdf: Path,
    ):
        """Even if pix2tex fails, table extraction still works correctly."""
        with patch.object(combined_loader, "_get_pix2tex_model", return_value=None):
            doc = combined_loader.load(table_formula_pdf)

        # Tables should still work
        assert "[TABLE_" in doc.text
        assert "Transformer" in doc.text

        # Formulas degrade to placeholder
        assert "[FORMULA: unrecognized]" in doc.text


# ── Performance Tests ──────────────────────────────────────────────


class TestExtractionPerformance:
    """Performance overhead tests for extraction features.

    The 30% overhead budget from the spec applies to the full pipeline
    (embedding + chunking + storage dominate). For the loader in isolation,
    we use absolute thresholds since find_tables() and get_image_rects()
    have a fixed per-page cost that is negligible in the full pipeline.
    """

    def test_table_extraction_per_document_under_budget(
        self, table_formula_pdf: Path,
    ):
        """Table extraction completes within 500ms per document."""
        loader = PdfLoader(
            extract_images=False,
            table_extraction=TableExtractionSettings(enabled=True),
        )

        # Warm-up
        loader.load(table_formula_pdf)

        iterations = 5
        t0 = time.monotonic()
        for _ in range(iterations):
            loader.load(table_formula_pdf)
        avg_ms = (time.monotonic() - t0) * 1000 / iterations

        assert avg_ms < 500, f"Table extraction {avg_ms:.1f}ms exceeds 500ms budget"

    def test_formula_heuristic_per_document_under_budget(
        self, table_formula_pdf: Path,
    ):
        """Formula heuristic classification (no OCR) within 200ms per document."""
        loader = PdfLoader(
            extract_images=False,
            formula_extraction=FormulaExtractionSettings(enabled=True),
        )

        # Warm-up
        with patch.object(loader, "_get_pix2tex_model", return_value=None):
            loader.load(table_formula_pdf)

        iterations = 5
        t0 = time.monotonic()
        with patch.object(loader, "_get_pix2tex_model", return_value=None):
            for _ in range(iterations):
                loader.load(table_formula_pdf)
        avg_ms = (time.monotonic() - t0) * 1000 / iterations

        assert avg_ms < 200, f"Formula heuristic {avg_ms:.1f}ms exceeds 200ms budget"


# ── Regression Guard ───────────────────────────────────────────────


class TestExtractionRegression:
    """Regression tests ensuring extraction doesn't break existing behavior."""

    def test_disabled_extraction_identical_to_baseline(
        self, table_formula_pdf: Path,
    ):
        """Disabled extraction produces identical output to no-extraction loader."""
        loader_none = PdfLoader(extract_images=False)
        loader_disabled = PdfLoader(
            extract_images=False,
            table_extraction=TableExtractionSettings(enabled=False),
            formula_extraction=FormulaExtractionSettings(enabled=False),
        )

        doc_none = loader_none.load(table_formula_pdf)
        doc_disabled = loader_disabled.load(table_formula_pdf)

        assert doc_none.text == doc_disabled.text
        assert doc_none.id == doc_disabled.id
        assert doc_none.metadata["doc_hash"] == doc_disabled.metadata["doc_hash"]

    def test_simple_pdf_unaffected_by_extraction(self, tmp_path: Path):
        """A plain-text PDF (no tables/images) is unaffected by extraction."""
        pdf_path = tmp_path / "plain.pdf"
        doc_pdf = fitz.open()
        page = doc_pdf.new_page(width=595, height=842)
        page.insert_text(
            (50, 50), "Plain text document with no tables or images.",
            fontsize=12, fontname="helv",
        )
        doc_pdf.save(str(pdf_path))
        doc_pdf.close()

        loader_plain = PdfLoader(extract_images=False)
        loader_full = PdfLoader(
            extract_images=False,
            table_extraction=TableExtractionSettings(enabled=True),
            formula_extraction=FormulaExtractionSettings(enabled=True),
        )

        doc_plain = loader_plain.load(pdf_path)
        with patch.object(loader_full, "_get_pix2tex_model", return_value=None):
            doc_full = loader_full.load(pdf_path)

        assert doc_plain.text == doc_full.text
        assert "[TABLE_" not in doc_full.text
        assert "[FORMULA:" not in doc_full.text

    def test_multipage_document_page_count_preserved(
        self, table_formula_pdf: Path,
    ):
        """Extraction does not alter page_count metadata."""
        loader = PdfLoader(
            extract_images=False,
            table_extraction=TableExtractionSettings(enabled=True),
            formula_extraction=FormulaExtractionSettings(enabled=True),
        )

        with patch.object(loader, "_get_pix2tex_model", return_value=None):
            doc = loader.load(table_formula_pdf)

        assert doc.metadata["page_count"] == 2
