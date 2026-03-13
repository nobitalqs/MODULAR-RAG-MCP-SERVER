"""PDF Loader implementation using PyMuPDF.

This module implements PDF parsing with image extraction support,
extracting text content and optionally handling embedded images.

Features:
- Text extraction via PyMuPDF
- Image extraction and storage
- Image placeholder insertion with metadata tracking
- Table extraction (K2) via PyMuPDF find_tables()
- Formula extraction (K3) via pix2tex LaTeX OCR with heuristic detection
- Graceful degradation if image/table/formula extraction fails
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# Context keywords that boost formula classification (Rule 4)
_FORMULA_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "equation", "formula", "where", "let", "given",
    "定义", "公式", "其中",
)


class PdfLoader(BaseLoader):
    """PDF Loader using PyMuPDF for text and image extraction.

    This loader:
    1. Extracts text from PDF pages
    2. Extracts images and saves to data/images/{doc_hash}/
    3. Inserts image placeholders in the format [IMAGE: {image_id}]
    4. Records image metadata in Document.metadata.images

    Configuration:
        extract_images: Enable/disable image extraction (default: True)
        image_storage_dir: Base directory for image storage (default: data/images)

    Graceful Degradation:
        If image extraction fails, logs warning and continues with text-only parsing.
    """

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
        table_extraction: Any | None = None,
        formula_extraction: Any | None = None,
    ):
        """Initialize PDF Loader.

        Args:
            extract_images: Whether to extract images from PDFs.
            image_storage_dir: Base directory for storing extracted images.
            table_extraction: Optional TableExtractionSettings. If None or
                enabled=False, table extraction is skipped.
            formula_extraction: Optional FormulaExtractionSettings. If None or
                enabled=False, formula extraction is skipped. Requires pix2tex
                as an optional dependency.
        """
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self.table_extraction = table_extraction
        self.formula_extraction = formula_extraction
        self._pix2tex_model: Any | None = None
        self._pix2tex_warned: bool = False

    def load(self, file_path: str | Path) -> Document:
        """Load and parse a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Document with extracted text and metadata.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the file is not a valid PDF.
            RuntimeError: If parsing fails critically.
        """
        # Validate file
        path = self._validate_file(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path}")

        # Compute document hash for unique ID and image directory
        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        # Parse PDF with PyMuPDF
        try:
            pdf_doc = fitz.open(str(path))
            text_content = self._extract_text(pdf_doc)
        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e

        # Initialize metadata
        metadata: dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "page_count": len(pdf_doc),
        }

        # Extract title from first lines if available
        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title

        # Handle image extraction (with graceful degradation)
        if self.extract_images:
            try:
                text_content, images_metadata = self._extract_and_process_images(
                    pdf_doc, text_content, doc_hash
                )
                if images_metadata:
                    metadata["images"] = images_metadata
            except Exception as e:
                logger.warning(
                    f"Image extraction failed for {path}, continuing with text-only: {e}"
                )

        pdf_doc.close()

        return Document(id=doc_id, text=text_content, metadata=metadata)

    def _extract_text(self, pdf_doc: fitz.Document) -> str:
        """Extract text content from all pages, with optional table/formula extraction.

        Args:
            pdf_doc: Opened PyMuPDF document.

        Returns:
            Concatenated text from all pages.
        """
        extract_tables = (
            self.table_extraction is not None
            and getattr(self.table_extraction, "enabled", False)
        )
        extract_formulas = (
            self.formula_extraction is not None
            and getattr(self.formula_extraction, "enabled", False)
        )

        text_parts: list[str] = []
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            page_text = page.get_text()

            if extract_tables:
                tables_md = self._extract_tables(page)
                if tables_md:
                    page_text = page_text + "\n\n" + tables_md

            if extract_formulas:
                formulas_text = self._extract_formulas(page)
                if formulas_text:
                    page_text = page_text + "\n\n" + formulas_text

            text_parts.append(page_text)

        return "\n\n".join(text_parts)

    def _extract_tables(self, page: fitz.Page) -> str:
        """Extract tables from a PDF page as Markdown.

        Uses PyMuPDF's built-in find_tables() API. Each detected table is
        converted to Markdown and tagged with [TABLE_n].

        Args:
            page: PyMuPDF page object.

        Returns:
            Concatenated Markdown tables with [TABLE_n] markers, or empty
            string if no tables found.
        """
        try:
            table_finder = page.find_tables()
        except Exception as e:
            logger.warning("Table detection failed on page %d: %s", page.number + 1, e)
            return ""

        if not table_finder.tables:
            return ""

        parts: list[str] = []
        for idx, table in enumerate(table_finder.tables, start=1):
            try:
                md = table.to_markdown()
                parts.append(f"[TABLE_{idx}]\n{md}")
            except Exception as e:
                logger.warning(
                    "Table %d on page %d failed to convert: %s",
                    idx, page.number + 1, e,
                )

        return "\n\n".join(parts)

    def _extract_formulas(self, page: fitz.Page) -> str:
        """Extract formulas from page images using pix2tex LaTeX OCR.

        Applies heuristic detection to classify image regions, then runs
        pix2tex inference on formula candidates.

        Args:
            page: PyMuPDF page object.

        Returns:
            Formula text with LaTeX delimiters, or empty string if none found.
        """
        try:
            image_list = page.get_images(full=True)
        except Exception as e:
            logger.warning("Failed to get images on page %d: %s", page.number + 1, e)
            return ""

        if not image_list:
            return ""

        page_height = page.rect.height
        page_width = page.rect.width
        page_text = page.get_text()

        formulas: list[str] = []
        for img_info in image_list:
            xref = img_info[0]

            # Get rendered rectangle on page for heuristic classification
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []

            if not rects:
                continue

            rect = rects[0]
            width = rect.width
            height = rect.height

            if width <= 0 or height <= 0:
                continue

            # Classify image via heuristics
            formula_type = self._classify_formula_candidate(
                width, height, page_width, page_height, page_text,
            )
            if formula_type is None:
                continue  # Not a formula candidate, skip

            # Run pix2tex OCR
            latex, confidence = self._ocr_formula(page.parent, xref)

            threshold = getattr(
                self.formula_extraction, "confidence_threshold", 0.5,
            )
            if latex and confidence >= threshold:
                if formula_type == "inline":
                    formulas.append(f"${latex}$")
                else:
                    formulas.append(f"$${latex}$$")
            else:
                formulas.append("[FORMULA: unrecognized]")

        return "\n".join(formulas)

    def _classify_formula_candidate(
        self,
        width: float,
        height: float,
        page_width: float,
        page_height: float,
        page_text: str,
    ) -> str | None:
        """Classify an image region as a formula candidate using heuristics.

        Returns:
            "inline" for inline formula, "block" for block formula,
            None if no heuristic matches (regular image).
        """
        # Rule 1: Wide aspect ratio → inline formula
        if height > 0 and width / height > 3:
            return "inline"

        # Rule 2: Small height + narrow width → formula symbol (inline)
        if height < 0.05 * page_height and width < 0.5 * page_width:
            return "inline"

        # Rule 3: Moderate height + context keywords → block formula
        if height < 0.15 * page_height:
            text_lower = page_text.lower()
            if any(kw in text_lower for kw in _FORMULA_CONTEXT_KEYWORDS):
                return "block"

        # No heuristic matched → regular image
        return None

    def _get_pix2tex_model(self) -> Any | None:
        """Lazy-load pix2tex LatexOCR model.

        Returns the model instance, or None if pix2tex is not installed
        or initialization fails. Logs a warning once per session.
        """
        if self._pix2tex_model is not None:
            return self._pix2tex_model

        try:
            from pix2tex.cli import LatexOCR  # type: ignore[import-untyped]
            self._pix2tex_model = LatexOCR()
            return self._pix2tex_model
        except ImportError:
            if not self._pix2tex_warned:
                logger.warning(
                    "pix2tex not installed; formula regions will be marked as "
                    "[FORMULA: unrecognized]. "
                    "Install with: pip install -r requirements-formula.txt",
                )
                self._pix2tex_warned = True
            return None
        except Exception as e:
            if not self._pix2tex_warned:
                logger.warning("pix2tex model initialization failed: %s", e)
                self._pix2tex_warned = True
            return None

    def _ocr_formula(
        self, pdf_doc: fitz.Document, xref: int,
    ) -> tuple[str | None, float]:
        """Run pix2tex OCR on a PDF image by xref.

        Args:
            pdf_doc: Opened PyMuPDF document.
            xref: Image cross-reference number.

        Returns:
            Tuple of (latex_string, confidence). Returns (None, 0.0) on failure.
        """
        model = self._get_pix2tex_model()
        if model is None:
            return None, 0.0

        try:
            from PIL import Image  # type: ignore[import-untyped]

            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes))

            result = model(img)

            # pix2tex may return a string or a dict with confidence
            if isinstance(result, dict):
                latex = result.get("latex", "")
                confidence = float(result.get("confidence", 1.0))
            else:
                latex = str(result).strip()
                confidence = 1.0 if latex else 0.0

            return latex or None, confidence
        except Exception as e:
            logger.warning("pix2tex inference failed for xref %d: %s", xref, e)
            return None, 0.0

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content.

        Args:
            file_path: Path to file.

        Returns:
            Hex string of SHA256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_title(self, text: str) -> str | None:
        """Extract title from first non-empty line.

        Args:
            text: Extracted text content.

        Returns:
            Title string if found, None otherwise.
        """
        lines = text.split("\n")

        # Use first non-empty line as title
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) > 0:
                # Limit title length
                return line[:200]

        return None

    def _extract_and_process_images(
        self, pdf_doc: fitz.Document, text_content: str, doc_hash: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Extract images from PDF and insert placeholders.

        Uses PyMuPDF to extract images, save them to disk, and insert
        placeholders in the text content.

        Args:
            pdf_doc: Opened PyMuPDF document.
            text_content: Extracted text content.
            doc_hash: Document hash for image directory.

        Returns:
            Tuple of (modified_text, images_metadata_list)
        """
        if not self.extract_images:
            logger.debug("Image extraction disabled")
            return text_content, []

        images_metadata = []
        image_placeholders = []

        try:
            # Create image storage directory
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)

            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img_info[0]
                        base_image = pdf_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Generate image ID and filename
                        image_id = self._generate_image_id(
                            doc_hash, page_num + 1, img_index + 1
                        )
                        image_filename = f"{image_id}.{image_ext}"
                        image_path = image_dir / image_filename

                        # Save image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        # Get image dimensions
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)

                        # Create placeholder
                        placeholder = f"[IMAGE: {image_id}]"
                        image_placeholders.append(placeholder)

                        # Convert path to be relative to project root or absolute
                        try:
                            relative_path = image_path.relative_to(Path.cwd())
                        except ValueError:
                            # If not in cwd, use absolute path
                            relative_path = image_path.absolute()

                        # Record metadata
                        image_metadata = {
                            "id": image_id,
                            "path": str(relative_path),
                            "page": page_num + 1,
                            "position": {
                                "width": width,
                                "height": height,
                                "page": page_num + 1,
                                "index": img_index,
                            },
                        }
                        images_metadata.append(image_metadata)

                        logger.debug(f"Extracted image {image_id} from page {page_num + 1}")

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract image {img_index} from page {page_num + 1}: {e}"
                        )
                        continue

            if images_metadata:
                logger.info(f"Extracted {len(images_metadata)} images")
            else:
                logger.debug("No images found")

            # Append placeholders at the end of text content
            if image_placeholders:
                modified_text = text_content + "\n\n" + "\n".join(image_placeholders)
            else:
                modified_text = text_content

            return modified_text, images_metadata

        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
            # Graceful degradation: return original text without images
            return text_content, []

    @staticmethod
    def _generate_image_id(doc_hash: str, page: int, sequence: int) -> str:
        """Generate unique image ID.

        Args:
            doc_hash: Document hash.
            page: Page number (1-based).
            sequence: Image sequence on page (1-based).

        Returns:
            Unique image ID string.
        """
        return f"{doc_hash[:8]}_{page}_{sequence}"
