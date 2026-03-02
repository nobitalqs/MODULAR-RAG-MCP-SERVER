"""PDF Loader implementation using PyMuPDF.

This module implements PDF parsing with image extraction support,
extracting text content and optionally handling embedded images.

Features:
- Text extraction via PyMuPDF
- Image extraction and storage
- Image placeholder insertion with metadata tracking
- Graceful degradation if image extraction fails
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


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
    ):
        """Initialize PDF Loader.

        Args:
            extract_images: Whether to extract images from PDFs.
            image_storage_dir: Base directory for storing extracted images.
        """
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)

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
        """Extract text content from all pages.

        Args:
            pdf_doc: Opened PyMuPDF document.

        Returns:
            Concatenated text from all pages.
        """
        text_parts = []
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            text_parts.append(page.get_text())

        return "\n\n".join(text_parts)

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
