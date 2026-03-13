"""Markdown Loader — parse Markdown files with optional YAML frontmatter.

Extracts:
- YAML frontmatter (flat-merged into metadata, reserved keys protected)
- Title via fallback chain: frontmatter → first ``# heading`` → filename
- Local image references (copied to storage, replaced with ``[IMAGE: id]``)

Design:
- ``**kwargs`` in ``__init__`` for LoaderFactory compatibility
- Image extraction: local files copied, URLs skipped, missing files logged
- Graceful degradation on malformed frontmatter or image errors (logged, skipped)
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# Keys managed by the loader — frontmatter must not overwrite these
_RESERVED_KEYS: frozenset[str] = frozenset({
    "source_path",
    "doc_type",
    "doc_hash",
})

# Regex: YAML frontmatter delimited by ``---``
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Regex: ATX heading ``# Title``
_HEADING_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)

# Regex: Markdown image reference ![alt](path)
_IMAGE_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Accepted extensions
_VALID_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown"})


class MarkdownLoader(BaseLoader):
    """Loads Markdown files into a standardised Document.

    Args:
        **kwargs: Ignored — present for LoaderFactory compatibility.
    """

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
        **kwargs: Any,
    ) -> None:
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)

    def load(self, file_path: str | Path) -> Document:
        """Load and parse a Markdown file.

        Args:
            file_path: Path to the Markdown file.

        Returns:
            Document with parsed content and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file extension is not .md or .markdown.
        """
        path = self._validate_file(file_path)
        if path.suffix.lower() not in _VALID_EXTENSIONS:
            raise ValueError(
                f"File is not a Markdown file: {path} "
                f"(expected {', '.join(sorted(_VALID_EXTENSIONS))})"
            )

        raw = path.read_text(encoding="utf-8")
        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        # Parse frontmatter
        frontmatter, body = self._split_frontmatter(raw)

        # Extract images (with graceful degradation)
        images_metadata: list[dict[str, Any]] = []
        if self.extract_images:
            try:
                body, images_metadata = self._extract_images(body, path, doc_hash)
            except Exception as e:
                logger.warning(
                    "Image extraction failed for %s, continuing with text-only: %s",
                    path,
                    e,
                )

        # Build metadata (loader-managed keys first)
        metadata: dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "markdown",
            "doc_hash": doc_hash,
        }

        # Add images metadata if any
        if images_metadata:
            metadata["images"] = images_metadata

        # Flat-merge frontmatter, protecting reserved keys
        if frontmatter:
            for key, value in frontmatter.items():
                if key in _RESERVED_KEYS:
                    logger.warning(
                        "Frontmatter key '%s' is reserved and will be skipped "
                        "(file: %s)",
                        key,
                        path.name,
                    )
                    continue
                metadata[key] = value

        # Title fallback chain: frontmatter → heading → filename
        if "title" not in metadata:
            heading_title = self._find_heading(body)
            metadata["title"] = heading_title if heading_title else path.stem

        return Document(id=doc_id, text=body, metadata=metadata)

    # ── internal helpers ──────────────────────────────────────────────

    def _extract_images(
        self,
        body: str,
        md_path: Path,
        doc_hash: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Extract local image references and copy to storage.

        Scans for ``![alt](path)`` patterns, copies local files to
        ``image_storage_dir/{doc_hash}/``, and replaces with ``[IMAGE: id]``.

        Args:
            body: Markdown body text (after frontmatter removal).
            md_path: Resolved path to the source Markdown file.
            doc_hash: Document hash for image directory and ID generation.

        Returns:
            Tuple of (modified_body, images_metadata_list).
        """
        matches = list(_IMAGE_MD_RE.finditer(body))
        if not matches:
            return body, []

        md_dir = md_path.parent
        image_dir = self.image_storage_dir / doc_hash
        images_metadata: list[dict[str, Any]] = []
        sequence = 0

        # Collect replacements: (start, end, replacement_str)
        replacements: list[tuple[int, int, str]] = []

        for match in matches:
            alt_text = match.group(1)
            ref = match.group(2).strip()

            # Skip URLs and non-file URI schemes
            if ref.startswith(("http://", "https://", "ftp://", "data:", "//")):
                continue

            # Resolve path
            if Path(ref).is_absolute():
                resolved = Path(ref).resolve()
            else:
                resolved = (md_dir / ref).resolve()

            # Skip if file doesn't exist
            if not resolved.is_file():
                logger.warning(
                    "Image file not found: %s (referenced in %s)",
                    resolved,
                    md_path.name,
                )
                continue

            # Copy image
            try:
                image_dir.mkdir(parents=True, exist_ok=True)
                image_id = self._generate_image_id(doc_hash, sequence + 1)
                dest_filename = f"{image_id}{resolved.suffix}"
                dest_path = image_dir / dest_filename
                shutil.copy2(resolved, dest_path)
                sequence += 1  # only increment after successful copy
            except Exception as e:
                logger.warning(
                    "Failed to copy image %s: %s",
                    resolved,
                    e,
                )
                continue

            # Build path for metadata (relative to cwd, fallback absolute)
            try:
                stored_path = str(dest_path.relative_to(Path.cwd()))
            except ValueError:
                stored_path = str(dest_path.absolute())

            # Record metadata
            images_metadata.append({
                "id": image_id,
                "path": stored_path,
                "alt_text": alt_text,
                "original_ref": ref,
            })

            # Schedule replacement (will apply in reverse order)
            placeholder = f"[IMAGE: {image_id}]"
            replacements.append((match.start(), match.end(), placeholder))

        # Apply replacements in reverse order to preserve offsets
        new_body = body
        for start, end, placeholder in reversed(replacements):
            new_body = new_body[:start] + placeholder + new_body[end:]

        return new_body, images_metadata

    @staticmethod
    def _split_frontmatter(raw: str) -> tuple[dict[str, Any] | None, str]:
        """Split raw text into (frontmatter_dict, body).

        Returns (None, raw) if no valid frontmatter is found.
        """
        match = _FRONTMATTER_RE.match(raw)
        if not match:
            return None, raw

        yaml_str = match.group(1)
        body = raw[match.end():]

        try:
            parsed = yaml.safe_load(yaml_str)
        except yaml.YAMLError as exc:
            logger.warning("Failed to parse YAML frontmatter: %s", exc)
            return None, body

        if not isinstance(parsed, dict):
            logger.warning("Frontmatter is not a mapping, skipping")
            return None, body

        return parsed, body

    @staticmethod
    def _find_heading(text: str) -> str | None:
        """Find the first ATX heading in the first 20 lines."""
        lines = text.split("\n")[:20]
        for line in lines:
            m = _HEADING_RE.match(line)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute SHA256 hex digest of text content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _generate_image_id(doc_hash: str, sequence: int) -> str:
        """Generate unique image ID for markdown images.

        Format: {doc_hash[:8]}_md_{sequence}
        """
        return f"{doc_hash[:8]}_md_{sequence}"

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hex digest of raw file bytes."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
