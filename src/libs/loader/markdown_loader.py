"""Markdown Loader — parse Markdown files with optional YAML frontmatter.

Extracts:
- YAML frontmatter (flat-merged into metadata, reserved keys protected)
- Title via fallback chain: frontmatter → first ``# heading`` → filename
- Full Markdown body text (images preserved as-is)

Design:
- ``**kwargs`` in ``__init__`` for LoaderFactory compatibility
- No image processing — ``![alt](path)`` kept verbatim
- Graceful degradation on malformed frontmatter (logged, skipped)
"""

from __future__ import annotations

import hashlib
import logging
import re
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

# Accepted extensions
_VALID_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown"})


class MarkdownLoader(BaseLoader):
    """Loads Markdown files into a standardised Document.

    Args:
        **kwargs: Ignored — present for LoaderFactory compatibility.
    """

    def __init__(self, **kwargs: Any) -> None:
        # kwargs absorbed for factory compatibility
        pass

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
        doc_hash = self._compute_hash(raw)
        doc_id = f"doc_{doc_hash[:16]}"

        # Parse frontmatter
        frontmatter, body = self._split_frontmatter(raw)

        # Build metadata (loader-managed keys first)
        metadata: dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "markdown",
            "doc_hash": doc_hash,
        }

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
