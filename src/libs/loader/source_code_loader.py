"""Source Code Loader — load C++/Python source files as Documents.

Detects language from file extension, reads raw text with UTF-8
(errors='replace' to prevent encoding crashes), and populates metadata
with language, filename, and line count.

Design:
- ``**kwargs`` in ``__init__`` for LoaderFactory compatibility
- ``_LANGUAGE_MAP`` defines supported extensions
- ``read_text(errors="replace")`` handles non-UTF-8 gracefully
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# Extension → language name mapping
_LANGUAGE_MAP: dict[str, str] = {
    ".c": "C++",
    ".cpp": "C++",
    ".cxx": "C++",
    ".cc": "C++",
    ".h": "C++",
    ".hxx": "C++",
    ".py": "Python",
}


class SourceCodeLoader(BaseLoader):
    """Loads source code files into a standardised Document.

    Args:
        **kwargs: Ignored — present for LoaderFactory compatibility.
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    def load(self, file_path: str | Path) -> Document:
        """Load and parse a source code file.

        Args:
            file_path: Path to the source file.

        Returns:
            Document with raw source text and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the extension is not in _LANGUAGE_MAP.
        """
        path = self._validate_file(file_path)
        suffix = path.suffix.lower()

        language = _LANGUAGE_MAP.get(suffix)
        if language is None:
            supported = ", ".join(sorted(_LANGUAGE_MAP))
            raise ValueError(
                f"Unsupported source code extension '{suffix}'. "
                f"Supported: {supported}"
            )

        text = path.read_text(encoding="utf-8", errors="replace")
        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = f"doc_{doc_hash[:16]}"

        metadata: dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "source_code",
            "doc_hash": doc_hash,
            "language": language,
            "filename": path.name,
            "line_count": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
        }

        return Document(id=doc_id, text=text, metadata=metadata)
