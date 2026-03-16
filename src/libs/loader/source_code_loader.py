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
import re
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

        brief = self._extract_brief(text, language)

        metadata: dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "source_code",
            "doc_hash": doc_hash,
            "language": language,
            "filename": path.name,
            "line_count": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
            "brief": brief,
        }

        return Document(id=doc_id, text=text, metadata=metadata)

    # ── internal helpers ──────────────────────────────────────────────

    # Doxygen \brief tag (/// \brief ... or ## \brief ...)
    _BRIEF_TAG_RE = re.compile(
        r"^(?:///|##)\s*\\brief\s+(.+)",
        re.MULTILINE,
    )

    # Python module docstring (first triple-quoted string)
    _PY_DOCSTRING_RE = re.compile(
        r'^(?:\"\"\"|\'\'\')(.*?)(?:\"\"\"|\'\'\')',
        re.DOTALL,
    )

    # C/C++ block comment at file start
    _C_BLOCK_COMMENT_RE = re.compile(
        r"^/\*[\s*]*(.*?)\*/",
        re.DOTALL,
    )

    @classmethod
    def _extract_brief(cls, text: str, language: str) -> str:
        """Extract file-level description from header comments.

        Extraction priority:
            1. Doxygen ``\\brief`` tag (both ``///`` and ``##`` styles)
            2. Python module docstring (triple-quoted)
            3. C/C++ block comment ``/* ... */``
            4. Leading ``#`` or ``//`` comment lines (first content lines)

        Returns:
            Brief description string, or empty string if none found.
        """
        # 1. Doxygen \brief tag
        m = cls._BRIEF_TAG_RE.search(text[:2000])
        if m:
            return m.group(1).strip()

        # 2. Python module docstring
        if language == "Python":
            # Skip shebang and encoding lines
            lines = text.lstrip().split("\n")
            body = "\n".join(lines)
            m = cls._PY_DOCSTRING_RE.match(body)
            if m:
                # Take first non-empty line of docstring
                doc_lines = [
                    ln.strip() for ln in m.group(1).strip().split("\n")
                    if ln.strip()
                ]
                return doc_lines[0] if doc_lines else ""

        # 3. C/C++ block comment
        if language == "C++":
            m = cls._C_BLOCK_COMMENT_RE.match(text.lstrip())
            if m:
                content_lines = [
                    ln.strip().lstrip("* ").strip()
                    for ln in m.group(1).strip().split("\n")
                    if ln.strip() and ln.strip() != "*"
                ]
                return content_lines[0] if content_lines else ""

        # 4. Leading comment lines
        comment_prefix = "#" if language == "Python" else "//"
        header_lines: list[str] = []
        for line in text.split("\n")[:30]:
            stripped = line.strip()
            if not stripped:
                if header_lines:
                    break
                continue
            if stripped.startswith(comment_prefix):
                # Strip comment prefix and Doxygen markers
                content = stripped.lstrip(comment_prefix).strip()
                # Skip Doxygen directives (\file, \ingroup, \macro_*, etc.)
                if content.startswith("\\") or not content:
                    continue
                header_lines.append(content)
            elif stripped.startswith("#!"):
                continue  # skip shebang
            else:
                break  # non-comment line → stop

        return " ".join(header_lines) if header_lines else ""
