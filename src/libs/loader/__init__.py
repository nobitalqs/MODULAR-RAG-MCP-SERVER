"""
Loader - Document loading abstraction.

Components:
- BaseLoader: Abstract base class
- LoaderFactory: Extension-based routing with registry pattern
- PdfLoader: PDF loading via PyMuPDF
- MarkdownLoader: Markdown loading with frontmatter support
- SourceCodeLoader: Source code loading with language detection
- FileIntegrityChecker: SHA256 hash-based dedup
"""

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.source_code_loader import SourceCodeLoader

__all__ = [
    "BaseLoader",
    "LoaderFactory",
    "MarkdownLoader",
    "SourceCodeLoader",
]
