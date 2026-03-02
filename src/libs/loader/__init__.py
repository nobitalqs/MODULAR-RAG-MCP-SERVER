"""
Loader - Document loading abstraction.

Components:
- BaseLoader: Abstract base class
- PdfLoader: PDF loading via MarkItDown + PyMuPDF
- FileIntegrityChecker: SHA256 hash-based dedup
"""

__all__: list[str] = []
