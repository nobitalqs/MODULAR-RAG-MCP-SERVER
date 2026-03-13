"""Tests for LoaderFactory — extension-based routing with registry pattern."""

from __future__ import annotations

import pytest

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.loader_factory import LoaderFactory


# ── Stub loaders for testing ───────────────────────────────────────


class StubPdfLoader(BaseLoader):
    """Minimal loader that records kwargs for verification."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load(self, file_path):
        return Document(
            id="stub_pdf",
            text="pdf content",
            metadata={"source_path": str(file_path)},
        )


class StubMarkdownLoader(BaseLoader):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load(self, file_path):
        return Document(
            id="stub_md",
            text="md content",
            metadata={"source_path": str(file_path)},
        )


# ── Tests ──────────────────────────────────────────────────────────


class TestLoaderFactory:
    """LoaderFactory registration, routing, and error handling."""

    def test_register_and_create_pdf(self):
        """Registered .pdf extension returns correct loader class."""
        factory = LoaderFactory()
        factory.register_provider(".pdf", StubPdfLoader)

        loader = factory.create_for_file("report.pdf")
        assert isinstance(loader, StubPdfLoader)

    def test_register_and_create_markdown(self):
        """Registered .md extension returns correct loader class."""
        factory = LoaderFactory()
        factory.register_provider(".md", StubMarkdownLoader)

        loader = factory.create_for_file("notes.md")
        assert isinstance(loader, StubMarkdownLoader)

    def test_case_insensitive_extension(self):
        """Extensions are matched case-insensitively."""
        factory = LoaderFactory()
        factory.register_provider(".pdf", StubPdfLoader)

        loader = factory.create_for_file("REPORT.PDF")
        assert isinstance(loader, StubPdfLoader)

    def test_unknown_extension_raises_value_error(self):
        """Unregistered extension raises ValueError with helpful message."""
        factory = LoaderFactory()
        factory.register_provider(".pdf", StubPdfLoader)

        with pytest.raises(ValueError, match=r"\.xyz"):
            factory.create_for_file("data.xyz")

    def test_kwargs_passthrough(self):
        """kwargs are forwarded to the loader constructor."""
        factory = LoaderFactory()
        factory.register_provider(".pdf", StubPdfLoader)

        loader = factory.create_for_file(
            "report.pdf",
            extract_images=True,
            image_storage_dir="data/images",
        )
        assert loader.kwargs["extract_images"] is True
        assert loader.kwargs["image_storage_dir"] == "data/images"

    def test_multiple_extensions_same_loader(self):
        """Multiple extensions can map to the same loader class."""
        factory = LoaderFactory()
        factory.register_provider(".md", StubMarkdownLoader)
        factory.register_provider(".markdown", StubMarkdownLoader)

        loader_md = factory.create_for_file("a.md")
        loader_markdown = factory.create_for_file("b.markdown")
        assert isinstance(loader_md, StubMarkdownLoader)
        assert isinstance(loader_markdown, StubMarkdownLoader)

    def test_pathlib_path_input(self, tmp_path):
        """Accepts pathlib.Path as well as str."""
        factory = LoaderFactory()
        factory.register_provider(".pdf", StubPdfLoader)

        loader = factory.create_for_file(tmp_path / "report.pdf")
        assert isinstance(loader, StubPdfLoader)

    def test_no_extension_raises(self):
        """File with no extension raises ValueError."""
        factory = LoaderFactory()
        factory.register_provider(".pdf", StubPdfLoader)

        with pytest.raises(ValueError):
            factory.create_for_file("Makefile")
