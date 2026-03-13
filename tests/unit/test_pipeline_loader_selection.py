"""Tests for pipeline LoaderFactory integration — extension routing + unsupported error."""

from __future__ import annotations

import pytest

from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.source_code_loader import SourceCodeLoader


def _build_factory() -> LoaderFactory:
    """Build a LoaderFactory with all Phase L extensions registered.

    Mirrors the setup done in IngestionPipeline.__init__.
    """
    factory = LoaderFactory()
    factory.register_provider(".pdf", PdfLoader)
    factory.register_provider(".md", MarkdownLoader)
    factory.register_provider(".markdown", MarkdownLoader)
    factory.register_provider(".c", SourceCodeLoader)
    factory.register_provider(".cpp", SourceCodeLoader)
    factory.register_provider(".cxx", SourceCodeLoader)
    factory.register_provider(".cc", SourceCodeLoader)
    factory.register_provider(".h", SourceCodeLoader)
    factory.register_provider(".hxx", SourceCodeLoader)
    factory.register_provider(".py", SourceCodeLoader)
    return factory


class TestPipelineLoaderSelection:
    """Verify that the factory routes each extension to the correct loader."""

    @pytest.mark.parametrize("filename", ["report.pdf", "REPORT.PDF"])
    def test_pdf_routed(self, filename):
        factory = _build_factory()
        loader = factory.create_for_file(filename)
        assert isinstance(loader, PdfLoader)

    @pytest.mark.parametrize("filename", ["notes.md", "doc.markdown"])
    def test_markdown_routed(self, filename):
        factory = _build_factory()
        loader = factory.create_for_file(filename)
        assert isinstance(loader, MarkdownLoader)

    @pytest.mark.parametrize(
        "filename",
        ["main.c", "main.cpp", "main.cxx", "main.cc", "foo.h", "bar.hxx", "script.py"],
    )
    def test_source_code_routed(self, filename):
        factory = _build_factory()
        loader = factory.create_for_file(filename)
        assert isinstance(loader, SourceCodeLoader)

    def test_unsupported_extension_raises(self):
        factory = _build_factory()
        with pytest.raises(ValueError, match=r"\.docx"):
            factory.create_for_file("document.docx")

    def test_pdf_kwargs_forwarded(self):
        """PdfLoader receives kwargs like extract_images from factory."""
        factory = _build_factory()
        loader = factory.create_for_file(
            "report.pdf",
            extract_images=True,
            image_storage_dir="data/images",
        )
        assert isinstance(loader, PdfLoader)
        assert loader.extract_images is True

    def test_markdown_ignores_pdf_kwargs(self):
        """MarkdownLoader's **kwargs absorbs irrelevant params without error."""
        factory = _build_factory()
        loader = factory.create_for_file(
            "notes.md",
            extract_images=True,
            image_storage_dir="data/images",
        )
        assert isinstance(loader, MarkdownLoader)

    def test_source_code_ignores_pdf_kwargs(self):
        """SourceCodeLoader's **kwargs absorbs irrelevant params without error."""
        factory = _build_factory()
        loader = factory.create_for_file(
            "main.cpp",
            extract_images=True,
            table_extraction=None,
        )
        assert isinstance(loader, SourceCodeLoader)
