"""Tests for MarkdownLoader — frontmatter parsing, title extraction, Document contract."""

from __future__ import annotations

import pytest

from src.core.types import Document
from src.libs.loader.markdown_loader import MarkdownLoader


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def loader():
    return MarkdownLoader()


@pytest.fixture()
def md_with_frontmatter(tmp_path):
    content = """\
---
title: My Document
author: Alice
tags:
  - python
  - rag
---

# Introduction

Hello world.
"""
    path = tmp_path / "doc.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def md_no_frontmatter(tmp_path):
    content = """\
# Plain Heading

Some text here.
"""
    path = tmp_path / "plain.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def md_malformed_frontmatter(tmp_path):
    content = """\
---
: invalid yaml [[[
---

Body text.
"""
    path = tmp_path / "bad_fm.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def md_no_heading(tmp_path):
    content = "Just some text without any heading.\n"
    path = tmp_path / "nohead.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def md_with_images(tmp_path):
    content = """\
# Images Test

![alt text](image.png)

Some text after image.

![another](http://example.com/pic.jpg)
"""
    path = tmp_path / "images.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def md_reserved_keys(tmp_path):
    """Frontmatter that tries to overwrite reserved keys."""
    content = """\
---
title: OK Title
source_path: /hacked/path
doc_type: hacked_type
doc_hash: fakehash
custom_key: custom_value
---

Content here.
"""
    path = tmp_path / "reserved.md"
    path.write_text(content, encoding="utf-8")
    return path


# ── Tests ──────────────────────────────────────────────────────────


class TestMarkdownLoaderFrontmatter:
    """Frontmatter extraction and metadata merging."""

    def test_frontmatter_parsed(self, loader, md_with_frontmatter):
        doc = loader.load(md_with_frontmatter)
        assert doc.metadata["title"] == "My Document"
        assert doc.metadata["author"] == "Alice"
        assert doc.metadata["tags"] == ["python", "rag"]

    def test_no_frontmatter(self, loader, md_no_frontmatter):
        doc = loader.load(md_no_frontmatter)
        # Should not crash; title from heading
        assert doc.metadata["title"] == "Plain Heading"
        assert "author" not in doc.metadata

    def test_malformed_frontmatter_graceful(self, loader, md_malformed_frontmatter):
        """Malformed YAML in frontmatter should not crash, just skip it."""
        doc = loader.load(md_malformed_frontmatter)
        assert isinstance(doc, Document)
        assert "Body text." in doc.text

    def test_reserved_keys_protected(self, loader, md_reserved_keys):
        """Reserved keys (source_path, doc_type, doc_hash) must not be overwritten."""
        doc = loader.load(md_reserved_keys)
        # source_path should be the actual file path, not the frontmatter value
        assert doc.metadata["source_path"] != "/hacked/path"
        assert doc.metadata["doc_type"] == "markdown"
        assert doc.metadata["doc_hash"] != "fakehash"
        # Non-reserved custom keys should be merged
        assert doc.metadata["custom_key"] == "custom_value"
        assert doc.metadata["title"] == "OK Title"


class TestMarkdownLoaderTitle:
    """Title extraction fallback chain: frontmatter → heading → filename."""

    def test_title_from_frontmatter(self, loader, md_with_frontmatter):
        doc = loader.load(md_with_frontmatter)
        assert doc.metadata["title"] == "My Document"

    def test_title_from_heading(self, loader, md_no_frontmatter):
        doc = loader.load(md_no_frontmatter)
        assert doc.metadata["title"] == "Plain Heading"

    def test_title_from_filename(self, loader, md_no_heading):
        doc = loader.load(md_no_heading)
        assert doc.metadata["title"] == "nohead"


class TestMarkdownLoaderContract:
    """Document contract: required fields and types."""

    def test_source_path_in_metadata(self, loader, md_with_frontmatter):
        doc = loader.load(md_with_frontmatter)
        assert "source_path" in doc.metadata
        assert str(md_with_frontmatter) in doc.metadata["source_path"]

    def test_doc_type_is_markdown(self, loader, md_with_frontmatter):
        doc = loader.load(md_with_frontmatter)
        assert doc.metadata["doc_type"] == "markdown"

    def test_doc_id_format(self, loader, md_with_frontmatter):
        doc = loader.load(md_with_frontmatter)
        assert doc.id.startswith("doc_")
        assert len(doc.id) == 20  # "doc_" + 16 hex chars

    def test_doc_hash_present(self, loader, md_with_frontmatter):
        doc = loader.load(md_with_frontmatter)
        assert "doc_hash" in doc.metadata
        assert len(doc.metadata["doc_hash"]) == 64  # SHA256 hex

    def test_text_is_nonempty(self, loader, md_with_frontmatter):
        doc = loader.load(md_with_frontmatter)
        assert doc.text.strip()


class TestMarkdownLoaderValidation:
    """Extension validation and error handling."""

    def test_wrong_extension_raises(self, loader, tmp_path):
        path = tmp_path / "file.txt"
        path.write_text("hello")
        with pytest.raises(ValueError, match="not a Markdown file"):
            loader.load(path)

    def test_file_not_found(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.md")

    def test_markdown_extension_accepted(self, loader, tmp_path):
        path = tmp_path / "doc.markdown"
        path.write_text("# Hello\n\nWorld.")
        doc = loader.load(path)
        assert isinstance(doc, Document)


class TestMarkdownLoaderImages:
    """Image references are preserved as-is."""

    def test_images_kept_as_is(self, loader, md_with_images):
        doc = loader.load(md_with_images)
        assert "![alt text](image.png)" in doc.text
        assert "![another](http://example.com/pic.jpg)" in doc.text


class TestMarkdownLoaderKwargs:
    """LoaderFactory compatibility — accepts **kwargs."""

    def test_accepts_kwargs(self):
        loader = MarkdownLoader(extract_images=True, image_storage_dir="data/images")
        assert isinstance(loader, MarkdownLoader)
