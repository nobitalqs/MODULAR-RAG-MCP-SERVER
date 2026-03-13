"""Tests for MarkdownLoader image extraction feature."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.libs.loader.markdown_loader import MarkdownLoader


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """Create a temporary image storage directory."""
    d = tmp_path / "image_store"
    d.mkdir()
    return d


@pytest.fixture()
def md_with_local_image(tmp_path: Path) -> Path:
    """Create a markdown file with a local image reference."""
    # Create the image file
    img = tmp_path / "screenshot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal PNG-like bytes

    # Create the markdown file
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nSome text ![diagram](./screenshot.png) more text\n")
    return md


class TestExtractImagesLocal:
    """Test image extraction with local files."""

    def test_local_relative_image_extracted(self, md_with_local_image, image_dir):
        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md_with_local_image)

        # Text should contain [IMAGE: id] placeholder, not ![...]()
        assert "![diagram]" not in doc.text
        assert "[IMAGE:" in doc.text

        # metadata should have images list
        assert "images" in doc.metadata
        assert len(doc.metadata["images"]) == 1

        img_meta = doc.metadata["images"][0]
        assert img_meta["id"].endswith("_md_1")
        assert img_meta["alt_text"] == "diagram"
        assert img_meta["original_ref"] == "./screenshot.png"

        # File should be copied to image_dir/{doc_hash}/
        doc_hash = doc.metadata["doc_hash"]
        copied_file = image_dir / doc_hash / Path(img_meta["path"]).name
        assert copied_file.exists()

    def test_copied_file_has_correct_extension(self, md_with_local_image, image_dir):
        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md_with_local_image)

        img_meta = doc.metadata["images"][0]
        assert img_meta["path"].endswith(".png")


class TestGenerateImageId:
    """Test image ID generation."""

    def test_format(self):
        result = MarkdownLoader._generate_image_id("abcdef1234567890", 1)
        assert result == "abcdef12_md_1"

    def test_sequence_increments(self):
        id1 = MarkdownLoader._generate_image_id("abcdef1234567890", 1)
        id2 = MarkdownLoader._generate_image_id("abcdef1234567890", 2)
        assert id1 == "abcdef12_md_1"
        assert id2 == "abcdef12_md_2"

    def test_different_hash_different_id(self):
        id1 = MarkdownLoader._generate_image_id("aaaa0000", 1)
        id2 = MarkdownLoader._generate_image_id("bbbb1111", 1)
        assert id1 != id2


class TestExtractImagesUrl:
    """URL images should be preserved as-is."""

    def test_url_image_preserved(self, tmp_path, image_dir):
        md = tmp_path / "doc.md"
        md.write_text("![logo](https://example.com/logo.png)\n")

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md)

        assert "![logo](https://example.com/logo.png)" in doc.text
        assert "images" not in doc.metadata

    def test_http_url_also_preserved(self, tmp_path, image_dir):
        md = tmp_path / "doc.md"
        md.write_text("![pic](http://example.com/pic.jpg)\n")

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md)

        assert "![pic](http://example.com/pic.jpg)" in doc.text
        assert "images" not in doc.metadata


class TestExtractImagesMissing:
    """Missing local images should be preserved with warning."""

    def test_missing_image_preserved(self, tmp_path, image_dir, caplog):
        md = tmp_path / "doc.md"
        md.write_text("![chart](./nonexistent.png)\n")

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))

        import logging
        with caplog.at_level(logging.WARNING):
            doc = loader.load(md)

        assert "![chart](./nonexistent.png)" in doc.text
        assert "images" not in doc.metadata
        assert "not found" in caplog.text.lower()


class TestExtractImagesMixed:
    """Mixed image references: local, URL, missing."""

    def test_mixed_references(self, tmp_path, image_dir):
        # Create one real image
        img = tmp_path / "real.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)  # JPEG-like bytes

        md = tmp_path / "doc.md"
        md.write_text(
            "# Doc\n"
            "Local: ![photo](./real.jpg)\n"
            "URL: ![logo](https://example.com/logo.png)\n"
            "Missing: ![gone](./deleted.png)\n"
        )

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md)

        # Local image → replaced
        assert "[IMAGE:" in doc.text
        assert "![photo]" not in doc.text

        # URL → preserved
        assert "![logo](https://example.com/logo.png)" in doc.text

        # Missing → preserved
        assert "![gone](./deleted.png)" in doc.text

        # Only 1 image in metadata
        assert len(doc.metadata["images"]) == 1
        assert doc.metadata["images"][0]["alt_text"] == "photo"
        assert doc.metadata["images"][0]["path"].endswith(".jpg")


class TestExtractImagesDisabled:
    """All images preserved when extraction is disabled."""

    def test_disabled_preserves_all(self, tmp_path):
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)

        md = tmp_path / "doc.md"
        md.write_text("![pic](./pic.png)\n")

        loader = MarkdownLoader(extract_images=False)
        doc = loader.load(md)

        assert "![pic](./pic.png)" in doc.text
        assert "images" not in doc.metadata


class TestExtractImagesNoImages:
    """Markdown without images should be unaffected."""

    def test_no_images_no_metadata_key(self, tmp_path, image_dir):
        md = tmp_path / "doc.md"
        md.write_text("# Plain\n\nJust text, no images.\n")

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md)

        assert "images" not in doc.metadata
        assert "Just text, no images." in doc.text


class TestExtractImagesEdgeCases:
    """Edge cases for image extraction."""

    def test_empty_alt_text(self, tmp_path, image_dir):
        img = tmp_path / "fig.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)

        md = tmp_path / "doc.md"
        md.write_text("![](./fig.png)\n")

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md)

        assert "[IMAGE:" in doc.text
        assert doc.metadata["images"][0]["alt_text"] == ""

    def test_duplicate_image_refs(self, tmp_path, image_dir):
        img = tmp_path / "same.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)

        md = tmp_path / "doc.md"
        md.write_text("![a](./same.png) text ![b](./same.png)\n")

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md)

        assert len(doc.metadata["images"]) == 2
        assert doc.metadata["images"][0]["id"] != doc.metadata["images"][1]["id"]
        assert doc.metadata["images"][0]["id"].endswith("_md_1")
        assert doc.metadata["images"][1]["id"].endswith("_md_2")

    def test_different_extensions(self, tmp_path, image_dir):
        for name in ("a.png", "b.jpg", "c.gif"):
            (tmp_path / name).write_bytes(b"\x00" * 50)

        md = tmp_path / "doc.md"
        md.write_text("![](./a.png)\n![](./b.jpg)\n![](./c.gif)\n")

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md)

        paths = [m["path"] for m in doc.metadata["images"]]
        assert any(p.endswith(".png") for p in paths)
        assert any(p.endswith(".jpg") for p in paths)
        assert any(p.endswith(".gif") for p in paths)

    def test_absolute_path_image(self, tmp_path, image_dir):
        img = tmp_path / "abs_img.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)

        md = tmp_path / "doc.md"
        md.write_text(f"![abs]({img})\n")  # absolute path

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md)

        assert "[IMAGE:" in doc.text
        assert len(doc.metadata["images"]) == 1


class TestExtractImagesWithFrontmatter:
    """Frontmatter should be parsed correctly alongside image extraction."""

    def test_frontmatter_and_images(self, tmp_path, image_dir):
        img = tmp_path / "hero.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)

        md = tmp_path / "doc.md"
        md.write_text(
            "---\ntitle: My Doc\ntags: [test]\n---\n\n"
            "# Heading\n\n![hero](./hero.png)\n"
        )

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))
        doc = loader.load(md)

        # Frontmatter parsed
        assert doc.metadata["title"] == "My Doc"
        assert doc.metadata["tags"] == ["test"]

        # Image extracted
        assert "[IMAGE:" in doc.text
        assert len(doc.metadata["images"]) == 1


class TestExtractImagesDegradation:
    """Graceful degradation when image copy fails."""

    def test_copy_failure_preserves_original(self, tmp_path, image_dir, caplog):
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)

        md = tmp_path / "doc.md"
        md.write_text("![pic](./pic.png)\n")

        loader = MarkdownLoader(extract_images=True, image_storage_dir=str(image_dir))

        import logging
        with (
            patch("src.libs.loader.markdown_loader.shutil.copy2", side_effect=PermissionError("denied")),
            caplog.at_level(logging.WARNING),
        ):
            doc = loader.load(md)

        # Original syntax preserved
        assert "![pic](./pic.png)" in doc.text
        assert "images" not in doc.metadata
        assert "denied" in caplog.text
