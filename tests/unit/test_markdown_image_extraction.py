"""Tests for MarkdownLoader image extraction feature."""

from __future__ import annotations

from pathlib import Path

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
