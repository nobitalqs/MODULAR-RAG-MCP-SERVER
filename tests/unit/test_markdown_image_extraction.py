"""Tests for MarkdownLoader image extraction feature."""

from __future__ import annotations

from src.libs.loader.markdown_loader import MarkdownLoader


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
