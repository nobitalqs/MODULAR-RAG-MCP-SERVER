"""Tests for MultimodalAssembler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.response.multimodal_assembler import MultimodalAssembler


@dataclass
class FakeResult:
    """Minimal RetrievalResult stand-in."""

    chunk_id: str = "chunk_1"
    score: float = 0.9
    text: str = "some text"
    metadata: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def png_file(tmp_path: Path) -> Path:
    """Create a minimal valid PNG file."""
    # Minimal 1x1 PNG (67 bytes)
    import base64

    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "nGNgYPgPAAEDAQAIicLsAAAABJRU5ErkJggg=="
    )
    p = tmp_path / "test_image.png"
    p.write_bytes(base64.b64decode(png_b64))
    return p


@pytest.fixture
def jpeg_file(tmp_path: Path) -> Path:
    """Create a fake JPEG file."""
    p = tmp_path / "test_image.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    return p


class TestAssembleFromImagesMeta:
    """Assemble using images metadata in result."""

    def test_returns_image_content_for_valid_png(self, png_file: Path):
        result = FakeResult(
            metadata={
                "images": [{"id": "img_001", "path": str(png_file)}],
            }
        )
        assembler = MultimodalAssembler()
        contents = assembler.assemble([result])
        assert len(contents) == 1
        assert contents[0].type == "image"
        assert contents[0].mimeType == "image/png"
        assert len(contents[0].data) > 0  # Base64 encoded

    def test_returns_image_content_for_jpeg(self, jpeg_file: Path):
        result = FakeResult(
            metadata={
                "images": [{"id": "img_002", "path": str(jpeg_file)}],
            }
        )
        assembler = MultimodalAssembler()
        contents = assembler.assemble([result])
        assert len(contents) == 1
        assert contents[0].mimeType == "image/jpeg"

    def test_skips_missing_file(self):
        result = FakeResult(
            metadata={
                "images": [{"id": "img_003", "path": "/nonexistent/image.png"}],
            }
        )
        assembler = MultimodalAssembler()
        contents = assembler.assemble([result])
        assert len(contents) == 0

    def test_skips_unsupported_format(self, tmp_path: Path):
        tiff = tmp_path / "image.tiff"
        tiff.write_bytes(b"\x00" * 100)
        result = FakeResult(
            metadata={
                "images": [{"id": "img_004", "path": str(tiff)}],
            }
        )
        assembler = MultimodalAssembler()
        contents = assembler.assemble([result])
        assert len(contents) == 0


class TestAssembleFromImageRefs:
    """Assemble using image_refs + ImageStorage lookup."""

    def test_resolves_via_image_storage(self, png_file: Path):
        storage = MagicMock()
        storage.get_image_path.return_value = str(png_file)

        result = FakeResult(metadata={"image_refs": ["img_010"]})
        assembler = MultimodalAssembler(image_storage=storage)
        contents = assembler.assemble([result])

        storage.get_image_path.assert_called_once_with("img_010")
        assert len(contents) == 1

    def test_skips_when_storage_returns_none(self):
        storage = MagicMock()
        storage.get_image_path.return_value = None

        result = FakeResult(metadata={"image_refs": ["img_011"]})
        assembler = MultimodalAssembler(image_storage=storage)
        contents = assembler.assemble([result])
        assert len(contents) == 0

    def test_no_storage_skips_image_refs(self):
        result = FakeResult(metadata={"image_refs": ["img_012"]})
        assembler = MultimodalAssembler(image_storage=None)
        contents = assembler.assemble([result])
        assert len(contents) == 0


class TestLimitsAndDedup:
    """Max images and deduplication."""

    def test_max_images_respected(self, png_file: Path):
        images = [{"id": f"img_{i:03d}", "path": str(png_file)} for i in range(10)]
        result = FakeResult(metadata={"images": images})
        assembler = MultimodalAssembler(max_images=3)
        contents = assembler.assemble([result])
        assert len(contents) == 3

    def test_dedup_across_results(self, png_file: Path):
        r1 = FakeResult(metadata={"images": [{"id": "img_dup", "path": str(png_file)}]})
        r2 = FakeResult(metadata={"images": [{"id": "img_dup", "path": str(png_file)}]})
        assembler = MultimodalAssembler()
        contents = assembler.assemble([r1, r2])
        assert len(contents) == 1

    def test_large_file_skipped(self, tmp_path: Path):
        big = tmp_path / "big.png"
        big.write_bytes(b"\x89PNG" + b"\x00" * 100)
        result = FakeResult(
            metadata={
                "images": [{"id": "img_big", "path": str(big)}],
            }
        )
        assembler = MultimodalAssembler(max_image_size_bytes=50)
        contents = assembler.assemble([result])
        assert len(contents) == 0


class TestEdgeCases:
    """Empty inputs and missing metadata."""

    def test_empty_results(self):
        assembler = MultimodalAssembler()
        assert assembler.assemble([]) == []

    def test_no_image_metadata(self):
        result = FakeResult(metadata={"source_path": "doc.pdf"})
        assembler = MultimodalAssembler()
        assert assembler.assemble([result]) == []

    def test_none_metadata(self):
        result = FakeResult(metadata={})
        assembler = MultimodalAssembler()
        assert assembler.assemble([result]) == []
