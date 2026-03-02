"""Contract tests for ImageCaptioner transform.

Tests cover:
- Disabled mode (no Vision LLM) → chunks returned with has_unprocessed_images
- Enabled mode → captions generated and written to metadata + text
- Graceful degradation on Vision LLM failure
- Caption caching (same image_id → single API call)
- Trace recording
- Edge cases: empty list, no image refs, missing image path
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.libs.llm.base_llm import ChatResponse
from src.libs.llm.base_vision_llm import BaseVisionLLM


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings_vision_disabled():
    """Settings with Vision LLM disabled."""
    settings = Mock(spec=Settings)
    settings.vision_llm = Mock()
    settings.vision_llm.enabled = False
    return settings


@pytest.fixture
def mock_settings_vision_enabled():
    """Settings with Vision LLM enabled."""
    settings = Mock(spec=Settings)
    settings.vision_llm = Mock()
    settings.vision_llm.enabled = True
    return settings


@pytest.fixture
def mock_settings_no_vision():
    """Settings without vision_llm section."""
    settings = Mock(spec=Settings)
    settings.vision_llm = None
    return settings


@pytest.fixture
def mock_vision_llm():
    """Mock Vision LLM that returns captions."""
    llm = Mock(spec=BaseVisionLLM)
    llm.chat_with_image.return_value = ChatResponse(
        content="A diagram showing data flow",
        model="gpt-4o",
        usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    )
    return llm


@pytest.fixture
def chunk_with_image():
    """Chunk with an image placeholder and image metadata."""
    return Chunk(
        id="chunk-img-1",
        text="See the architecture diagram:\n[IMAGE: img_001]\nThis shows the system.",
        metadata={
            "source_path": "/test/doc.pdf",
            "images": [{"id": "img_001", "path": "/tmp/img_001.png"}],
        },
        source_ref="doc-1",
    )


@pytest.fixture
def chunk_without_image():
    """Chunk with no image references."""
    return Chunk(
        id="chunk-text-1",
        text="This is a plain text chunk without images.",
        metadata={"source_path": "/test/doc.pdf"},
        source_ref="doc-1",
    )


@pytest.fixture
def chunk_multi_image():
    """Chunk with multiple image placeholders."""
    return Chunk(
        id="chunk-img-multi",
        text="First image: [IMAGE: img_a]\nSecond image: [IMAGE: img_b]",
        metadata={
            "source_path": "/test/doc.pdf",
            "images": [
                {"id": "img_a", "path": "/tmp/img_a.png"},
                {"id": "img_b", "path": "/tmp/img_b.png"},
            ],
        },
        source_ref="doc-1",
    )


# ── BaseTransform contract ────────────────────────────────────────────

class TestBaseContract:
    def test_implements_base_transform(self, mock_settings_vision_disabled):
        captioner = ImageCaptioner(mock_settings_vision_disabled)
        assert isinstance(captioner, BaseTransform)

    def test_transform_empty_list(self, mock_settings_vision_disabled):
        captioner = ImageCaptioner(mock_settings_vision_disabled)
        assert captioner.transform([]) == []


# ── Disabled / degradation mode ───────────────────────────────────────

class TestDisabledMode:
    def test_disabled_returns_chunks_with_unprocessed_flag(
        self, mock_settings_vision_disabled, chunk_with_image
    ):
        """When vision disabled, chunks with images get has_unprocessed_images."""
        captioner = ImageCaptioner(mock_settings_vision_disabled)
        result = captioner.transform([chunk_with_image])

        assert len(result) == 1
        assert result[0].metadata["has_unprocessed_images"] is True
        assert "image_captions" not in result[0].metadata

    def test_disabled_no_image_chunks_unchanged(
        self, mock_settings_vision_disabled, chunk_without_image
    ):
        """Chunks without image refs should be unchanged even in disabled mode."""
        captioner = ImageCaptioner(mock_settings_vision_disabled)
        result = captioner.transform([chunk_without_image])

        assert len(result) == 1
        assert "has_unprocessed_images" not in result[0].metadata

    def test_no_vision_settings_degrades(
        self, mock_settings_no_vision, chunk_with_image
    ):
        """When vision_llm is None in settings, should degrade gracefully."""
        captioner = ImageCaptioner(mock_settings_no_vision)
        result = captioner.transform([chunk_with_image])

        assert len(result) == 1
        assert result[0].metadata["has_unprocessed_images"] is True


# ── Enabled mode — caption success ────────────────────────────────────

class TestCaptionSuccess:
    def test_caption_added_to_metadata(
        self, mock_settings_vision_enabled, mock_vision_llm, chunk_with_image
    ):
        """Should add image_captions list to metadata."""
        captioner = ImageCaptioner(
            mock_settings_vision_enabled, vision_llm=mock_vision_llm
        )
        result = captioner.transform([chunk_with_image])

        assert len(result) == 1
        captions = result[0].metadata.get("image_captions", [])
        assert len(captions) == 1
        assert captions[0]["id"] == "img_001"
        assert captions[0]["caption"] == "A diagram showing data flow"

    def test_caption_injected_into_text(
        self, mock_settings_vision_enabled, mock_vision_llm, chunk_with_image
    ):
        """Should inject caption description after [IMAGE: id] placeholder."""
        captioner = ImageCaptioner(
            mock_settings_vision_enabled, vision_llm=mock_vision_llm
        )
        result = captioner.transform([chunk_with_image])

        assert "[IMAGE: img_001]" in result[0].text
        assert "(Description: A diagram showing data flow)" in result[0].text

    def test_text_preserved_for_no_image_chunk(
        self, mock_settings_vision_enabled, mock_vision_llm, chunk_without_image
    ):
        """Chunks without image refs should be unchanged."""
        captioner = ImageCaptioner(
            mock_settings_vision_enabled, vision_llm=mock_vision_llm
        )
        result = captioner.transform([chunk_without_image])

        assert result[0].text == chunk_without_image.text
        assert "image_captions" not in result[0].metadata

    def test_multiple_images_captioned(
        self, mock_settings_vision_enabled, chunk_multi_image
    ):
        """Should caption all images in a chunk."""
        llm = Mock(spec=BaseVisionLLM)
        llm.chat_with_image.side_effect = [
            ChatResponse(content="Caption A", model="m", usage={}),
            ChatResponse(content="Caption B", model="m", usage={}),
        ]
        captioner = ImageCaptioner(
            mock_settings_vision_enabled, vision_llm=llm
        )
        result = captioner.transform([chunk_multi_image])

        captions = result[0].metadata["image_captions"]
        assert len(captions) == 2
        assert {c["id"] for c in captions} == {"img_a", "img_b"}
        assert llm.chat_with_image.call_count == 2

    def test_existing_metadata_preserved(
        self, mock_settings_vision_enabled, mock_vision_llm, chunk_with_image
    ):
        """Should preserve existing metadata fields."""
        captioner = ImageCaptioner(
            mock_settings_vision_enabled, vision_llm=mock_vision_llm
        )
        result = captioner.transform([chunk_with_image])

        assert result[0].metadata["source_path"] == "/test/doc.pdf"
        assert result[0].source_ref == "doc-1"


# ── Caption cache ─────────────────────────────────────────────────────

class TestCaptionCache:
    def test_same_image_captioned_once(self, mock_settings_vision_enabled, mock_vision_llm):
        """Same image_id referenced in two chunks → one API call."""
        chunks = [
            Chunk(
                id="c1",
                text="First ref: [IMAGE: shared_img]",
                metadata={
                    "source_path": "/test.pdf",
                    "images": [{"id": "shared_img", "path": "/tmp/shared.png"}],
                },
            ),
            Chunk(
                id="c2",
                text="Second ref: [IMAGE: shared_img]",
                metadata={
                    "source_path": "/test.pdf",
                    "images": [{"id": "shared_img", "path": "/tmp/shared.png"}],
                },
            ),
        ]
        captioner = ImageCaptioner(
            mock_settings_vision_enabled, vision_llm=mock_vision_llm
        )
        result = captioner.transform(chunks)

        # Both chunks should have caption
        assert result[0].metadata["image_captions"][0]["caption"] == "A diagram showing data flow"
        assert result[1].metadata["image_captions"][0]["caption"] == "A diagram showing data flow"
        # But only ONE API call
        assert mock_vision_llm.chat_with_image.call_count == 1


# ── Failure / degradation ─────────────────────────────────────────────

class TestFailureDegradation:
    def test_llm_failure_marks_unprocessed(
        self, mock_settings_vision_enabled, chunk_with_image
    ):
        """Should mark has_unprocessed_images when Vision LLM fails."""
        llm = Mock(spec=BaseVisionLLM)
        llm.chat_with_image.side_effect = RuntimeError("Vision API down")

        captioner = ImageCaptioner(
            mock_settings_vision_enabled, vision_llm=llm
        )
        result = captioner.transform([chunk_with_image])

        assert len(result) == 1
        assert result[0].metadata["has_unprocessed_images"] is True
        assert "image_captions" not in result[0].metadata

    def test_missing_image_path_skips(
        self, mock_settings_vision_enabled, mock_vision_llm
    ):
        """Image ref in text but no path in metadata → skip, mark unprocessed."""
        chunk = Chunk(
            id="chunk-no-path",
            text="See [IMAGE: orphan_img] here.",
            metadata={"source_path": "/test.pdf"},
        )
        captioner = ImageCaptioner(
            mock_settings_vision_enabled, vision_llm=mock_vision_llm
        )
        result = captioner.transform([chunk])

        assert result[0].metadata["has_unprocessed_images"] is True
        mock_vision_llm.chat_with_image.assert_not_called()


# ── Trace recording ───────────────────────────────────────────────────

class TestTraceRecording:
    def test_trace_recorded_on_success(
        self, mock_settings_vision_enabled, mock_vision_llm, chunk_with_image
    ):
        captioner = ImageCaptioner(
            mock_settings_vision_enabled, vision_llm=mock_vision_llm
        )
        trace = TraceContext(trace_type="ingestion")
        captioner.transform([chunk_with_image], trace=trace)

        stage = trace.get_stage_data("image_captioner")
        assert stage is not None
        assert stage["total_chunks"] == 1
        assert stage["captions_generated"] == 1

    def test_trace_recorded_on_disabled(
        self, mock_settings_vision_disabled, chunk_with_image
    ):
        captioner = ImageCaptioner(mock_settings_vision_disabled)
        trace = TraceContext(trace_type="ingestion")
        captioner.transform([chunk_with_image], trace=trace)

        stage = trace.get_stage_data("image_captioner")
        assert stage is not None
        assert stage["enabled"] is False
