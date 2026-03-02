"""Tests for B8: Vision LLM abstract interface and factory integration.

Tests cover:
- ImageInput dataclass validation
- BaseVisionLLM ABC contract (validate_text, validate_image, preprocess_image)
- FakeVisionLLM concrete implementation
- LLMFactory vision provider registry (register_vision_provider, create_vision_llm)
- LLMFactory.create_vision_llm_from_settings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory


# ── Fixtures ────────────────────────────────────────────────────────────


class FakeVisionLLM(BaseVisionLLM):
    """Minimal concrete VisionLLM for testing."""

    def __init__(self, model: str = "fake-vision", **kwargs: Any) -> None:
        self.model = model
        self.config = kwargs
        self.call_count = 0

    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: list[Message] | None = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.validate_text(text)
        self.validate_image(image)
        self.call_count += 1
        return ChatResponse(
            content=f"Described: {text}",
            model=self.model,
        )


# ── ImageInput Tests ────────────────────────────────────────────────────


class TestImageInput:
    """Tests for ImageInput dataclass validation."""

    def test_path_only(self) -> None:
        img = ImageInput(path="/tmp/test.png")
        assert img.path == "/tmp/test.png"
        assert img.data is None
        assert img.base64 is None

    def test_data_only(self) -> None:
        img = ImageInput(data=b"\x89PNG")
        assert img.data == b"\x89PNG"
        assert img.path is None

    def test_base64_only(self) -> None:
        img = ImageInput(base64="iVBORw0KGgo=")
        assert img.base64 == "iVBORw0KGgo="

    def test_custom_mime_type(self) -> None:
        img = ImageInput(path="photo.jpg", mime_type="image/jpeg")
        assert img.mime_type == "image/jpeg"

    def test_default_mime_type(self) -> None:
        img = ImageInput(path="test.png")
        assert img.mime_type == "image/png"

    def test_no_input_raises(self) -> None:
        with pytest.raises(ValueError, match="Must provide one of"):
            ImageInput()

    def test_multiple_inputs_raises(self) -> None:
        with pytest.raises(ValueError, match="Must provide exactly one"):
            ImageInput(path="/tmp/x.png", data=b"bytes")

    def test_all_three_raises(self) -> None:
        with pytest.raises(ValueError, match="Must provide exactly one"):
            ImageInput(path="/tmp/x.png", data=b"bytes", base64="abc")

    def test_path_and_base64_raises(self) -> None:
        with pytest.raises(ValueError, match="Must provide exactly one"):
            ImageInput(path="/tmp/x.png", base64="abc")


# ── BaseVisionLLM Validation Tests ──────────────────────────────────────


class TestBaseVisionLLMValidation:
    """Tests for validate_text and validate_image."""

    @pytest.fixture()
    def vlm(self) -> FakeVisionLLM:
        return FakeVisionLLM()

    def test_validate_text_valid(self, vlm: FakeVisionLLM) -> None:
        vlm.validate_text("Describe this image")  # should not raise

    def test_validate_text_empty_raises(self, vlm: FakeVisionLLM) -> None:
        with pytest.raises(ValueError, match="empty"):
            vlm.validate_text("")

    def test_validate_text_whitespace_raises(self, vlm: FakeVisionLLM) -> None:
        with pytest.raises(ValueError, match="empty"):
            vlm.validate_text("   ")

    def test_validate_text_non_string_raises(self, vlm: FakeVisionLLM) -> None:
        with pytest.raises(ValueError, match="string"):
            vlm.validate_text(123)  # type: ignore[arg-type]

    def test_validate_image_valid(self, vlm: FakeVisionLLM) -> None:
        vlm.validate_image(ImageInput(path="/tmp/x.png"))  # should not raise

    def test_validate_image_non_image_input_raises(self, vlm: FakeVisionLLM) -> None:
        with pytest.raises(ValueError, match="ImageInput"):
            vlm.validate_image("not_an_image")  # type: ignore[arg-type]

    def test_validate_image_dict_raises(self, vlm: FakeVisionLLM) -> None:
        with pytest.raises(ValueError, match="ImageInput"):
            vlm.validate_image({"path": "/tmp/x.png"})  # type: ignore[arg-type]


# ── BaseVisionLLM preprocess_image ──────────────────────────────────────


class TestPreprocessImage:
    """Tests for default preprocess_image (pass-through)."""

    def test_default_returns_same(self) -> None:
        vlm = FakeVisionLLM()
        img = ImageInput(path="/tmp/x.png")
        result = vlm.preprocess_image(img)
        assert result is img

    def test_default_with_max_size_returns_same(self) -> None:
        vlm = FakeVisionLLM()
        img = ImageInput(data=b"\x89PNG")
        result = vlm.preprocess_image(img, max_size=(1024, 1024))
        assert result is img


# ── FakeVisionLLM Behavior ──────────────────────────────────────────────


class TestFakeVisionLLM:
    """Tests for the FakeVisionLLM contract."""

    def test_chat_with_image_returns_response(self) -> None:
        vlm = FakeVisionLLM(model="test-v")
        img = ImageInput(path="/tmp/x.png")
        resp = vlm.chat_with_image("What is this?", img)
        assert isinstance(resp, ChatResponse)
        assert resp.content == "Described: What is this?"
        assert resp.model == "test-v"

    def test_chat_with_image_validates_text(self) -> None:
        vlm = FakeVisionLLM()
        img = ImageInput(path="/tmp/x.png")
        with pytest.raises(ValueError, match="empty"):
            vlm.chat_with_image("", img)

    def test_chat_with_image_validates_image(self) -> None:
        vlm = FakeVisionLLM()
        with pytest.raises(ValueError, match="ImageInput"):
            vlm.chat_with_image("Describe", "not_image")  # type: ignore[arg-type]

    def test_call_count_increments(self) -> None:
        vlm = FakeVisionLLM()
        img = ImageInput(base64="abc123")
        vlm.chat_with_image("A", img)
        vlm.chat_with_image("B", img)
        assert vlm.call_count == 2

    def test_kwargs_forwarded(self) -> None:
        vlm = FakeVisionLLM(model="v1", extra="val")
        assert vlm.config == {"extra": "val"}


# ── LLMFactory Vision Provider Registry ────────────────────────────────


class TestLLMFactoryVisionRegistry:
    """Tests for LLMFactory vision provider registration and creation."""

    def test_register_and_create_vision(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("fake", FakeVisionLLM)
        vlm = factory.create_vision_llm("fake", model="v1")
        assert isinstance(vlm, FakeVisionLLM)
        assert vlm.model == "v1"

    def test_case_insensitive_register(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("FakeVision", FakeVisionLLM)
        vlm = factory.create_vision_llm("fakevision")
        assert isinstance(vlm, FakeVisionLLM)

    def test_case_insensitive_create(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("fake", FakeVisionLLM)
        vlm = factory.create_vision_llm("FAKE")
        assert isinstance(vlm, FakeVisionLLM)

    def test_unknown_vision_provider_raises(self) -> None:
        factory = LLMFactory()
        with pytest.raises(ValueError, match="Unknown Vision LLM provider"):
            factory.create_vision_llm("nonexistent")

    def test_error_lists_available_vision_providers(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("alpha", FakeVisionLLM)
        factory.register_vision_provider("beta", FakeVisionLLM)
        with pytest.raises(ValueError, match="alpha, beta"):
            factory.create_vision_llm("gamma")

    def test_register_non_subclass_raises(self) -> None:
        factory = LLMFactory()
        with pytest.raises(TypeError, match="BaseVisionLLM"):
            factory.register_vision_provider("bad", str)  # type: ignore[arg-type]

    def test_list_vision_providers(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("beta", FakeVisionLLM)
        factory.register_vision_provider("alpha", FakeVisionLLM)
        assert factory.list_vision_providers() == ["alpha", "beta"]

    def test_vision_registry_independent_from_llm(self) -> None:
        """Vision and text LLM registries are separate."""
        factory = LLMFactory()
        factory.register_vision_provider("fake", FakeVisionLLM)
        assert factory.list_providers() == []  # text LLM registry empty
        assert factory.list_vision_providers() == ["fake"]

    def test_create_vision_llm_with_kwargs(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("fake", FakeVisionLLM)
        vlm = factory.create_vision_llm("fake", model="gpt-4o", extra="val")
        assert vlm.model == "gpt-4o"
        assert vlm.config == {"extra": "val"}


# ── LLMFactory.create_vision_llm_from_settings ─────────────────────────


class TestCreateVisionLLMFromSettings:
    """Tests for factory creation from VisionLLMSettings."""

    def test_create_from_settings_enabled(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("fake", FakeVisionLLM)

        @dataclass(frozen=True)
        class FakeSettings:
            enabled: bool = True
            provider: str = "fake"
            model: str = "gpt-4o"
            max_image_size: int = 2048
            api_key: str | None = None

        vlm = factory.create_vision_llm_from_settings(FakeSettings())
        assert isinstance(vlm, FakeVisionLLM)
        assert vlm.model == "gpt-4o"

    def test_create_from_settings_disabled_returns_none(self) -> None:
        factory = LLMFactory()

        @dataclass(frozen=True)
        class FakeSettings:
            enabled: bool = False
            provider: str = "fake"
            model: str = "gpt-4o"
            max_image_size: int = 2048

        result = factory.create_vision_llm_from_settings(FakeSettings())
        assert result is None

    def test_create_from_settings_forwards_fields(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("fake", FakeVisionLLM)

        @dataclass(frozen=True)
        class FakeSettings:
            enabled: bool = True
            provider: str = "fake"
            model: str = "vision-v1"
            max_image_size: int = 4096
            api_key: str | None = "sk-test"

        vlm = factory.create_vision_llm_from_settings(FakeSettings())
        assert vlm.model == "vision-v1"
        assert vlm.config["max_image_size"] == 4096
        assert vlm.config["api_key"] == "sk-test"

    def test_create_from_settings_filters_none_values(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("fake", FakeVisionLLM)

        @dataclass(frozen=True)
        class FakeSettings:
            enabled: bool = True
            provider: str = "fake"
            model: str = "v1"
            max_image_size: int = 2048
            api_key: str | None = None
            azure_endpoint: str | None = None

        vlm = factory.create_vision_llm_from_settings(FakeSettings())
        assert "api_key" not in vlm.config
        assert "azure_endpoint" not in vlm.config

    def test_create_from_real_settings_type(self) -> None:
        """Test with actual VisionLLMSettings dataclass."""
        from src.core.settings import VisionLLMSettings

        factory = LLMFactory()
        factory.register_vision_provider("fake", FakeVisionLLM)

        settings = VisionLLMSettings(
            enabled=True,
            provider="fake",
            model="gpt-4o",
            max_image_size=2048,
        )
        vlm = factory.create_vision_llm_from_settings(settings)
        assert isinstance(vlm, FakeVisionLLM)
        assert vlm.model == "gpt-4o"
