"""Tests for OpenAI Vision LLM implementation.

Tests cover:
- Constructor validation (api_key required)
- Environment variable fallback for api_key
- chat_with_image with mocked HTTP
- Image encoding (path, bytes, base64)
- Conversation history forwarding
- Error handling (HTTP errors, unexpected response format)
- API key not leaked in error messages
- Factory integration (register + create_vision_llm)
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM, OpenAIVisionLLMError
from src.libs.llm.llm_factory import LLMFactory


# ── Fixtures ────────────────────────────────────────────────────────────


def _make_vlm(**overrides: Any) -> OpenAIVisionLLM:
    """Create OpenAIVisionLLM with default valid args."""
    defaults = {
        "model": "gpt-4o",
        "api_key": "sk-test-key",
    }
    defaults.update(overrides)
    return OpenAIVisionLLM(**defaults)


MOCK_API_RESPONSE = {
    "choices": [
        {
            "message": {"content": "This is a bar chart showing revenue growth."},
            "finish_reason": "stop",
        }
    ],
    "model": "gpt-4o",
    "usage": {
        "prompt_tokens": 150,
        "completion_tokens": 20,
        "total_tokens": 170,
    },
}


# ── Constructor Validation Tests ────────────────────────────────────────


class TestOpenAIVisionLLMInit:
    """Tests for constructor validation and defaults."""

    def test_valid_init(self) -> None:
        vlm = _make_vlm()
        assert vlm.model == "gpt-4o"
        assert vlm.api_key == "sk-test-key"

    def test_is_base_vision_llm(self) -> None:
        vlm = _make_vlm()
        assert isinstance(vlm, BaseVisionLLM)

    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenAIVisionLLM(model="gpt-4o")

    def test_api_key_env_fallback(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}, clear=True):
            vlm = OpenAIVisionLLM(model="gpt-4o")
            assert vlm.api_key == "env-key"

    def test_default_base_url(self) -> None:
        vlm = _make_vlm()
        assert vlm.base_url == "https://api.openai.com/v1"

    def test_custom_base_url(self) -> None:
        vlm = _make_vlm(base_url="https://custom.api.com/v1")
        assert vlm.base_url == "https://custom.api.com/v1"

    def test_default_max_image_size(self) -> None:
        vlm = _make_vlm()
        assert vlm.max_image_size == 2048

    def test_custom_max_image_size(self) -> None:
        vlm = _make_vlm(max_image_size=4096)
        assert vlm.max_image_size == 4096

    def test_default_temperature(self) -> None:
        vlm = _make_vlm()
        assert vlm.temperature == 0.7

    def test_custom_temperature(self) -> None:
        vlm = _make_vlm(temperature=0.0)
        assert vlm.temperature == 0.0

    def test_default_max_tokens(self) -> None:
        vlm = _make_vlm()
        assert vlm.max_tokens == 1024

    def test_ignores_unknown_kwargs(self) -> None:
        vlm = _make_vlm(deployment_name="ignored", azure_endpoint="ignored")
        assert vlm.model == "gpt-4o"


# ── chat_with_image Tests ───────────────────────────────────────────────


class TestChatWithImage:
    """Tests for chat_with_image with mocked API calls."""

    def test_basic_call(self) -> None:
        vlm = _make_vlm()
        vlm._call_api = MagicMock(return_value=MOCK_API_RESPONSE)

        img = ImageInput(base64="iVBORw0KGgo=")
        resp = vlm.chat_with_image("Describe this chart", img)

        assert isinstance(resp, ChatResponse)
        assert resp.content == "This is a bar chart showing revenue growth."
        assert resp.model == "gpt-4o"
        assert resp.usage["total_tokens"] == 170

    def test_validates_empty_text(self) -> None:
        vlm = _make_vlm()
        img = ImageInput(base64="abc")
        with pytest.raises(ValueError, match="empty"):
            vlm.chat_with_image("", img)

    def test_validates_image_type(self) -> None:
        vlm = _make_vlm()
        with pytest.raises(ValueError, match="ImageInput"):
            vlm.chat_with_image("Describe", "not_image")  # type: ignore[arg-type]

    def test_payload_contains_model(self) -> None:
        vlm = _make_vlm(model="gpt-4o")
        captured_payload: dict = {}

        def mock_call_api(payload: dict) -> dict:
            captured_payload.update(payload)
            return MOCK_API_RESPONSE

        vlm._call_api = mock_call_api  # type: ignore[assignment]

        img = ImageInput(base64="abc")
        vlm.chat_with_image("Describe", img)
        assert captured_payload["model"] == "gpt-4o"

    def test_payload_contains_image_url(self) -> None:
        vlm = _make_vlm()
        captured_payload: dict = {}

        def mock_call_api(payload: dict) -> dict:
            captured_payload.update(payload)
            return MOCK_API_RESPONSE

        vlm._call_api = mock_call_api  # type: ignore[assignment]

        img = ImageInput(base64="dGVzdA==", mime_type="image/jpeg")
        vlm.chat_with_image("Describe", img)

        messages = captured_payload["messages"]
        last_msg = messages[-1]
        assert last_msg["role"] == "user"
        content = last_msg["content"]
        assert any(
            c["type"] == "image_url"
            and "data:image/jpeg;base64,dGVzdA==" in c["image_url"]["url"]
            for c in content
        )

    def test_payload_contains_text(self) -> None:
        vlm = _make_vlm()
        captured_payload: dict = {}

        def mock_call_api(payload: dict) -> dict:
            captured_payload.update(payload)
            return MOCK_API_RESPONSE

        vlm._call_api = mock_call_api  # type: ignore[assignment]

        img = ImageInput(base64="abc")
        vlm.chat_with_image("What is this?", img)

        messages = captured_payload["messages"]
        last_msg = messages[-1]
        content = last_msg["content"]
        assert any(
            c["type"] == "text" and c["text"] == "What is this?" for c in content
        )

    def test_conversation_history_forwarded(self) -> None:
        vlm = _make_vlm()
        captured_payload: dict = {}

        def mock_call_api(payload: dict) -> dict:
            captured_payload.update(payload)
            return MOCK_API_RESPONSE

        vlm._call_api = mock_call_api  # type: ignore[assignment]

        history = [
            Message(role="system", content="You are an image analyst."),
            Message(role="user", content="Previous question"),
        ]
        img = ImageInput(base64="abc")
        vlm.chat_with_image("Follow-up", img, messages=history)

        messages = captured_payload["messages"]
        assert len(messages) == 3  # 2 history + 1 current
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "Previous question"

    def test_temperature_override(self) -> None:
        vlm = _make_vlm(temperature=0.5)
        captured_payload: dict = {}

        def mock_call_api(payload: dict) -> dict:
            captured_payload.update(payload)
            return MOCK_API_RESPONSE

        vlm._call_api = mock_call_api  # type: ignore[assignment]

        img = ImageInput(base64="abc")
        vlm.chat_with_image("Describe", img, temperature=0.0)
        assert captured_payload["temperature"] == 0.0

    def test_max_tokens_override(self) -> None:
        vlm = _make_vlm(max_tokens=512)
        captured_payload: dict = {}

        def mock_call_api(payload: dict) -> dict:
            captured_payload.update(payload)
            return MOCK_API_RESPONSE

        vlm._call_api = mock_call_api  # type: ignore[assignment]

        img = ImageInput(base64="abc")
        vlm.chat_with_image("Describe", img, max_tokens=2048)
        assert captured_payload["max_tokens"] == 2048


# ── Image Encoding Tests ───────────────────────────────────────────────


class TestImageEncoding:
    """Tests for _get_image_base64."""

    def test_base64_passthrough(self) -> None:
        vlm = _make_vlm()
        img = ImageInput(base64="already_encoded")
        assert vlm._get_image_base64(img) == "already_encoded"

    def test_bytes_to_base64(self) -> None:
        vlm = _make_vlm()
        raw = b"fake image data"
        img = ImageInput(data=raw)
        result = vlm._get_image_base64(img)
        assert result == base64.b64encode(raw).decode("utf-8")

    def test_path_to_base64(self, tmp_path: Any) -> None:
        vlm = _make_vlm()
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG_fake_data")
        img = ImageInput(path=str(img_file))
        result = vlm._get_image_base64(img)
        expected = base64.b64encode(b"\x89PNG_fake_data").decode("utf-8")
        assert result == expected


# ── Error Handling Tests ────────────────────────────────────────────────


class TestErrorHandling:
    """Tests for error wrapping."""

    def test_api_error_wrapped(self) -> None:
        vlm = _make_vlm()
        vlm._call_api = MagicMock(side_effect=RuntimeError("Connection refused"))

        img = ImageInput(base64="abc")
        with pytest.raises(OpenAIVisionLLMError, match="API call failed"):
            vlm.chat_with_image("Describe", img)

    def test_unexpected_response_format(self) -> None:
        vlm = _make_vlm()
        vlm._call_api = MagicMock(return_value={"unexpected": "format"})

        img = ImageInput(base64="abc")
        with pytest.raises(OpenAIVisionLLMError, match="Unexpected response"):
            vlm.chat_with_image("Describe", img)

    def test_api_key_not_leaked_in_error(self) -> None:
        vlm = _make_vlm(api_key="super-secret-key")
        vlm._call_api = MagicMock(
            side_effect=RuntimeError("Error with super-secret-key in message")
        )

        img = ImageInput(base64="abc")
        with pytest.raises(OpenAIVisionLLMError) as exc_info:
            vlm.chat_with_image("Describe", img)
        assert "super-secret-key" not in str(exc_info.value)

    def test_openai_vision_error_reraise(self) -> None:
        vlm = _make_vlm()
        vlm._call_api = MagicMock(
            side_effect=OpenAIVisionLLMError("Already wrapped")
        )

        img = ImageInput(base64="abc")
        with pytest.raises(OpenAIVisionLLMError, match="Already wrapped"):
            vlm.chat_with_image("Describe", img)

    def test_no_usage_in_response(self) -> None:
        vlm = _make_vlm()
        response_no_usage = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
        }
        vlm._call_api = MagicMock(return_value=response_no_usage)

        img = ImageInput(base64="abc")
        resp = vlm.chat_with_image("Describe", img)
        assert resp.usage is None


# ── Factory Integration Tests ──────────────────────────────────────────


class TestFactoryIntegration:
    """Tests for LLMFactory vision provider integration."""

    def test_register_and_create(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("openai", OpenAIVisionLLM)
        vlm = factory.create_vision_llm(
            "openai",
            model="gpt-4o",
            api_key="sk-test",
        )
        assert isinstance(vlm, OpenAIVisionLLM)
        assert vlm.model == "gpt-4o"

    def test_create_from_settings(self) -> None:
        from src.core.settings import VisionLLMSettings

        factory = LLMFactory()
        factory.register_vision_provider("openai", OpenAIVisionLLM)

        settings = VisionLLMSettings(
            enabled=True,
            provider="openai",
            model="gpt-4o",
            max_image_size=2048,
            api_key="sk-settings-key",
        )
        vlm = factory.create_vision_llm_from_settings(settings)
        assert isinstance(vlm, OpenAIVisionLLM)
        assert vlm.api_key == "sk-settings-key"
