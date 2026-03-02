"""Tests for B9: Azure Vision LLM implementation.

Tests cover:
- Constructor validation (api_key, azure_endpoint, deployment_name)
- Environment variable fallback for api_key/azure_endpoint
- chat_with_image with mocked HTTP
- Image encoding (path, bytes, base64)
- Conversation history forwarding
- Error handling (HTTP errors, unexpected response format)
- Factory integration (register + create_vision_llm)
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.azure_vision_llm import AzureVisionLLM, AzureVisionLLMError
from src.libs.llm.llm_factory import LLMFactory


# ── Fixtures ────────────────────────────────────────────────────────────


def _make_vlm(**overrides: Any) -> AzureVisionLLM:
    """Create AzureVisionLLM with default valid args."""
    defaults = {
        "model": "gpt-4o",
        "api_key": "sk-test-key",
        "azure_endpoint": "https://my-resource.openai.azure.com",
        "deployment_name": "gpt-4o-vision",
    }
    defaults.update(overrides)
    return AzureVisionLLM(**defaults)


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


class TestAzureVisionLLMInit:
    """Tests for constructor validation and defaults."""

    def test_valid_init(self) -> None:
        vlm = _make_vlm()
        assert vlm.model == "gpt-4o"
        assert vlm.deployment_name == "gpt-4o-vision"
        assert vlm.api_key == "sk-test-key"

    def test_is_base_vision_llm(self) -> None:
        vlm = _make_vlm()
        assert isinstance(vlm, BaseVisionLLM)

    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                AzureVisionLLM(
                    model="gpt-4o",
                    azure_endpoint="https://x.openai.azure.com",
                    deployment_name="gpt-4o",
                )

    def test_missing_endpoint_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="endpoint"):
                AzureVisionLLM(
                    model="gpt-4o",
                    api_key="sk-key",
                    deployment_name="gpt-4o",
                )

    def test_missing_deployment_name_raises(self) -> None:
        with pytest.raises(ValueError, match="deployment_name"):
            AzureVisionLLM(
                model="gpt-4o",
                api_key="sk-key",
                azure_endpoint="https://x.openai.azure.com",
            )

    def test_api_key_env_fallback(self) -> None:
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "env-key"}, clear=True):
            vlm = AzureVisionLLM(
                model="gpt-4o",
                azure_endpoint="https://x.openai.azure.com",
                deployment_name="gpt-4o",
            )
            assert vlm.api_key == "env-key"

    def test_endpoint_env_fallback(self) -> None:
        with patch.dict(
            "os.environ",
            {"AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com"},
            clear=True,
        ):
            vlm = AzureVisionLLM(
                model="gpt-4o",
                api_key="sk-key",
                deployment_name="gpt-4o",
            )
            assert vlm.endpoint == "https://env.openai.azure.com"

    def test_default_api_version(self) -> None:
        vlm = _make_vlm()
        assert vlm.api_version == "2024-02-15-preview"

    def test_custom_api_version(self) -> None:
        vlm = _make_vlm(api_version="2024-06-01")
        assert vlm.api_version == "2024-06-01"

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
            c["type"] == "image_url" and "data:image/jpeg;base64,dGVzdA==" in c["image_url"]["url"]
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
        assert any(c["type"] == "text" and c["text"] == "What is this?" for c in content)

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
        with pytest.raises(AzureVisionLLMError, match="API call failed"):
            vlm.chat_with_image("Describe", img)

    def test_unexpected_response_format(self) -> None:
        vlm = _make_vlm()
        vlm._call_api = MagicMock(return_value={"unexpected": "format"})

        img = ImageInput(base64="abc")
        with pytest.raises(AzureVisionLLMError, match="Unexpected response"):
            vlm.chat_with_image("Describe", img)

    def test_api_key_not_leaked_in_error(self) -> None:
        vlm = _make_vlm(api_key="super-secret-key")
        vlm._call_api = MagicMock(
            side_effect=RuntimeError("Error with super-secret-key in message")
        )

        img = ImageInput(base64="abc")
        with pytest.raises(AzureVisionLLMError) as exc_info:
            vlm.chat_with_image("Describe", img)
        assert "super-secret-key" not in str(exc_info.value)


# ── Factory Integration Tests ──────────────────────────────────────────


class TestFactoryIntegration:
    """Tests for LLMFactory vision provider integration."""

    def test_register_and_create(self) -> None:
        factory = LLMFactory()
        factory.register_vision_provider("azure", AzureVisionLLM)
        vlm = factory.create_vision_llm(
            "azure",
            model="gpt-4o",
            api_key="sk-test",
            azure_endpoint="https://x.openai.azure.com",
            deployment_name="gpt-4o",
        )
        assert isinstance(vlm, AzureVisionLLM)
        assert vlm.model == "gpt-4o"

    def test_create_from_settings(self) -> None:
        from src.core.settings import VisionLLMSettings

        factory = LLMFactory()
        factory.register_vision_provider("azure", AzureVisionLLM)

        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            max_image_size=2048,
            api_key="sk-settings-key",
            azure_endpoint="https://settings.openai.azure.com",
            deployment_name="gpt-4o-deploy",
        )
        vlm = factory.create_vision_llm_from_settings(settings)
        assert isinstance(vlm, AzureVisionLLM)
        assert vlm.api_key == "sk-settings-key"
        assert vlm.endpoint == "https://settings.openai.azure.com"
