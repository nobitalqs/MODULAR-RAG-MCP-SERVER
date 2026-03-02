"""Detailed tests for Ollama LLM provider.

Tests Ollama-specific behavior including base URL handling, timeout,
connection errors, and the fact that no API key is required.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.ollama_llm import OllamaLLM, OllamaLLMError


def test_ollama_factory_can_create() -> None:
    """Test that OllamaLLM can be registered and created via factory."""
    factory = LLMFactory()
    factory.register_provider("ollama", OllamaLLM)

    llm = factory.create("ollama", model="llama2")

    assert isinstance(llm, OllamaLLM)
    assert llm.model == "llama2"


def test_ollama_chat_success() -> None:
    """Test successful chat with Ollama-specific response format."""
    llm = OllamaLLM(model="mistral", temperature=0.5, max_tokens=512)

    mock_response = {
        "message": {"content": "I am a helpful assistant."},
        "model": "mistral",
        "prompt_eval_count": 20,
        "eval_count": 8,
    }

    with patch.object(llm, "_call_api", return_value=mock_response):
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
        ]
        response = llm.chat(messages, temperature=0.3)

    assert isinstance(response, ChatResponse)
    assert response.content == "I am a helpful assistant."
    assert response.model == "mistral"
    assert response.usage == {
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 28,
    }


def test_ollama_default_base_url() -> None:
    """Test that Ollama uses default base URL when none is provided."""
    with patch.dict("os.environ", {}, clear=True):
        llm = OllamaLLM(model="llama2")
        assert llm.base_url == "http://localhost:11434"


def test_ollama_env_var_base_url() -> None:
    """Test that Ollama respects OLLAMA_BASE_URL env var."""
    with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://192.168.1.100:11434"}):
        llm = OllamaLLM(model="llama2")
        assert llm.base_url == "http://192.168.1.100:11434"


def test_ollama_explicit_base_url() -> None:
    """Test that explicit base_url parameter takes precedence."""
    with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://env-url:11434"}):
        llm = OllamaLLM(model="llama2", base_url="http://explicit-url:11434")
        assert llm.base_url == "http://explicit-url:11434"


def test_ollama_connection_error() -> None:
    """Test that connection errors are properly wrapped."""
    llm = OllamaLLM(model="llama2")

    with patch.object(
        llm,
        "_call_api",
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        with pytest.raises(OllamaLLMError, match=r"\[Ollama\] API call failed"):
            llm.chat([Message(role="user", content="Hello")])


def test_ollama_timeout_error() -> None:
    """Test that timeout errors are properly wrapped."""
    llm = OllamaLLM(model="llama2")

    with patch.object(
        llm,
        "_call_api",
        side_effect=httpx.TimeoutException("Request timed out"),
    ):
        with pytest.raises(OllamaLLMError, match=r"\[Ollama\] API call failed"):
            llm.chat([Message(role="user", content="Hello")])


def test_ollama_no_api_key_needed() -> None:
    """Test that Ollama does not require an API key."""
    # Should not raise even with no env vars set
    with patch.dict("os.environ", {}, clear=True):
        llm = OllamaLLM(model="llama2")
        assert llm.model == "llama2"
        # Verify no api_key attribute exists
        assert not hasattr(llm, "api_key")


def test_ollama_response_without_usage() -> None:
    """Test that Ollama handles responses without usage stats."""
    llm = OllamaLLM(model="llama2")

    # Some Ollama versions may not include usage stats
    mock_response = {
        "message": {"content": "Response without usage"},
        "model": "llama2",
    }

    with patch.object(llm, "_call_api", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]
        response = llm.chat(messages)

    assert response.content == "Response without usage"
    assert response.usage is None


def test_ollama_unexpected_response_format() -> None:
    """Test that malformed responses raise OllamaLLMError."""
    llm = OllamaLLM(model="llama2")

    # Missing 'message' field
    mock_response = {"model": "llama2"}

    with patch.object(llm, "_call_api", return_value=mock_response):
        with pytest.raises(OllamaLLMError, match=r"\[Ollama\] Unexpected response format"):
            llm.chat([Message(role="user", content="Hello")])


def test_ollama_validates_empty_messages() -> None:
    """Test that Ollama validates empty message list."""
    llm = OllamaLLM(model="llama2")

    with pytest.raises(ValueError, match="messages must contain at least one message"):
        llm.chat([])


def test_ollama_validates_empty_content() -> None:
    """Test that Ollama validates empty message content."""
    llm = OllamaLLM(model="llama2")

    with pytest.raises(ValueError, match="empty content"):
        llm.chat([Message(role="user", content="")])


def test_ollama_validates_invalid_role() -> None:
    """Test that Ollama validates message roles."""
    llm = OllamaLLM(model="llama2")

    with pytest.raises(ValueError, match="Invalid role"):
        llm.chat([Message(role="invalid", content="Hello")])
