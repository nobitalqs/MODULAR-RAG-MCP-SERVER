"""Smoke tests for all LLM providers.

Tests factory registration, chat functionality, validation, and error handling
for OpenAI, Azure, DeepSeek, and Ollama providers using mocked API calls.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.libs.llm.azure_llm import AzureLLM, AzureLLMError
from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.deepseek_llm import DeepSeekLLM, DeepSeekLLMError
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.ollama_llm import OllamaLLM, OllamaLLMError
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError


# ---------------------------------------------------------------------------
# OpenAI Provider Tests
# ---------------------------------------------------------------------------


def test_openai_factory_can_create() -> None:
    """Test that OpenAILLM can be registered and created via factory."""
    factory = LLMFactory()
    factory.register_provider("openai", OpenAILLM)

    llm = factory.create(
        "openai",
        model="gpt-4o",
        api_key="test-key",
    )

    assert isinstance(llm, OpenAILLM)
    assert llm.model == "gpt-4o"


def test_openai_chat_returns_response() -> None:
    """Test that OpenAILLM.chat returns a valid ChatResponse."""
    llm = OpenAILLM(model="gpt-4o", api_key="test-key")

    mock_response = {
        "choices": [{"message": {"content": "Hello, world!"}}],
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    with patch.object(llm, "_call_api", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]
        response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.content == "Hello, world!"
    assert response.model == "gpt-4o"
    assert response.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_openai_validates_messages() -> None:
    """Test that OpenAILLM validates messages before sending."""
    llm = OpenAILLM(model="gpt-4o", api_key="test-key")

    with pytest.raises(ValueError, match="messages must contain at least one message"):
        llm.chat([])

    with pytest.raises(ValueError, match="empty content"):
        llm.chat([Message(role="user", content="")])


def test_openai_api_error_raises() -> None:
    """Test that OpenAILLM raises OpenAILLMError on API failures."""
    llm = OpenAILLM(model="gpt-4o", api_key="test-key")

    with patch.object(llm, "_call_api", side_effect=Exception("Connection failed")):
        with pytest.raises(OpenAILLMError, match=r"\[OpenAI\] API call failed"):
            llm.chat([Message(role="user", content="Hello")])


def test_openai_missing_api_key_raises() -> None:
    """Test that OpenAILLM raises ValueError if api_key is missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match=r"\[OpenAI\] Missing API key"):
            OpenAILLM(model="gpt-4o")


# ---------------------------------------------------------------------------
# Azure Provider Tests
# ---------------------------------------------------------------------------


def test_azure_factory_can_create() -> None:
    """Test that AzureLLM can be registered and created via factory."""
    factory = LLMFactory()
    factory.register_provider("azure", AzureLLM)

    llm = factory.create(
        "azure",
        model="gpt-4o",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment_name="gpt-4o-deployment",
    )

    assert isinstance(llm, AzureLLM)
    assert llm.model == "gpt-4o"
    assert llm.deployment_name == "gpt-4o-deployment"


def test_azure_chat_returns_response() -> None:
    """Test that AzureLLM.chat returns a valid ChatResponse."""
    llm = AzureLLM(
        model="gpt-4o",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment_name="gpt-4o-deployment",
    )

    mock_response = {
        "choices": [{"message": {"content": "Azure response"}}],
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 3,
            "total_tokens": 11,
        },
    }

    with patch.object(llm, "_call_api", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]
        response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.content == "Azure response"
    assert response.model == "gpt-4o"


def test_azure_validates_messages() -> None:
    """Test that AzureLLM validates messages before sending."""
    llm = AzureLLM(
        model="gpt-4o",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment_name="gpt-4o-deployment",
    )

    with pytest.raises(ValueError, match="messages must contain at least one message"):
        llm.chat([])


def test_azure_api_error_raises() -> None:
    """Test that AzureLLM raises AzureLLMError on API failures."""
    llm = AzureLLM(
        model="gpt-4o",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment_name="gpt-4o-deployment",
    )

    with patch.object(llm, "_call_api", side_effect=Exception("Azure error")):
        with pytest.raises(AzureLLMError, match=r"\[Azure\] API call failed"):
            llm.chat([Message(role="user", content="Hello")])


def test_azure_missing_api_key_raises() -> None:
    """Test that AzureLLM raises ValueError if api_key is missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match=r"\[Azure\] Missing API key"):
            AzureLLM(
                model="gpt-4o",
                azure_endpoint="https://test.openai.azure.com",
                api_version="2024-02-15-preview",
                deployment_name="gpt-4o-deployment",
            )


def test_azure_missing_endpoint_raises() -> None:
    """Test that AzureLLM raises ValueError if azure_endpoint is missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match=r"\[Azure\] Missing endpoint"):
            AzureLLM(
                model="gpt-4o",
                api_key="test-key",
                api_version="2024-02-15-preview",
                deployment_name="gpt-4o-deployment",
            )


def test_azure_missing_deployment_name_raises() -> None:
    """Test that AzureLLM raises ValueError if deployment_name is missing."""
    with pytest.raises(ValueError, match=r"\[Azure\] Missing deployment_name"):
        AzureLLM(
            model="gpt-4o",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
        )


# ---------------------------------------------------------------------------
# DeepSeek Provider Tests
# ---------------------------------------------------------------------------


def test_deepseek_factory_can_create() -> None:
    """Test that DeepSeekLLM can be registered and created via factory."""
    factory = LLMFactory()
    factory.register_provider("deepseek", DeepSeekLLM)

    llm = factory.create(
        "deepseek",
        model="deepseek-chat",
        api_key="test-key",
    )

    assert isinstance(llm, DeepSeekLLM)
    assert llm.model == "deepseek-chat"


def test_deepseek_chat_returns_response() -> None:
    """Test that DeepSeekLLM.chat returns a valid ChatResponse."""
    llm = DeepSeekLLM(model="deepseek-chat", api_key="test-key")

    mock_response = {
        "choices": [{"message": {"content": "DeepSeek response"}}],
        "model": "deepseek-chat",
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
        },
    }

    with patch.object(llm, "_call_api", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]
        response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.content == "DeepSeek response"
    assert response.model == "deepseek-chat"


def test_deepseek_validates_messages() -> None:
    """Test that DeepSeekLLM validates messages before sending."""
    llm = DeepSeekLLM(model="deepseek-chat", api_key="test-key")

    with pytest.raises(ValueError, match="messages must contain at least one message"):
        llm.chat([])


def test_deepseek_api_error_raises() -> None:
    """Test that DeepSeekLLM raises DeepSeekLLMError on API failures."""
    llm = DeepSeekLLM(model="deepseek-chat", api_key="test-key")

    with patch.object(llm, "_call_api", side_effect=Exception("Network error")):
        with pytest.raises(DeepSeekLLMError, match=r"\[DeepSeek\] API call failed"):
            llm.chat([Message(role="user", content="Hello")])


def test_deepseek_missing_api_key_raises() -> None:
    """Test that DeepSeekLLM raises ValueError if api_key is missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match=r"\[DeepSeek\] Missing API key"):
            DeepSeekLLM(model="deepseek-chat")


# ---------------------------------------------------------------------------
# Ollama Provider Tests
# ---------------------------------------------------------------------------


def test_ollama_factory_can_create() -> None:
    """Test that OllamaLLM can be registered and created via factory."""
    factory = LLMFactory()
    factory.register_provider("ollama", OllamaLLM)

    llm = factory.create(
        "ollama",
        model="llama2",
    )

    assert isinstance(llm, OllamaLLM)
    assert llm.model == "llama2"


def test_ollama_chat_returns_response() -> None:
    """Test that OllamaLLM.chat returns a valid ChatResponse."""
    llm = OllamaLLM(model="llama2")

    mock_response = {
        "message": {"content": "Ollama response"},
        "model": "llama2",
        "prompt_eval_count": 15,
        "eval_count": 10,
    }

    with patch.object(llm, "_call_api", return_value=mock_response):
        messages = [Message(role="user", content="Hello")]
        response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.content == "Ollama response"
    assert response.model == "llama2"
    assert response.usage == {
        "prompt_tokens": 15,
        "completion_tokens": 10,
        "total_tokens": 25,
    }


def test_ollama_validates_messages() -> None:
    """Test that OllamaLLM validates messages before sending."""
    llm = OllamaLLM(model="llama2")

    with pytest.raises(ValueError, match="messages must contain at least one message"):
        llm.chat([])


def test_ollama_api_error_raises() -> None:
    """Test that OllamaLLM raises OllamaLLMError on API failures."""
    llm = OllamaLLM(model="llama2")

    with patch.object(llm, "_call_api", side_effect=Exception("Connection refused")):
        with pytest.raises(OllamaLLMError, match=r"\[Ollama\] API call failed"):
            llm.chat([Message(role="user", content="Hello")])
