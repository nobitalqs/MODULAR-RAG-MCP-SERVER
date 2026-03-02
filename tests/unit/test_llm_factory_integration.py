"""Integration tests for LLM factory with all providers.

Tests that all LLM providers work correctly with LLMFactory's
create_from_settings method and factory registration.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.core.settings import LLMSettings
from src.libs.llm import (
    AzureLLM,
    DeepSeekLLM,
    LLMFactory,
    OllamaLLM,
    OpenAILLM,
)
from src.libs.llm.base_llm import Message


def test_factory_registers_all_providers() -> None:
    """Test that all providers can be registered with the factory."""
    factory = LLMFactory()

    factory.register_provider("openai", OpenAILLM)
    factory.register_provider("azure", AzureLLM)
    factory.register_provider("deepseek", DeepSeekLLM)
    factory.register_provider("ollama", OllamaLLM)

    providers = factory.list_providers()
    assert "openai" in providers
    assert "azure" in providers
    assert "deepseek" in providers
    assert "ollama" in providers


def test_factory_create_from_settings_openai() -> None:
    """Test creating OpenAI LLM from settings."""
    factory = LLMFactory()
    factory.register_provider("openai", OpenAILLM)

    settings = LLMSettings(
        provider="openai",
        model="gpt-4o",
        temperature=0.5,
        max_tokens=2048,
        api_key="test-key",
    )

    llm = factory.create_from_settings(settings)

    assert isinstance(llm, OpenAILLM)
    assert llm.model == "gpt-4o"
    assert llm.temperature == 0.5
    assert llm.max_tokens == 2048


def test_factory_create_from_settings_azure() -> None:
    """Test creating Azure LLM from settings."""
    factory = LLMFactory()
    factory.register_provider("azure", AzureLLM)

    settings = LLMSettings(
        provider="azure",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1024,
        api_key="test-key",
        api_version="2024-02-15-preview",
        azure_endpoint="https://test.openai.azure.com",
        deployment_name="gpt-4o-deployment",
    )

    llm = factory.create_from_settings(settings)

    assert isinstance(llm, AzureLLM)
    assert llm.model == "gpt-4o"
    assert llm.deployment_name == "gpt-4o-deployment"
    assert llm.api_version == "2024-02-15-preview"


def test_factory_create_from_settings_deepseek() -> None:
    """Test creating DeepSeek LLM from settings."""
    factory = LLMFactory()
    factory.register_provider("deepseek", DeepSeekLLM)

    settings = LLMSettings(
        provider="deepseek",
        model="deepseek-chat",
        temperature=0.8,
        max_tokens=512,
        api_key="test-key",
    )

    llm = factory.create_from_settings(settings)

    assert isinstance(llm, DeepSeekLLM)
    assert llm.model == "deepseek-chat"
    assert llm.temperature == 0.8


def test_factory_create_from_settings_ollama() -> None:
    """Test creating Ollama LLM from settings."""
    factory = LLMFactory()
    factory.register_provider("ollama", OllamaLLM)

    settings = LLMSettings(
        provider="ollama",
        model="llama2",
        temperature=0.6,
        max_tokens=1024,
        base_url="http://localhost:11434",
    )

    llm = factory.create_from_settings(settings)

    assert isinstance(llm, OllamaLLM)
    assert llm.model == "llama2"
    assert llm.base_url == "http://localhost:11434"


def test_factory_filters_none_values() -> None:
    """Test that create_from_settings filters None values."""
    factory = LLMFactory()
    factory.register_provider("openai", OpenAILLM)

    # Settings with None values
    settings = LLMSettings(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1024,
        api_key="test-key",
        api_version=None,  # Should be filtered out
        azure_endpoint=None,  # Should be filtered out
        deployment_name=None,  # Should be filtered out
        base_url=None,  # Should be filtered out
    )

    llm = factory.create_from_settings(settings)

    assert isinstance(llm, OpenAILLM)
    assert llm.model == "gpt-4o"


def test_factory_unknown_provider_raises() -> None:
    """Test that unknown provider raises ValueError."""
    factory = LLMFactory()
    factory.register_provider("openai", OpenAILLM)

    settings = LLMSettings(
        provider="unknown",
        model="test",
        temperature=0.7,
        max_tokens=1024,
    )

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        factory.create_from_settings(settings)


def test_all_providers_can_chat() -> None:
    """Test that all providers can handle chat requests."""
    factory = LLMFactory()
    factory.register_provider("openai", OpenAILLM)
    factory.register_provider("azure", AzureLLM)
    factory.register_provider("deepseek", DeepSeekLLM)
    factory.register_provider("ollama", OllamaLLM)

    messages = [Message(role="user", content="Hello")]

    # OpenAI
    openai_llm = factory.create("openai", model="gpt-4o", api_key="test-key")
    mock_response = {
        "choices": [{"message": {"content": "OpenAI response"}}],
        "model": "gpt-4o",
    }
    with patch.object(openai_llm, "_call_api", return_value=mock_response):
        response = openai_llm.chat(messages)
        assert response.content == "OpenAI response"

    # Azure
    azure_llm = factory.create(
        "azure",
        model="gpt-4o",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment_name="gpt-4o-deployment",
    )
    with patch.object(azure_llm, "_call_api", return_value=mock_response):
        response = azure_llm.chat(messages)
        assert response.content == "OpenAI response"

    # DeepSeek
    deepseek_llm = factory.create("deepseek", model="deepseek-chat", api_key="test-key")
    with patch.object(deepseek_llm, "_call_api", return_value=mock_response):
        response = deepseek_llm.chat(messages)
        assert response.content == "OpenAI response"

    # Ollama
    ollama_llm = factory.create("ollama", model="llama2")
    ollama_response = {"message": {"content": "Ollama response"}, "model": "llama2"}
    with patch.object(ollama_llm, "_call_api", return_value=ollama_response):
        response = ollama_llm.chat(messages)
        assert response.content == "Ollama response"
