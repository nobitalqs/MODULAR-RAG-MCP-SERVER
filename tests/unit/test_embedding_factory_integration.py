"""Integration test for embedding factory with all providers."""

from __future__ import annotations

from src.libs.embedding import (
    AzureEmbedding,
    EmbeddingFactory,
    OllamaEmbedding,
    OpenAIEmbedding,
)


def test_factory_can_create_all_providers():
    """Factory can register and create all embedding providers."""
    factory = EmbeddingFactory()
    factory.register_provider("openai", OpenAIEmbedding)
    factory.register_provider("azure", AzureEmbedding)
    factory.register_provider("ollama", OllamaEmbedding)

    # Check all providers are registered
    providers = factory.list_providers()
    assert "azure" in providers
    assert "ollama" in providers
    assert "openai" in providers

    # Test OpenAI creation
    openai_emb = factory.create(
        "openai",
        model="text-embedding-3-small",
        api_key="test-key",
    )
    assert isinstance(openai_emb, OpenAIEmbedding)
    assert openai_emb.get_dimension() == 1536

    # Test Azure creation
    azure_emb = factory.create(
        "azure",
        model="ada-002",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
    )
    assert isinstance(azure_emb, AzureEmbedding)
    assert azure_emb.get_dimension() == 1536

    # Test Ollama creation
    ollama_emb = factory.create("ollama", model="nomic-embed-text")
    assert isinstance(ollama_emb, OllamaEmbedding)
    assert ollama_emb.get_dimension() == 768
