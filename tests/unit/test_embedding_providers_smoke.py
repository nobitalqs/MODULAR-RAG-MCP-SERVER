"""Smoke tests for OpenAI and Azure Embedding providers.

Tests factory registration, embedding logic, validation, and error handling
with mocked OpenAI/AzureOpenAI clients.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.libs.embedding.azure_embedding import AzureEmbedding, AzureEmbeddingError
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.openai_embedding import OpenAIEmbedding, OpenAIEmbeddingError


# ============================================================================
# OpenAI Embedding Tests
# ============================================================================


def test_openai_factory_can_create():
    """Factory can create OpenAI provider."""
    factory = EmbeddingFactory()
    factory.register_provider("openai", OpenAIEmbedding)

    emb = factory.create(
        "openai",
        model="text-embedding-3-small",
        api_key="test-key",
    )

    assert isinstance(emb, OpenAIEmbedding)
    assert emb.model == "text-embedding-3-small"
    assert emb.dimensions == 1536  # default


def test_openai_embed_returns_vectors():
    """OpenAI embed returns list of vectors."""
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_client.embeddings.create.return_value = mock_response

    emb = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")

    # Patch _create_client to return our mock
    with patch.object(emb, "_create_client", return_value=mock_client):
        result = emb.embed(["hello", "world"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_client.embeddings.create.assert_called_once_with(
        input=["hello", "world"],
        model="text-embedding-3-small",
        dimensions=1536,
    )


def test_openai_validates_texts():
    """OpenAI embed validates texts before API call."""
    emb = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")

    with pytest.raises(ValueError, match="Texts list cannot be empty"):
        emb.embed([])

    with pytest.raises(ValueError, match="not a string"):
        emb.embed([123])  # type: ignore

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        emb.embed(["  "])


def test_openai_api_error_raises():
    """OpenAI embed raises OpenAIEmbeddingError on API failure."""
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = Exception("API rate limit")

    emb = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")

    with patch.object(emb, "_create_client", return_value=mock_client):
        with pytest.raises(OpenAIEmbeddingError, match="API call failed"):
            emb.embed(["test"])


def test_openai_missing_api_key_raises():
    """OpenAI requires api_key parameter or OPENAI_API_KEY env var."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key is required"):
            OpenAIEmbedding(model="text-embedding-3-small")


def test_openai_get_dimension():
    """OpenAI get_dimension returns configured dimensions."""
    emb = OpenAIEmbedding(
        model="text-embedding-3-small",
        dimensions=512,
        api_key="test-key",
    )
    assert emb.get_dimension() == 512


# ============================================================================
# Azure Embedding Tests
# ============================================================================


def test_azure_factory_can_create():
    """Factory can create Azure provider."""
    factory = EmbeddingFactory()
    factory.register_provider("azure", AzureEmbedding)

    emb = factory.create(
        "azure",
        model="text-embedding-ada-002",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
    )

    assert isinstance(emb, AzureEmbedding)
    assert emb.model == "text-embedding-ada-002"
    assert emb.dimensions == 1536


def test_azure_embed_returns_vectors():
    """Azure embed returns list of vectors."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.7, 0.8]),
    ]
    mock_client.embeddings.create.return_value = mock_response

    emb = AzureEmbedding(
        model="text-embedding-ada-002",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
    )

    with patch.object(emb, "_create_client", return_value=mock_client):
        result = emb.embed(["test"])

    assert result == [[0.7, 0.8]]
    mock_client.embeddings.create.assert_called_once_with(
        input=["test"],
        model="text-embedding-ada-002",  # deployment_name defaults to model
        dimensions=1536,
    )


def test_azure_validates_texts():
    """Azure embed validates texts before API call."""
    emb = AzureEmbedding(
        model="text-embedding-ada-002",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
    )

    with pytest.raises(ValueError, match="Texts list cannot be empty"):
        emb.embed([])


def test_azure_api_error_raises():
    """Azure embed raises AzureEmbeddingError on API failure."""
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = Exception("Network timeout")

    emb = AzureEmbedding(
        model="text-embedding-ada-002",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
    )

    with patch.object(emb, "_create_client", return_value=mock_client):
        with pytest.raises(AzureEmbeddingError, match="API call failed"):
            emb.embed(["test"])


def test_azure_missing_api_key_raises():
    """Azure requires api_key parameter or env var."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key is required"):
            AzureEmbedding(
                model="text-embedding-ada-002",
                azure_endpoint="https://test.openai.azure.com",
            )


def test_azure_missing_endpoint_raises():
    """Azure requires azure_endpoint parameter or env var."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="endpoint is required"):
            AzureEmbedding(
                model="text-embedding-ada-002",
                api_key="test-key",
            )


def test_azure_get_dimension():
    """Azure get_dimension returns configured dimensions."""
    emb = AzureEmbedding(
        model="text-embedding-ada-002",
        dimensions=256,
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
    )
    assert emb.get_dimension() == 256


def test_azure_deployment_name_override():
    """Azure uses deployment_name if provided, else defaults to model."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1])]
    mock_client.embeddings.create.return_value = mock_response

    emb = AzureEmbedding(
        model="text-embedding-ada-002",
        deployment_name="my-custom-deployment",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
    )

    with patch.object(emb, "_create_client", return_value=mock_client):
        emb.embed(["test"])

    # Should use deployment_name, not model
    mock_client.embeddings.create.assert_called_once_with(
        input=["test"],
        model="my-custom-deployment",
        dimensions=1536,
    )
