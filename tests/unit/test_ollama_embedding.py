"""Unit tests for Ollama Embedding provider.

Tests factory registration, embedding logic, connection handling,
and error scenarios with mocked httpx calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.ollama_embedding import OllamaEmbedding, OllamaEmbeddingError


def test_ollama_factory_can_create():
    """Factory can create Ollama provider."""
    factory = EmbeddingFactory()
    factory.register_provider("ollama", OllamaEmbedding)

    emb = factory.create("ollama", model="nomic-embed-text")

    assert isinstance(emb, OllamaEmbedding)
    assert emb.model == "nomic-embed-text"
    assert emb.dimensions == 768  # default


def test_ollama_embed_success():
    """Ollama embed returns vectors on successful API call."""
    emb = OllamaEmbedding(model="nomic-embed-text")

    mock_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    with patch.object(emb, "_call_api", return_value=mock_vectors) as mock_call:
        result = emb.embed(["hello", "world"])

    assert result == mock_vectors
    mock_call.assert_called_once_with(["hello", "world"])


def test_ollama_batch_embed():
    """Ollama handles multiple texts correctly."""
    emb = OllamaEmbedding(model="nomic-embed-text")

    mock_vectors = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]

    with patch.object(emb, "_call_api", return_value=mock_vectors):
        result = emb.embed(["first", "second", "third"])

    assert len(result) == 3
    assert result == mock_vectors


def test_ollama_default_base_url():
    """Ollama defaults to http://localhost:11434."""
    with patch.dict("os.environ", {}, clear=True):
        emb = OllamaEmbedding()
        assert emb.base_url == "http://localhost:11434"


def test_ollama_env_var_base_url():
    """Ollama uses OLLAMA_BASE_URL env var if set."""
    with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://custom:8080"}):
        emb = OllamaEmbedding()
        assert emb.base_url == "http://custom:8080"


def test_ollama_param_base_url_overrides_env():
    """Ollama base_url parameter overrides env var."""
    with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://env:8080"}):
        emb = OllamaEmbedding(base_url="http://param:9090")
        assert emb.base_url == "http://param:9090"


def test_ollama_connection_error():
    """Ollama raises OllamaEmbeddingError on connection failure."""
    emb = OllamaEmbedding()

    import httpx

    with patch.object(
        emb,
        "_call_api",
        side_effect=OllamaEmbeddingError("Ollama connection error: Connection refused"),
    ):
        with pytest.raises(OllamaEmbeddingError, match="connection error"):
            emb.embed(["test"])


def test_ollama_http_error():
    """Ollama raises OllamaEmbeddingError on HTTP error."""
    emb = OllamaEmbedding()

    with patch.object(
        emb,
        "_call_api",
        side_effect=OllamaEmbeddingError("Ollama HTTP error 500: Internal Server Error"),
    ):
        with pytest.raises(OllamaEmbeddingError, match="HTTP error 500"):
            emb.embed(["test"])


def test_ollama_get_dimension():
    """Ollama get_dimension returns configured dimensions."""
    emb = OllamaEmbedding(dimensions=1024)
    assert emb.get_dimension() == 1024


def test_ollama_no_api_key_needed():
    """Ollama can be instantiated without any API key."""
    # Should not raise
    emb = OllamaEmbedding(model="nomic-embed-text")
    assert emb.model == "nomic-embed-text"


def test_ollama_validates_texts():
    """Ollama validates texts before API call."""
    emb = OllamaEmbedding()

    with pytest.raises(ValueError, match="Texts list cannot be empty"):
        emb.embed([])

    with pytest.raises(ValueError, match="not a string"):
        emb.embed([42])  # type: ignore

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        emb.embed(["   "])


def test_ollama_call_api_integration():
    """Test _call_api with mocked httpx.Client."""
    emb = OllamaEmbedding(model="nomic-embed-text", base_url="http://localhost:11434")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2], [0.3, 0.4]]
    }

    import httpx

    with patch("httpx.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client.post.return_value = mock_response
        MockClient.return_value = mock_client

        result = emb._call_api(["hello", "world"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_client.post.assert_called_once_with(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": ["hello", "world"]},
    )


def test_ollama_call_api_missing_embeddings_key():
    """_call_api raises error if response lacks 'embeddings' key."""
    emb = OllamaEmbedding()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"wrong_key": []}

    import httpx

    with patch("httpx.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client.post.return_value = mock_response
        MockClient.return_value = mock_client

        with pytest.raises(OllamaEmbeddingError, match="unexpected response format"):
            emb._call_api(["test"])
