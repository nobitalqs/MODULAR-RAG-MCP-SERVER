"""Integration tests for retry_with_backoff applied to Embedding providers.

Verifies that each provider's _call_api retries on transient errors (429, 500)
and does NOT retry on non-transient errors (400).

Strategy:
- For OpenAI/Azure: mock _call_api directly (SDK calls, not httpx).
  The @retry_with_backoff on _call_api is exercised by patching _call_api's
  inner SDK call via the client mock.
- For Ollama: patch httpx.Client.post so the real _call_api (with its
  @retry_with_backoff decorator) is exercised end-to-end.
- time.sleep is patched in all retry tests to keep the suite fast.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.libs.embedding.azure_embedding import AzureEmbedding, AzureEmbeddingError
from src.libs.embedding.ollama_embedding import OllamaEmbedding, OllamaEmbeddingError
from src.libs.embedding.openai_embedding import OpenAIEmbedding, OpenAIEmbeddingError
from src.libs.resilience.retry import RetryableError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VECTORS = [[0.1, 0.2, 0.3]]


def _make_openai_api_error(status_code: int):
    """Build a mock openai.APIStatusError with the given status code."""
    import openai

    request = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    return openai.APIStatusError(
        message=f"HTTP {status_code}",
        response=response,
        body=None,
    )


def _make_http_error(status_code: int) -> httpx.HTTPStatusError:
    """Build an httpx.HTTPStatusError with the given status code."""
    request = httpx.Request("POST", "http://test")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


def _mock_httpx_response(json_data: dict) -> MagicMock:
    """Build a mock httpx.Response that returns json_data."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = json_data
    return mock


# ---------------------------------------------------------------------------
# Error class inheritance tests
# ---------------------------------------------------------------------------


def test_openai_embedding_error_is_retryable() -> None:
    """OpenAIEmbeddingError must inherit from RetryableError for retry to work."""
    assert issubclass(OpenAIEmbeddingError, RetryableError)


def test_azure_embedding_error_is_retryable() -> None:
    """AzureEmbeddingError must inherit from RetryableError for retry to work."""
    assert issubclass(AzureEmbeddingError, RetryableError)


def test_ollama_embedding_error_is_retryable() -> None:
    """OllamaEmbeddingError must inherit from RetryableError for retry to work."""
    assert issubclass(OllamaEmbeddingError, RetryableError)


# ---------------------------------------------------------------------------
# OpenAI Embedding retry tests
# ---------------------------------------------------------------------------


def test_openai_embedding_retries_on_429() -> None:
    """OpenAI _call_api retries on 429 then succeeds on 3rd attempt."""
    emb = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")

    api_error_429 = _make_openai_api_error(429)
    call_count = 0

    def fake_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise api_error_429
        item = MagicMock()
        item.embedding = [0.1, 0.2, 0.3]
        result = MagicMock()
        result.data = [item]
        return result

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = fake_create

    with patch("time.sleep"), patch.object(emb, "_create_client", return_value=mock_client):
        result = emb.embed(["hello"])

    assert result == [[0.1, 0.2, 0.3]]
    assert call_count == 3


def test_openai_embedding_no_retry_on_400() -> None:
    """OpenAI _call_api does NOT retry on 400 Bad Request (non-retryable)."""
    emb = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")

    api_error_400 = _make_openai_api_error(400)
    call_count = 0

    def fake_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise api_error_400

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = fake_create

    with patch.object(emb, "_create_client", return_value=mock_client):
        with pytest.raises(OpenAIEmbeddingError):
            emb.embed(["hello"])

    assert call_count == 1


def test_openai_embedding_exhausts_retries_on_500() -> None:
    """OpenAI raises after exhausting all retries on persistent 500 errors."""
    emb = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")

    api_error_500 = _make_openai_api_error(500)
    call_count = 0

    def fake_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise api_error_500

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = fake_create

    with patch("time.sleep"), patch.object(emb, "_create_client", return_value=mock_client):
        with pytest.raises(OpenAIEmbeddingError):
            emb.embed(["hello"])

    # 1 initial + 3 retries = 4 calls
    assert call_count == 4


# ---------------------------------------------------------------------------
# Azure Embedding retry tests
# ---------------------------------------------------------------------------


def test_azure_embedding_retries_on_429() -> None:
    """Azure _call_api retries on 429 then succeeds on 3rd attempt."""
    emb = AzureEmbedding(
        model="text-embedding-ada-002",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
    )

    api_error_429 = _make_openai_api_error(429)
    call_count = 0

    def fake_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise api_error_429
        item = MagicMock()
        item.embedding = [0.4, 0.5, 0.6]
        result = MagicMock()
        result.data = [item]
        return result

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = fake_create

    with patch("time.sleep"), patch.object(emb, "_create_client", return_value=mock_client):
        result = emb.embed(["world"])

    assert result == [[0.4, 0.5, 0.6]]
    assert call_count == 3


def test_azure_embedding_no_retry_on_400() -> None:
    """Azure _call_api does NOT retry on 400 Bad Request (non-retryable)."""
    emb = AzureEmbedding(
        model="text-embedding-ada-002",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
    )

    api_error_400 = _make_openai_api_error(400)
    call_count = 0

    def fake_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise api_error_400

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = fake_create

    with patch.object(emb, "_create_client", return_value=mock_client):
        with pytest.raises(AzureEmbeddingError):
            emb.embed(["world"])

    assert call_count == 1


# ---------------------------------------------------------------------------
# Ollama Embedding retry tests
# ---------------------------------------------------------------------------


def test_ollama_embedding_retries_on_429() -> None:
    """Ollama _call_api retries on 429 then succeeds on 3rd attempt."""
    emb = OllamaEmbedding(model="nomic-embed-text")

    http_429 = _make_http_error(429)
    ok_response = _mock_httpx_response({"embeddings": [[0.7, 0.8, 0.9]]})

    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise http_429
        return ok_response

    with patch("time.sleep"), patch("httpx.Client.post", side_effect=fake_post):
        result = emb.embed(["test"])

    assert result == [[0.7, 0.8, 0.9]]
    assert call_count == 3


def test_ollama_embedding_no_retry_on_400() -> None:
    """Ollama _call_api does NOT retry on 400 Bad Request (non-retryable)."""
    emb = OllamaEmbedding(model="nomic-embed-text")

    http_400 = _make_http_error(400)
    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise http_400

    with patch("httpx.Client.post", side_effect=fake_post):
        with pytest.raises(OllamaEmbeddingError):
            emb.embed(["test"])

    assert call_count == 1


def test_ollama_embedding_exhausts_retries_on_503() -> None:
    """Ollama raises after exhausting all retries on persistent 503 errors."""
    emb = OllamaEmbedding(model="nomic-embed-text")

    http_503 = _make_http_error(503)
    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise http_503

    with patch("time.sleep"), patch("httpx.Client.post", side_effect=fake_post):
        with pytest.raises(OllamaEmbeddingError):
            emb.embed(["test"])

    # 1 initial + 3 retries = 4 calls
    assert call_count == 4
