"""Integration tests for retry_with_backoff applied to LLM providers.

Verifies that each provider's _call_api retries on transient errors (429, 500, 502)
and does NOT retry on non-transient errors (400, 404).

Strategy: use unittest.mock.patch to replace httpx.Client.post so that the real
_call_api (with its @retry_with_backoff decorator) is exercised.
time.sleep is patched in all retry tests to keep the suite fast.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.libs.llm.azure_llm import AzureLLM, AzureLLMError
from src.libs.llm.base_llm import Message
from src.libs.llm.deepseek_llm import DeepSeekLLM, DeepSeekLLMError
from src.libs.llm.ollama_llm import OllamaLLM, OllamaLLMError
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError
from src.libs.resilience.retry import RetryableError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPENAI_OK = {
    "choices": [{"message": {"content": "ok"}}],
    "model": "gpt-4o",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_AZURE_OK = {
    "choices": [{"message": {"content": "ok"}}],
    "model": "gpt-4o",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_DEEPSEEK_OK = {
    "choices": [{"message": {"content": "ok"}}],
    "model": "deepseek-chat",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_OLLAMA_OK = {
    "message": {"content": "ok"},
    "model": "llama2",
}


def _make_http_error(status_code: int) -> httpx.HTTPStatusError:
    """Build an httpx.HTTPStatusError with the given status code."""
    request = httpx.Request("POST", "http://test")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


def _mock_response(json_data: dict) -> MagicMock:
    """Build a mock httpx.Response that returns json_data."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = json_data
    return mock


# ---------------------------------------------------------------------------
# Error class inheritance tests
# ---------------------------------------------------------------------------


def test_openai_error_is_retryable_error() -> None:
    """OpenAILLMError must inherit from RetryableError for retry to work."""
    assert issubclass(OpenAILLMError, RetryableError)


def test_azure_error_is_retryable_error() -> None:
    """AzureLLMError must inherit from RetryableError for retry to work."""
    assert issubclass(AzureLLMError, RetryableError)


def test_deepseek_error_is_retryable_error() -> None:
    """DeepSeekLLMError must inherit from RetryableError for retry to work."""
    assert issubclass(DeepSeekLLMError, RetryableError)


def test_ollama_error_is_retryable_error() -> None:
    """OllamaLLMError must inherit from RetryableError for retry to work."""
    assert issubclass(OllamaLLMError, RetryableError)


# ---------------------------------------------------------------------------
# OpenAI retry tests
# ---------------------------------------------------------------------------


def test_openai_retries_on_429() -> None:
    """OpenAI _call_api retries twice on 429 then succeeds on 3rd attempt."""
    llm = OpenAILLM(model="gpt-4o", api_key="test-key")
    messages = [Message(role="user", content="Hello")]

    http_429 = _make_http_error(429)
    ok_response = _mock_response(_OPENAI_OK)

    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise http_429
        return ok_response

    with patch("time.sleep"), patch("httpx.Client.post", side_effect=fake_post):
        response = llm.chat(messages)

    assert response.content == "ok"
    assert call_count == 3


def test_openai_no_retry_on_400() -> None:
    """OpenAI _call_api does NOT retry on 400 Bad Request (non-retryable)."""
    llm = OpenAILLM(model="gpt-4o", api_key="test-key")
    messages = [Message(role="user", content="Hello")]

    http_400 = _make_http_error(400)
    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise http_400

    with patch("httpx.Client.post", side_effect=fake_post):
        with pytest.raises(OpenAILLMError):
            llm.chat(messages)

    assert call_count == 1


def test_openai_exhausts_retries_on_500() -> None:
    """OpenAI raises after exhausting all retries on persistent 500 errors."""
    llm = OpenAILLM(model="gpt-4o", api_key="test-key")
    messages = [Message(role="user", content="Hello")]

    http_500 = _make_http_error(500)
    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise http_500

    with patch("time.sleep"), patch("httpx.Client.post", side_effect=fake_post):
        with pytest.raises(OpenAILLMError):
            llm.chat(messages)

    # 1 initial + 3 retries = 4 calls
    assert call_count == 4


# ---------------------------------------------------------------------------
# DeepSeek retry tests
# ---------------------------------------------------------------------------


def test_deepseek_retries_on_500() -> None:
    """DeepSeek _call_api retries twice on 500 then succeeds on 3rd attempt."""
    llm = DeepSeekLLM(model="deepseek-chat", api_key="test-key")
    messages = [Message(role="user", content="Hello")]

    http_500 = _make_http_error(500)
    ok_response = _mock_response(_DEEPSEEK_OK)

    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise http_500
        return ok_response

    with patch("time.sleep"), patch("httpx.Client.post", side_effect=fake_post):
        response = llm.chat(messages)

    assert response.content == "ok"
    assert call_count == 3


def test_deepseek_no_retry_on_401() -> None:
    """DeepSeek _call_api does NOT retry on 401 Unauthorized."""
    llm = DeepSeekLLM(model="deepseek-chat", api_key="test-key")
    messages = [Message(role="user", content="Hello")]

    http_401 = _make_http_error(401)
    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise http_401

    with patch("httpx.Client.post", side_effect=fake_post):
        with pytest.raises(DeepSeekLLMError):
            llm.chat(messages)

    assert call_count == 1


# ---------------------------------------------------------------------------
# Azure retry tests
# ---------------------------------------------------------------------------


def test_azure_retries_on_502() -> None:
    """Azure _call_api retries twice on 502 then succeeds on 3rd attempt."""
    llm = AzureLLM(
        model="gpt-4o",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment_name="gpt-4o-deployment",
    )
    messages = [Message(role="user", content="Hello")]

    http_502 = _make_http_error(502)
    ok_response = _mock_response(_AZURE_OK)

    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise http_502
        return ok_response

    with patch("time.sleep"), patch("httpx.Client.post", side_effect=fake_post):
        response = llm.chat(messages)

    assert response.content == "ok"
    assert call_count == 3


def test_azure_no_retry_on_403() -> None:
    """Azure _call_api does NOT retry on 403 Forbidden."""
    llm = AzureLLM(
        model="gpt-4o",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment_name="gpt-4o-deployment",
    )
    messages = [Message(role="user", content="Hello")]

    http_403 = _make_http_error(403)
    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise http_403

    with patch("httpx.Client.post", side_effect=fake_post):
        with pytest.raises(AzureLLMError):
            llm.chat(messages)

    assert call_count == 1


# ---------------------------------------------------------------------------
# Ollama retry tests
# ---------------------------------------------------------------------------


def test_ollama_retries_on_503() -> None:
    """Ollama _call_api retries twice on 503 then succeeds on 3rd attempt."""
    llm = OllamaLLM(model="llama2")
    messages = [Message(role="user", content="Hello")]

    http_503 = _make_http_error(503)
    ok_response = _mock_response(_OLLAMA_OK)

    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise http_503
        return ok_response

    with patch("time.sleep"), patch("httpx.Client.post", side_effect=fake_post):
        response = llm.chat(messages)

    assert response.content == "ok"
    assert call_count == 3


def test_ollama_no_retry_on_404() -> None:
    """Ollama _call_api does NOT retry on 404 Not Found."""
    llm = OllamaLLM(model="llama2")
    messages = [Message(role="user", content="Hello")]

    http_404 = _make_http_error(404)
    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise http_404

    with patch("httpx.Client.post", side_effect=fake_post):
        with pytest.raises(OllamaLLMError):
            llm.chat(messages)

    assert call_count == 1
