"""Tests for RateLimitedLLM wrapper — TDD (write first, run RED, then GREEN)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.rate_limiter.base_limiter import BaseLimiter
from src.libs.rate_limiter.null_limiter import NullLimiter
from src.libs.resilience.rate_limited_llm import RateLimitedLLM

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM(BaseLLM):
    """Minimal LLM stub that returns a fixed response."""

    def __init__(self, response: ChatResponse | None = None) -> None:
        self._response = response or ChatResponse(
            content="fake response",
            model="fake-model",
            usage={"prompt_tokens": 5, "completion_tokens": 3},
        )
        self.last_messages: list[Message] | None = None
        self.last_trace: Any = None
        self.last_kwargs: dict[str, Any] = {}

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.last_messages = messages
        self.last_trace = trace
        self.last_kwargs = kwargs
        return self._response


class ErrorLLM(BaseLLM):
    """LLM stub that always raises an exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        raise self._exc


def _make_limiter_mock() -> MagicMock:
    """Return a MagicMock that satisfies BaseLimiter interface."""
    limiter = MagicMock(spec=BaseLimiter)
    limiter.acquire.return_value = True
    return limiter


def _messages() -> list[Message]:
    return [Message(role="user", content="hello")]


# ---------------------------------------------------------------------------
# 1. acquire and release both called once on success
# ---------------------------------------------------------------------------


class TestAcquireReleaseOnSuccess:
    def test_acquire_called_before_chat(self) -> None:
        limiter = _make_limiter_mock()
        llm = RateLimitedLLM(FakeLLM(), limiter)

        llm.chat(_messages())

        limiter.acquire.assert_called_once()

    def test_release_called_after_chat(self) -> None:
        limiter = _make_limiter_mock()
        llm = RateLimitedLLM(FakeLLM(), limiter)

        llm.chat(_messages())

        limiter.release.assert_called_once()

    def test_acquire_called_before_release(self) -> None:
        call_order: list[str] = []
        limiter = MagicMock(spec=BaseLimiter)
        limiter.acquire.side_effect = lambda **kw: call_order.append("acquire") or True
        limiter.release.side_effect = lambda: call_order.append("release")

        llm = RateLimitedLLM(FakeLLM(), limiter)
        llm.chat(_messages())

        assert call_order == ["acquire", "release"]


# ---------------------------------------------------------------------------
# 2. release called even when chat() raises
# ---------------------------------------------------------------------------


class TestReleaseOnError:
    def test_release_on_exception(self) -> None:
        limiter = _make_limiter_mock()
        inner_exc = RuntimeError("llm failed")
        llm = RateLimitedLLM(ErrorLLM(inner_exc), limiter)

        with pytest.raises(RuntimeError, match="llm failed"):
            llm.chat(_messages())

        limiter.release.assert_called_once()

    def test_exception_propagates_unchanged(self) -> None:
        limiter = _make_limiter_mock()
        inner_exc = ValueError("bad request")
        llm = RateLimitedLLM(ErrorLLM(inner_exc), limiter)

        with pytest.raises(ValueError, match="bad request"):
            llm.chat(_messages())


# ---------------------------------------------------------------------------
# 3. delegates to inner LLM with correct arguments
# ---------------------------------------------------------------------------


class TestDelegatesToInnerLLM:
    def test_returns_inner_response(self) -> None:
        expected = ChatResponse(content="delegated", model="inner-model")
        llm = RateLimitedLLM(FakeLLM(response=expected), _make_limiter_mock())

        result = llm.chat(_messages())

        assert result == expected

    def test_passes_messages_through(self) -> None:
        inner = FakeLLM()
        llm = RateLimitedLLM(inner, _make_limiter_mock())
        msgs = _messages()

        llm.chat(msgs)

        assert inner.last_messages == msgs

    def test_passes_trace_through(self) -> None:
        inner = FakeLLM()
        llm = RateLimitedLLM(inner, _make_limiter_mock())
        sentinel_trace = object()

        llm.chat(_messages(), trace=sentinel_trace)

        assert inner.last_trace is sentinel_trace

    def test_passes_kwargs_through(self) -> None:
        inner = FakeLLM()
        llm = RateLimitedLLM(inner, _make_limiter_mock())

        llm.chat(_messages(), temperature=0.7, max_tokens=256)

        assert inner.last_kwargs == {"temperature": 0.7, "max_tokens": 256}


# ---------------------------------------------------------------------------
# 4. validate_messages delegates to inner
# ---------------------------------------------------------------------------


class TestValidateDelegates:
    def test_delegates_valid_messages(self) -> None:
        inner = MagicMock(spec=BaseLLM)
        llm = RateLimitedLLM(inner, _make_limiter_mock())
        msgs = _messages()

        llm.validate_messages(msgs)

        inner.validate_messages.assert_called_once_with(msgs)

    def test_propagates_validation_error(self) -> None:
        inner = MagicMock(spec=BaseLLM)
        inner.validate_messages.side_effect = ValueError("invalid")
        llm = RateLimitedLLM(inner, _make_limiter_mock())

        with pytest.raises(ValueError, match="invalid"):
            llm.validate_messages(_messages())


# ---------------------------------------------------------------------------
# 5. works transparently with NullLimiter
# ---------------------------------------------------------------------------


class TestWithNullLimiter:
    def test_chat_returns_response(self) -> None:
        expected = ChatResponse(content="null limiter ok", model="test-model")
        llm = RateLimitedLLM(FakeLLM(response=expected), NullLimiter())

        result = llm.chat(_messages())

        assert result == expected

    def test_no_rate_limit_raised(self) -> None:
        llm = RateLimitedLLM(FakeLLM(), NullLimiter())

        # Should not raise any exception
        for _ in range(10):
            llm.chat(_messages())
