"""Integration test: CircuitBreaker + ProviderChain + LLM.

Verifies that the circuit breaker trips after repeated failures, and the
ProviderChain correctly fails over to backup providers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.libs.circuit_breaker.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from src.libs.circuit_breaker.provider_chain import (
    AllProvidersUnavailableError,
    ProviderChain,
)
from src.libs.llm.base_llm import ChatResponse, Message


def _make_llm(name: str, succeed: bool = True):
    mock = MagicMock()
    mock.__class__.__name__ = name
    if succeed:
        mock.chat.return_value = ChatResponse(
            content=f"Response from {name}", model=name,
        )
    else:
        mock.chat.side_effect = RuntimeError(f"{name} is down")
    return mock


class TestCircuitBreakerWithProviderChain:
    """ProviderChain + CircuitBreaker end-to-end failover."""

    def test_primary_succeeds(self):
        primary = _make_llm("Primary")
        fallback = _make_llm("Fallback")
        chain = ProviderChain([
            (primary, CircuitBreaker(failure_threshold=3)),
            (fallback, CircuitBreaker(failure_threshold=3)),
        ])

        result = chain.chat([Message("user", "Hello")])
        assert result.content == "Response from Primary"
        assert primary.chat.call_count == 1
        assert fallback.chat.call_count == 0

    def test_failover_on_primary_failure(self):
        primary = _make_llm("Primary", succeed=False)
        fallback = _make_llm("Fallback")
        chain = ProviderChain([
            (primary, CircuitBreaker(failure_threshold=3)),
            (fallback, CircuitBreaker(failure_threshold=3)),
        ])

        result = chain.chat([Message("user", "Hello")])
        assert result.content == "Response from Fallback"
        assert primary.chat.call_count == 1
        assert fallback.chat.call_count == 1

    def test_circuit_trips_after_threshold(self):
        breaker = CircuitBreaker(failure_threshold=2, cooldown=60.0)
        primary = _make_llm("Primary", succeed=False)
        fallback = _make_llm("Fallback")
        chain = ProviderChain([
            (primary, breaker),
            (fallback, CircuitBreaker(failure_threshold=3)),
        ])

        # First two calls fail primary, but fallback succeeds
        for _ in range(2):
            result = chain.chat([Message("user", "Hello")])
            assert result.content == "Response from Fallback"

        # Primary breaker should now be OPEN
        assert breaker.state == CircuitState.OPEN

        # Third call skips primary entirely (circuit open)
        primary.chat.reset_mock()
        result = chain.chat([Message("user", "Hello")])
        assert result.content == "Response from Fallback"
        assert primary.chat.call_count == 0  # skipped

    def test_all_providers_down_raises(self):
        primary = _make_llm("Primary", succeed=False)
        fallback = _make_llm("Fallback", succeed=False)
        chain = ProviderChain([
            (primary, CircuitBreaker(failure_threshold=3)),
            (fallback, CircuitBreaker(failure_threshold=3)),
        ])

        with pytest.raises(AllProvidersUnavailableError):
            chain.chat([Message("user", "Hello")])

    def test_circuit_breaker_recovers_in_half_open(self):
        breaker = CircuitBreaker(failure_threshold=1, cooldown=0.0)
        failing_llm = _make_llm("Failing", succeed=False)

        # Trip the breaker
        try:
            protected = breaker.protect(failing_llm.chat)
            protected([Message("user", "Hi")])
        except RuntimeError:
            pass

        # cooldown=0.0 means state property immediately transitions OPEN -> HALF_OPEN
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.allow_request() is True

        # Successful call resets to CLOSED
        succeeding_llm = _make_llm("Recovered")
        protected_good = breaker.protect(succeeding_llm.chat)
        protected_good([Message("user", "Hi")])
        assert breaker.state == CircuitState.CLOSED
