"""Unit tests for ProviderChain — multi-provider failover."""

from unittest.mock import MagicMock

import pytest

from src.libs.circuit_breaker.circuit_breaker import CircuitBreaker, CircuitState
from src.libs.circuit_breaker.provider_chain import (
    AllProvidersUnavailableError,
    ProviderChain,
)
from src.libs.llm.base_llm import ChatResponse, Message


def _make_llm(name: str, fail: bool = False) -> MagicMock:
    llm = MagicMock()
    llm.__class__.__name__ = name
    if fail:
        llm.chat.side_effect = RuntimeError(f"{name} down")
    else:
        llm.chat.return_value = ChatResponse(content=f"from {name}", model=name)
    return llm


class TestProviderChain:
    def test_first_provider_succeeds(self):
        llm1 = _make_llm("primary")
        chain = ProviderChain([(llm1, CircuitBreaker())])
        msgs = [Message("user", "hi")]
        result = chain.chat(msgs)
        assert result.content == "from primary"

    def test_failover_to_second(self):
        llm1 = _make_llm("primary", fail=True)
        llm2 = _make_llm("backup")
        chain = ProviderChain([
            (llm1, CircuitBreaker(failure_threshold=1)),
            (llm2, CircuitBreaker()),
        ])
        msgs = [Message("user", "hi")]
        result = chain.chat(msgs)
        assert result.content == "from backup"

    def test_skips_circuit_open_provider(self):
        llm1 = _make_llm("primary")
        cb1 = CircuitBreaker(failure_threshold=1)
        cb1.record_failure()  # opens circuit
        assert cb1.state == CircuitState.OPEN

        llm2 = _make_llm("backup")
        chain = ProviderChain([(llm1, cb1), (llm2, CircuitBreaker())])
        result = chain.chat([Message("user", "hi")])
        assert result.content == "from backup"
        llm1.chat.assert_not_called()

    def test_all_unavailable_raises(self):
        llm1 = _make_llm("p1", fail=True)
        llm2 = _make_llm("p2", fail=True)
        chain = ProviderChain([
            (llm1, CircuitBreaker(failure_threshold=1)),
            (llm2, CircuitBreaker(failure_threshold=1)),
        ])
        with pytest.raises(AllProvidersUnavailableError):
            chain.chat([Message("user", "hi")])

    def test_kwargs_forwarded(self):
        llm1 = _make_llm("primary")
        chain = ProviderChain([(llm1, CircuitBreaker())])
        chain.chat([Message("user", "hi")], temperature=0.5)
        llm1.chat.assert_called_once_with([Message("user", "hi")], temperature=0.5)
