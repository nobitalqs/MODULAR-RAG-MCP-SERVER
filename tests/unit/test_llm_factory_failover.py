"""Unit tests for LLMFactory.create_with_failover() — circuit breaker integration."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import (
    CircuitBreakerSettings,
    FallbackProviderSettings,
    LLMSettings,
)
from src.libs.circuit_breaker.provider_chain import ProviderChain
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory


def _make_factory() -> LLMFactory:
    factory = LLMFactory()
    factory.register_provider("openai", type("FakeOpenAI", (BaseLLM,), {
        "chat": lambda self, messages, **kw: None,
        "__init__": lambda self, **kw: None,
    }))
    factory.register_provider("ollama", type("FakeOllama", (BaseLLM,), {
        "chat": lambda self, messages, **kw: None,
        "__init__": lambda self, **kw: None,
    }))
    return factory


class TestCreateWithFailover:
    def test_no_circuit_breaker_returns_single_llm(self):
        factory = _make_factory()
        settings = LLMSettings(
            provider="openai", model="gpt-4o", temperature=0.0, max_tokens=1024,
        )
        result = factory.create_with_failover(settings)
        assert isinstance(result, BaseLLM)
        assert not isinstance(result, ProviderChain)

    def test_circuit_breaker_without_fallbacks_returns_single_llm(self):
        factory = _make_factory()
        settings = LLMSettings(
            provider="openai", model="gpt-4o", temperature=0.0, max_tokens=1024,
            circuit_breaker=CircuitBreakerSettings(
                enabled=True, failure_threshold=3, cooldown_seconds=30.0,
            ),
        )
        result = factory.create_with_failover(settings)
        assert isinstance(result, BaseLLM)

    def test_with_fallbacks_returns_provider_chain(self):
        factory = _make_factory()
        settings = LLMSettings(
            provider="openai", model="gpt-4o", temperature=0.0, max_tokens=1024,
            circuit_breaker=CircuitBreakerSettings(
                enabled=True, failure_threshold=3, cooldown_seconds=30.0,
            ),
            fallback_providers=[
                FallbackProviderSettings(provider="ollama", model="llama3"),
            ],
        )
        result = factory.create_with_failover(settings)
        assert isinstance(result, ProviderChain)

    def test_disabled_circuit_breaker_with_fallbacks_returns_chain(self):
        """Even with CB disabled, fallbacks still create a chain."""
        factory = _make_factory()
        settings = LLMSettings(
            provider="openai", model="gpt-4o", temperature=0.0, max_tokens=1024,
            circuit_breaker=CircuitBreakerSettings(
                enabled=False, failure_threshold=3, cooldown_seconds=30.0,
            ),
            fallback_providers=[
                FallbackProviderSettings(provider="ollama", model="llama3"),
            ],
        )
        result = factory.create_with_failover(settings)
        assert isinstance(result, ProviderChain)
