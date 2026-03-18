"""Verify resilience stack wiring."""
from unittest.mock import MagicMock, patch
from src.libs.llm.llm_factory import LLMFactory
from src.libs.circuit_breaker.provider_chain import ProviderChain


def test_create_with_failover_returns_provider_chain():
    """When fallback_providers configured, returns ProviderChain."""
    from src.core.settings import LLMSettings, CircuitBreakerSettings, FallbackProviderSettings
    settings = LLMSettings(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1024,
        api_key="test-key",
        circuit_breaker=CircuitBreakerSettings(enabled=True, failure_threshold=3, cooldown_seconds=30),
        fallback_providers=[
            FallbackProviderSettings(provider="deepseek", model="deepseek-chat", api_key="test-key-2"),
        ],
    )
    factory = LLMFactory()
    from src.libs.llm.openai_llm import OpenAILLM
    from src.libs.llm.deepseek_llm import DeepSeekLLM
    factory.register_provider("openai", OpenAILLM)
    factory.register_provider("deepseek", DeepSeekLLM)
    result = factory.create_with_failover(settings)
    assert isinstance(result, ProviderChain)


def test_create_without_failover_returns_base_llm():
    """When no fallback, returns plain BaseLLM."""
    from src.core.settings import LLMSettings
    from src.libs.llm.base_llm import BaseLLM
    settings = LLMSettings(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1024,
        api_key="test-key",
    )
    factory = LLMFactory()
    from src.libs.llm.openai_llm import OpenAILLM
    factory.register_provider("openai", OpenAILLM)
    result = factory.create_with_failover(settings)
    assert isinstance(result, BaseLLM)
    assert not isinstance(result, ProviderChain)
