"""Multi-provider failover chain with circuit breaker integration."""

from __future__ import annotations

import logging
from typing import Any

from src.libs.circuit_breaker.circuit_breaker import CircuitBreaker
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message

logger = logging.getLogger(__name__)


class AllProvidersUnavailableError(Exception):
    """Raised when all providers in the chain are unavailable."""


class ProviderChain:
    """Tries LLM providers in priority order, skipping circuit-open ones.

    Args:
        providers: List of (BaseLLM, CircuitBreaker) tuples in priority order.
    """

    def __init__(self, providers: list[tuple[BaseLLM, CircuitBreaker]]) -> None:
        if not providers:
            raise ValueError("ProviderChain requires at least one provider")
        self._providers = providers

    def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        errors: list[tuple[str, Exception]] = []

        for llm, breaker in self._providers:
            if not breaker.allow_request():
                logger.info("Skipping %s (circuit open)", llm.__class__.__name__)
                continue
            try:
                result = llm.chat(messages, **kwargs)
                breaker.record_success()
                return result
            except Exception as exc:
                breaker.record_failure()
                name = llm.__class__.__name__
                logger.warning("Provider %s failed: %s", name, exc)
                errors.append((name, exc))

        raise AllProvidersUnavailableError(
            f"All {len(self._providers)} providers unavailable. "
            f"Errors: {errors}"
        )
