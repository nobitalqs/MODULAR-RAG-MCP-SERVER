"""Multi-provider failover chain with circuit breaker integration for embeddings."""

from __future__ import annotations

import logging
from typing import Any

from src.libs.circuit_breaker.circuit_breaker import CircuitBreaker
from src.libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class AllEmbeddingProvidersUnavailableError(Exception):
    """Raised when all embedding providers in the chain are unavailable."""


class EmbeddingChain(BaseEmbedding):
    """Tries Embedding providers in priority order, skipping circuit-open ones.

    Implements the same ``BaseEmbedding`` interface so it can be used as a
    drop-in replacement wherever a single ``BaseEmbedding`` is expected.

    Args:
        providers: List of (BaseEmbedding, CircuitBreaker) tuples in priority order.

    Raises:
        ValueError: If the providers list is empty.
    """

    def __init__(self, providers: list[tuple[BaseEmbedding, CircuitBreaker]]) -> None:
        if not providers:
            raise ValueError("EmbeddingChain requires at least one provider")
        self._providers = providers

    def embed(
        self,
        texts: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings, failing over to the next available provider.

        Args:
            texts: Non-empty list of text strings to embed.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific overrides forwarded as-is.

        Returns:
            List of embedding vectors from the first successful provider.

        Raises:
            AllEmbeddingProvidersUnavailableError: If every provider is either
                circuit-open or raises an exception.
        """
        errors: list[tuple[str, Exception]] = []

        for emb, breaker in self._providers:
            if not breaker.allow_request():
                logger.info("Skipping %s (circuit open)", emb.__class__.__name__)
                continue
            try:
                result = emb.embed(texts, **kwargs)
                breaker.record_success()
                return result
            except Exception as exc:
                breaker.record_failure()
                name = emb.__class__.__name__
                logger.warning("Embedding provider %s failed: %s", name, exc)
                errors.append((name, exc))

        raise AllEmbeddingProvidersUnavailableError(
            f"All {len(self._providers)} embedding providers unavailable. Errors: {errors}"
        )

    def get_dimension(self) -> int:
        """Return the dimensionality from the first available (circuit-closed) provider.

        Raises:
            AllEmbeddingProvidersUnavailableError: If no provider is available.
        """
        for emb, breaker in self._providers:
            if breaker.allow_request():
                return emb.get_dimension()

        raise AllEmbeddingProvidersUnavailableError(
            "All embedding providers are circuit-open; cannot determine dimension"
        )
