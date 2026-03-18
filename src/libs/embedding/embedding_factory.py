"""Embedding Factory — configuration-driven provider routing.

Uses a registry pattern so new providers can be added without
modifying this module. Providers register via
:meth:`EmbeddingFactory.register_provider` and are instantiated via
:meth:`EmbeddingFactory.create`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.core.settings import EmbeddingSettings
from src.libs.embedding.base_embedding import BaseEmbedding


class EmbeddingFactory:
    """Factory for creating Embedding instances by provider name.

    Example::

        factory = EmbeddingFactory()
        factory.register_provider("openai", OpenAIEmbedding)
        emb = factory.create("openai", model="text-embedding-3-small")
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseEmbedding]] = {}

    def register_provider(
        self,
        name: str,
        provider_class: type[BaseEmbedding],
    ) -> None:
        """Register an Embedding provider class.

        Args:
            name: Provider identifier (case-insensitive).
            provider_class: A subclass of BaseEmbedding.

        Raises:
            TypeError: If provider_class is not a BaseEmbedding subclass.
        """
        if not (isinstance(provider_class, type) and issubclass(provider_class, BaseEmbedding)):
            raise TypeError(f"{provider_class} must be a subclass of BaseEmbedding")
        self._providers[name.lower()] = provider_class

    def create(self, provider: str, **kwargs: Any) -> BaseEmbedding:
        """Create an Embedding instance by provider name.

        Args:
            provider: Provider identifier (case-insensitive).
            **kwargs: Passed to the provider constructor.

        Returns:
            An instance of the requested Embedding provider.

        Raises:
            ValueError: If the provider is not registered.
        """
        key = provider.lower()
        cls = self._providers.get(key)
        if cls is None:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise ValueError(f"Unknown Embedding provider '{provider}'. Available: {available}")
        return cls(**kwargs)

    def create_from_settings(self, settings: EmbeddingSettings) -> BaseEmbedding:
        """Create an Embedding instance from an EmbeddingSettings object.

        Extracts ``provider`` for routing, and forwards remaining
        fields as keyword arguments to the provider constructor.

        Args:
            settings: Parsed Embedding configuration.

        Returns:
            An instance of the configured Embedding provider.
        """
        fields = asdict(settings)
        provider = fields.pop("provider")
        # circuit_breaker is a factory-level concern; providers don't accept it
        fields.pop("circuit_breaker", None)
        kwargs = {k: v for k, v in fields.items() if v is not None}
        return self.create(provider, **kwargs)

    def create_with_failover(
        self,
        settings: EmbeddingSettings,
    ) -> BaseEmbedding:
        """Create an Embedding with optional circuit breaker wrapping.

        When ``circuit_breaker`` is configured on *settings*, the provider is
        wrapped inside an :class:`EmbeddingChain` so that circuit-breaker
        state is tracked.  When no circuit breaker is configured, returns a
        plain :class:`BaseEmbedding` instance.

        Multi-provider fallback for embeddings is reserved for future use;
        the chain infrastructure (``EmbeddingChain``) is already available.

        Args:
            settings: Parsed Embedding configuration with optional
                ``circuit_breaker`` sub-config.

        Returns:
            A single ``BaseEmbedding`` or a single-entry ``EmbeddingChain``.
        """
        primary = self.create_from_settings(settings)

        # No circuit breaker config → return plain provider
        if settings.circuit_breaker is None:
            return primary

        # Lazy import to avoid circular dependency at module level
        from src.libs.circuit_breaker.circuit_breaker import CircuitBreaker
        from src.libs.circuit_breaker.embedding_chain import EmbeddingChain

        cb_cfg = settings.circuit_breaker
        if cb_cfg.enabled:
            breaker = CircuitBreaker(
                failure_threshold=cb_cfg.failure_threshold,
                cooldown=cb_cfg.cooldown_seconds,
            )
        else:
            # Disabled config → lenient default (effectively no protection)
            breaker = CircuitBreaker(failure_threshold=5, cooldown=60.0)

        return EmbeddingChain([(primary, breaker)])

    def list_providers(self) -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._providers)
