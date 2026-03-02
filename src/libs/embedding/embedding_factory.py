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
            raise TypeError(
                f"{provider_class} must be a subclass of BaseEmbedding"
            )
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
            raise ValueError(
                f"Unknown Embedding provider '{provider}'. "
                f"Available: {available}"
            )
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
        kwargs = {k: v for k, v in fields.items() if v is not None}
        return self.create(provider, **kwargs)

    def list_providers(self) -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._providers)
