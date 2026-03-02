"""Reranker Factory — configuration-driven provider routing.

Uses a registry pattern so new providers can be added without
modifying this module. Providers register via
:meth:`RerankerFactory.register_provider` and are instantiated via
:meth:`RerankerFactory.create`.

Special handling: when ``enabled=False`` or ``provider="none"``,
:meth:`create_from_settings` returns a :class:`NoneReranker`
without requiring registry registration.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.core.settings import RerankSettings
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker


class RerankerFactory:
    """Factory for creating Reranker instances by provider name.

    Example::

        factory = RerankerFactory()
        factory.register_provider("cross_encoder", CrossEncoderReranker)
        reranker = factory.create("cross_encoder", model="ms-marco")
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseReranker]] = {}

    def register_provider(
        self,
        name: str,
        provider_class: type[BaseReranker],
    ) -> None:
        """Register a Reranker provider class.

        Args:
            name: Provider identifier (case-insensitive).
            provider_class: A subclass of BaseReranker.

        Raises:
            TypeError: If provider_class is not a BaseReranker subclass.
        """
        if not (isinstance(provider_class, type) and issubclass(provider_class, BaseReranker)):
            raise TypeError(f"{provider_class} must be a subclass of BaseReranker")
        self._providers[name.lower()] = provider_class

    def create(self, provider: str, **kwargs: Any) -> BaseReranker:
        """Create a Reranker instance by provider name.

        Args:
            provider: Provider identifier (case-insensitive).
            **kwargs: Passed to the provider constructor.

        Returns:
            An instance of the requested Reranker provider.

        Raises:
            ValueError: If the provider is not registered.
        """
        key = provider.lower()
        cls = self._providers.get(key)
        if cls is None:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise ValueError(f"Unknown Reranker provider '{provider}'. Available: {available}")
        return cls(**kwargs)

    def create_from_settings(self, settings: RerankSettings) -> BaseReranker:
        """Create a Reranker instance from a RerankSettings object.

        When ``enabled`` is False or ``provider`` is ``"none"``,
        returns a :class:`NoneReranker` directly (no registry lookup).
        Otherwise extracts ``provider`` for routing and forwards
        remaining fields as keyword arguments.

        Args:
            settings: Parsed Rerank configuration.

        Returns:
            An instance of the configured Reranker provider.
        """
        fields = asdict(settings)
        provider = fields.pop("provider")
        enabled = fields.pop("enabled")

        if not enabled or provider.lower() == "none":
            return NoneReranker(**{k: v for k, v in fields.items() if v is not None})

        kwargs = {k: v for k, v in fields.items() if v is not None}
        return self.create(provider, **kwargs)

    def list_providers(self) -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._providers)
