"""VectorStore Factory — configuration-driven provider routing.

Uses a registry pattern so new providers can be added without
modifying this module. Providers register via
:meth:`VectorStoreFactory.register_provider` and are instantiated via
:meth:`VectorStoreFactory.create`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.core.settings import VectorStoreSettings
from src.libs.vector_store.base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """Factory for creating VectorStore instances by provider name.

    Example::

        factory = VectorStoreFactory()
        factory.register_provider("chroma", ChromaStore)
        store = factory.create("chroma", persist_directory="./data")
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseVectorStore]] = {}

    def register_provider(
        self,
        name: str,
        provider_class: type[BaseVectorStore],
    ) -> None:
        """Register a VectorStore provider class.

        Args:
            name: Provider identifier (case-insensitive).
            provider_class: A subclass of BaseVectorStore.

        Raises:
            TypeError: If provider_class is not a BaseVectorStore subclass.
        """
        if not (isinstance(provider_class, type) and issubclass(provider_class, BaseVectorStore)):
            raise TypeError(
                f"{provider_class} must be a subclass of BaseVectorStore"
            )
        self._providers[name.lower()] = provider_class

    def create(self, provider: str, **kwargs: Any) -> BaseVectorStore:
        """Create a VectorStore instance by provider name.

        Args:
            provider: Provider identifier (case-insensitive).
            **kwargs: Passed to the provider constructor.

        Returns:
            An instance of the requested VectorStore provider.

        Raises:
            ValueError: If the provider is not registered.
        """
        key = provider.lower()
        cls = self._providers.get(key)
        if cls is None:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise ValueError(
                f"Unknown VectorStore provider '{provider}'. "
                f"Available: {available}"
            )
        return cls(**kwargs)

    def create_from_settings(self, settings: VectorStoreSettings) -> BaseVectorStore:
        """Create a VectorStore instance from a VectorStoreSettings object.

        Extracts ``provider`` for routing, and forwards remaining
        fields as keyword arguments to the provider constructor.

        Args:
            settings: Parsed VectorStore configuration.

        Returns:
            An instance of the configured VectorStore provider.
        """
        fields = asdict(settings)
        provider = fields.pop("provider")
        kwargs = {k: v for k, v in fields.items() if v is not None}
        return self.create(provider, **kwargs)

    def list_providers(self) -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._providers)
