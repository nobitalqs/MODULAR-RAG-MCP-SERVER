"""Splitter Factory — configuration-driven strategy routing.

Uses a registry pattern so new splitter strategies can be added without
modifying this module. Strategies register via
:meth:`SplitterFactory.register_provider` and are instantiated via
:meth:`SplitterFactory.create`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.core.settings import IngestionSettings
from src.libs.splitter.base_splitter import BaseSplitter


class SplitterFactory:
    """Factory for creating Splitter instances by strategy name.

    Example::

        factory = SplitterFactory()
        factory.register_provider("recursive", RecursiveSplitter)
        sp = factory.create("recursive", chunk_size=1000)
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseSplitter]] = {}

    def register_provider(
        self,
        name: str,
        provider_class: type[BaseSplitter],
    ) -> None:
        """Register a Splitter strategy class.

        Args:
            name: Strategy identifier (case-insensitive).
            provider_class: A subclass of BaseSplitter.

        Raises:
            TypeError: If provider_class is not a BaseSplitter subclass.
        """
        if not (isinstance(provider_class, type) and issubclass(provider_class, BaseSplitter)):
            raise TypeError(
                f"{provider_class} must be a subclass of BaseSplitter"
            )
        self._providers[name.lower()] = provider_class

    def create(self, provider: str, **kwargs: Any) -> BaseSplitter:
        """Create a Splitter instance by strategy name.

        Args:
            provider: Strategy identifier (case-insensitive).
            **kwargs: Passed to the strategy constructor.

        Returns:
            An instance of the requested Splitter strategy.

        Raises:
            ValueError: If the strategy is not registered.
        """
        key = provider.lower()
        cls = self._providers.get(key)
        if cls is None:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise ValueError(
                f"Unknown Splitter provider '{provider}'. "
                f"Available: {available}"
            )
        return cls(**kwargs)

    def create_from_settings(self, settings: IngestionSettings) -> BaseSplitter:
        """Create a Splitter instance from an IngestionSettings object.

        Extracts ``splitter`` for routing, and forwards remaining
        fields as keyword arguments to the strategy constructor.

        Args:
            settings: Parsed Ingestion configuration.

        Returns:
            An instance of the configured Splitter strategy.
        """
        fields = asdict(settings)
        provider = fields.pop("splitter")
        kwargs = {k: v for k, v in fields.items() if v is not None}
        return self.create(provider, **kwargs)

    def list_providers(self) -> list[str]:
        """Return sorted list of registered strategy names."""
        return sorted(self._providers)
