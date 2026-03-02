"""LLM Factory — configuration-driven provider routing.

Uses a registry pattern so new providers can be added without
modifying this module. Providers register via
:meth:`LLMFactory.register_provider` and are instantiated via
:meth:`LLMFactory.create`.

Vision LLM providers use a separate registry accessible via
:meth:`register_vision_provider` and :meth:`create_vision_llm`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.core.settings import LLMSettings
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.base_vision_llm import BaseVisionLLM


class LLMFactory:
    """Factory for creating LLM instances by provider name.

    Maintains two registries: one for text LLMs, one for Vision LLMs.

    Example::

        factory = LLMFactory()
        factory.register_provider("openai", OpenAILLM)
        llm = factory.create("openai", model="gpt-4o")

        factory.register_vision_provider("azure", AzureVisionLLM)
        vlm = factory.create_vision_llm("azure", model="gpt-4o")
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseLLM]] = {}
        self._vision_providers: dict[str, type[BaseVisionLLM]] = {}

    # ── Text LLM Registry ───────────────────────────────────────────

    def register_provider(
        self,
        name: str,
        provider_class: type[BaseLLM],
    ) -> None:
        """Register an LLM provider class.

        Args:
            name: Provider identifier (case-insensitive).
            provider_class: A subclass of BaseLLM.

        Raises:
            TypeError: If provider_class is not a BaseLLM subclass.
        """
        if not (isinstance(provider_class, type) and issubclass(provider_class, BaseLLM)):
            raise TypeError(
                f"{provider_class} must be a subclass of BaseLLM"
            )
        self._providers[name.lower()] = provider_class

    def create(self, provider: str, **kwargs: Any) -> BaseLLM:
        """Create an LLM instance by provider name.

        Args:
            provider: Provider identifier (case-insensitive).
            **kwargs: Passed to the provider constructor.

        Returns:
            An instance of the requested LLM provider.

        Raises:
            ValueError: If the provider is not registered.
        """
        key = provider.lower()
        cls = self._providers.get(key)
        if cls is None:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Available: {available}"
            )
        return cls(**kwargs)

    def create_from_settings(self, settings: LLMSettings) -> BaseLLM:
        """Create an LLM instance from a LLMSettings object.

        Extracts ``provider`` for routing, and forwards remaining
        fields as keyword arguments to the provider constructor.

        Args:
            settings: Parsed LLM configuration.

        Returns:
            An instance of the configured LLM provider.
        """
        fields = asdict(settings)
        provider = fields.pop("provider")
        # Remove None values so providers only see explicitly set config
        kwargs = {k: v for k, v in fields.items() if v is not None}
        return self.create(provider, **kwargs)

    def list_providers(self) -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._providers)

    # ── Vision LLM Registry ─────────────────────────────────────────

    def register_vision_provider(
        self,
        name: str,
        provider_class: type[BaseVisionLLM],
    ) -> None:
        """Register a Vision LLM provider class.

        Args:
            name: Provider identifier (case-insensitive).
            provider_class: A subclass of BaseVisionLLM.

        Raises:
            TypeError: If provider_class is not a BaseVisionLLM subclass.
        """
        if not (isinstance(provider_class, type) and issubclass(provider_class, BaseVisionLLM)):
            raise TypeError(
                f"{provider_class} must be a subclass of BaseVisionLLM"
            )
        self._vision_providers[name.lower()] = provider_class

    def create_vision_llm(self, provider: str, **kwargs: Any) -> BaseVisionLLM:
        """Create a Vision LLM instance by provider name.

        Args:
            provider: Provider identifier (case-insensitive).
            **kwargs: Passed to the provider constructor.

        Returns:
            An instance of the requested Vision LLM provider.

        Raises:
            ValueError: If the provider is not registered.
        """
        key = provider.lower()
        cls = self._vision_providers.get(key)
        if cls is None:
            available = ", ".join(sorted(self._vision_providers)) or "(none)"
            raise ValueError(
                f"Unknown Vision LLM provider '{provider}'. "
                f"Available: {available}"
            )
        return cls(**kwargs)

    def create_vision_llm_from_settings(
        self,
        settings: Any,
    ) -> BaseVisionLLM | None:
        """Create a Vision LLM instance from VisionLLMSettings.

        When ``enabled`` is False, returns ``None`` (no vision capability).
        Otherwise extracts ``provider`` for routing and forwards remaining
        fields as keyword arguments.

        Args:
            settings: Parsed VisionLLM configuration (VisionLLMSettings).

        Returns:
            A Vision LLM instance, or None if disabled.
        """
        fields = asdict(settings)
        provider = fields.pop("provider")
        enabled = fields.pop("enabled")

        if not enabled:
            return None

        kwargs = {k: v for k, v in fields.items() if v is not None}
        return self.create_vision_llm(provider, **kwargs)

    def list_vision_providers(self) -> list[str]:
        """Return sorted list of registered vision provider names."""
        return sorted(self._vision_providers)
