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
            raise TypeError(f"{provider_class} must be a subclass of BaseLLM")
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
            raise ValueError(f"Unknown LLM provider '{provider}'. Available: {available}")
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

    def create_with_failover(
        self,
        settings: LLMSettings,
    ) -> BaseLLM:
        """Create an LLM with optional circuit breaker and failover chain.

        When ``fallback_providers`` is configured, returns a ProviderChain
        that tries providers in order. Otherwise returns a single BaseLLM
        (optionally wrapped with a circuit breaker via the ``protect``
        decorator — but that's left to the caller).

        Args:
            settings: Parsed LLM configuration with optional
                circuit_breaker and fallback_providers.

        Returns:
            A single BaseLLM or a ProviderChain for multi-provider failover.
        """
        primary = self.create_from_settings(settings)

        # No fallbacks → return single LLM
        if not settings.fallback_providers:
            return primary

        # Lazy imports to avoid circular dependency (llm/__init__ → llm_factory → provider_chain → base_llm)
        from src.libs.circuit_breaker.circuit_breaker import CircuitBreaker
        from src.libs.circuit_breaker.provider_chain import ProviderChain

        # Build circuit breaker config
        cb_cfg = settings.circuit_breaker
        if cb_cfg and cb_cfg.enabled:
            make_cb = lambda: CircuitBreaker(
                failure_threshold=cb_cfg.failure_threshold,
                cooldown=cb_cfg.cooldown_seconds,
            )
        else:
            # Default lenient circuit breaker for chain orchestration
            make_cb = lambda: CircuitBreaker(failure_threshold=5, cooldown=60.0)

        providers: list[tuple[BaseLLM, CircuitBreaker]] = [
            (primary, make_cb()),
        ]

        for fb in settings.fallback_providers:
            fb_kwargs: dict[str, Any] = {"model": fb.model}
            if fb.api_key:
                fb_kwargs["api_key"] = fb.api_key
            if fb.azure_endpoint:
                fb_kwargs["azure_endpoint"] = fb.azure_endpoint
            if fb.base_url:
                fb_kwargs["base_url"] = fb.base_url
            fb_llm = self.create(fb.provider, **fb_kwargs)
            providers.append((fb_llm, make_cb()))

        return ProviderChain(providers)

    @classmethod
    def create_llm(cls, settings: Any) -> BaseLLM:
        """Convenience class method: create an LLM from full Settings.

        Instantiates a factory, registers all known text LLM providers,
        and routes to the configured provider via ``settings.llm``.

        Args:
            settings: Root Settings object (must have ``.llm`` attribute).

        Returns:
            A configured BaseLLM instance.
        """
        from src.libs.llm.azure_llm import AzureLLM
        from src.libs.llm.deepseek_llm import DeepSeekLLM
        from src.libs.llm.ollama_llm import OllamaLLM
        from src.libs.llm.openai_llm import OpenAILLM

        factory = cls()
        factory.register_provider("openai", OpenAILLM)
        factory.register_provider("azure", AzureLLM)
        factory.register_provider("deepseek", DeepSeekLLM)
        factory.register_provider("ollama", OllamaLLM)
        return factory.create_from_settings(settings.llm)

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
            raise TypeError(f"{provider_class} must be a subclass of BaseVisionLLM")
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
            raise ValueError(f"Unknown Vision LLM provider '{provider}'. Available: {available}")
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
