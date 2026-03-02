"""Evaluator Factory -- configuration-driven provider routing.

Uses a registry pattern so new providers can be added without
modifying this module.  Supports two usage modes:

1. **Instance-based** (legacy): register + create by provider name.
2. **Class-based** (config-driven): create from full Settings object
   with lazy-loaded providers for optional dependencies (ragas, composite).

Special handling: when ``enabled=False`` or ``provider="none"``/``"disabled"``,
returns a :class:`NoneEvaluator` without requiring registry lookup.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable

from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator

if TYPE_CHECKING:
    from src.core.settings import EvaluationSettings, Settings


def _get_ragas_evaluator() -> type[BaseEvaluator]:
    """Lazy import RagasEvaluator to avoid hard dependency on ragas."""
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator

    return RagasEvaluator


def _get_composite_evaluator() -> type[BaseEvaluator]:
    """Lazy import CompositeEvaluator."""
    from src.observability.evaluation.composite_evaluator import CompositeEvaluator

    return CompositeEvaluator


class EvaluatorFactory:
    """Factory for creating Evaluator instances by provider name.

    Supports both instance-based and class-based usage:

    Instance-based::

        factory = EvaluatorFactory()
        factory.register_provider("custom", CustomEvaluator)
        evaluator = factory.create_by_name("custom", metrics=["hit_rate"])

    Class-based (config-driven)::

        evaluator = EvaluatorFactory.create(settings)
    """

    # Class-level registries for config-driven creation
    _PROVIDERS: dict[str, type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
    }

    _LAZY_PROVIDERS: dict[str, Callable[[], type[BaseEvaluator]]] = {
        "ragas": _get_ragas_evaluator,
        "composite": _get_composite_evaluator,
    }

    # -- instance-based API (backward compatible) ----------------------

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseEvaluator]] = {}

    def register_provider(
        self,
        name: str,
        provider_class: type[BaseEvaluator],
    ) -> None:
        """Register an Evaluator provider class (instance-level).

        Args:
            name: Provider identifier (case-insensitive).
            provider_class: A subclass of BaseEvaluator.

        Raises:
            TypeError: If provider_class is not a BaseEvaluator subclass.
        """
        if not (isinstance(provider_class, type) and issubclass(provider_class, BaseEvaluator)):
            raise TypeError(
                f"{provider_class} must be a subclass of BaseEvaluator"
            )
        self._providers[name.lower()] = provider_class

    def create_by_name(self, provider: str, **kwargs: Any) -> BaseEvaluator:
        """Create an Evaluator instance by provider name (instance-level).

        Args:
            provider: Provider identifier (case-insensitive).
            **kwargs: Passed to the provider constructor.

        Returns:
            An instance of the requested Evaluator provider.

        Raises:
            ValueError: If the provider is not registered.
        """
        key = provider.lower()
        cls = self._providers.get(key)
        if cls is None:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise ValueError(
                f"Unknown Evaluator provider '{provider}'. "
                f"Available: {available}"
            )
        return cls(**kwargs)

    def create_from_settings(
        self,
        settings: EvaluationSettings,
    ) -> BaseEvaluator:
        """Create an Evaluator from an EvaluationSettings object (instance-level).

        When ``enabled`` is False or ``provider`` is ``"none"``,
        returns a :class:`NoneEvaluator` directly.

        Args:
            settings: Parsed Evaluation configuration.

        Returns:
            An instance of the configured Evaluator provider.
        """
        fields = asdict(settings)
        provider = fields.pop("provider")
        enabled = fields.pop("enabled")

        if not enabled or provider.lower() == "none":
            return NoneEvaluator(**{k: v for k, v in fields.items() if v is not None})

        kwargs = {k: v for k, v in fields.items() if v is not None}
        return self.create_by_name(provider, **kwargs)

    # -- class-based API (config-driven with lazy loading) -------------

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEvaluator:
        """Create an Evaluator instance from full Settings with lazy loading.

        Supports lazy-loaded providers (ragas, composite) that are only
        imported when first requested, avoiding hard dependencies.

        Args:
            settings: The application settings containing evaluation config.
            **override_kwargs: Optional parameters to override config values.

        Returns:
            An instance of the configured Evaluator provider.

        Raises:
            ValueError: If the configured provider is not supported.
            RuntimeError: If provider initialization fails.
        """
        try:
            evaluation_settings = settings.evaluation
            if evaluation_settings is None:
                raise AttributeError("settings.evaluation is None")
            provider_name = evaluation_settings.provider.lower()
            enabled = bool(evaluation_settings.enabled)
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.evaluation.provider. "
                "Please ensure 'evaluation.provider' is specified in settings.yaml"
            ) from e

        if not enabled or provider_name in {"none", "disabled"}:
            return NoneEvaluator(settings=settings, **override_kwargs)

        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None and provider_name in cls._LAZY_PROVIDERS:
            try:
                provider_class = cls._LAZY_PROVIDERS[provider_name]()
                cls._PROVIDERS[provider_name] = provider_class
            except ImportError as e:
                raise ValueError(
                    f"Provider '{provider_name}' requires additional dependencies: {e}"
                ) from e

        if provider_class is None:
            all_providers = sorted(set(cls._PROVIDERS) | set(cls._LAZY_PROVIDERS))
            available = ", ".join(all_providers) if all_providers else "none"
            raise ValueError(
                f"Unsupported Evaluator provider: '{provider_name}'. "
                f"Available providers: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Evaluator provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return sorted list of all known provider names (eager + lazy)."""
        return sorted(set(cls._PROVIDERS) | set(cls._LAZY_PROVIDERS))
