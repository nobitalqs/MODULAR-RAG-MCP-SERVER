"""Query Router Factory — configuration-driven provider routing."""

from __future__ import annotations

from src.core.settings import QueryRoutingSettings
from src.libs.llm.base_llm import BaseLLM
from src.libs.query_router.base_router import BaseQueryRouter
from src.libs.query_router.none_router import NoneRouter


class QueryRouterFactory:
    """Factory for creating query router instances from settings."""

    @staticmethod
    def create_from_settings(
        settings: QueryRoutingSettings,
        llm: BaseLLM | None = None,
    ) -> BaseQueryRouter:
        if not settings.enabled or settings.provider == "none":
            return NoneRouter()

        provider = settings.provider.lower()

        if provider == "llm":
            if llm is None:
                raise ValueError(
                    "An llm instance is required for the 'llm' query router provider"
                )
            from src.libs.query_router.llm_router import LLMRouter

            return LLMRouter(llm=llm, routes=list(settings.routes))

        else:
            raise ValueError(
                f"Unknown query router provider '{settings.provider}'. "
                f"Available: none, llm"
            )

    @staticmethod
    def create_default() -> BaseQueryRouter:
        """Create a default no-op router (for when no config provided)."""
        return NoneRouter()
