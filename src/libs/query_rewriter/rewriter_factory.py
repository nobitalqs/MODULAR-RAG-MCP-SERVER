"""Query Rewriter Factory — configuration-driven provider routing."""

from __future__ import annotations

from src.core.settings import QueryRewritingSettings
from src.libs.llm.base_llm import BaseLLM
from src.libs.query_rewriter.base_rewriter import BaseQueryRewriter
from src.libs.query_rewriter.none_rewriter import NoneRewriter


class QueryRewriterFactory:
    """Factory for creating query rewriter instances from settings."""

    @staticmethod
    def create_from_settings(
        settings: QueryRewritingSettings,
        llm: BaseLLM | None = None,
    ) -> BaseQueryRewriter:
        if not settings.enabled or settings.provider == "none":
            return NoneRewriter()

        provider = settings.provider.lower()

        if provider == "llm":
            if llm is None:
                raise ValueError(
                    "An llm instance is required for the 'llm' query rewriter provider"
                )
            from src.libs.query_rewriter.llm_rewriter import LLMRewriter

            return LLMRewriter(llm=llm, max_rewrites=settings.max_rewrites)

        elif provider == "hyde":
            if llm is None:
                raise ValueError(
                    "An llm instance is required for the 'hyde' query rewriter provider"
                )
            from src.libs.query_rewriter.hyde_rewriter import HyDERewriter

            return HyDERewriter(llm=llm)

        else:
            raise ValueError(
                f"Unknown query rewriter provider '{settings.provider}'. "
                f"Available: none, llm, hyde"
            )

    @staticmethod
    def create_default() -> BaseQueryRewriter:
        """Create a default no-op rewriter (for when no config provided)."""
        return NoneRewriter()
