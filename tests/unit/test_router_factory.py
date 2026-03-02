"""Unit tests for QueryRouterFactory."""

from unittest.mock import MagicMock

import pytest

from src.core.settings import QueryRoutingSettings, RouteConfig
from src.libs.query_router.base_router import BaseQueryRouter
from src.libs.query_router.none_router import NoneRouter
from src.libs.query_router.router_factory import QueryRouterFactory


_SAMPLE_ROUTES = (
    RouteConfig(name="knowledge_search", description="Search the knowledge base"),
    RouteConfig(name="direct_answer", description="Answer directly from LLM"),
)


class TestQueryRouterFactory:
    def test_disabled_returns_none_router(self):
        settings = QueryRoutingSettings(
            enabled=False, provider="llm", routes=_SAMPLE_ROUTES,
        )
        router = QueryRouterFactory.create_from_settings(settings)
        assert isinstance(router, NoneRouter)

    def test_provider_none_returns_none_router(self):
        settings = QueryRoutingSettings(
            enabled=True, provider="none", routes=_SAMPLE_ROUTES,
        )
        router = QueryRouterFactory.create_from_settings(settings)
        assert isinstance(router, NoneRouter)

    def test_provider_llm(self):
        settings = QueryRoutingSettings(
            enabled=True, provider="llm", routes=_SAMPLE_ROUTES,
        )
        mock_llm = MagicMock()
        router = QueryRouterFactory.create_from_settings(settings, llm=mock_llm)
        from src.libs.query_router.llm_router import LLMRouter

        assert isinstance(router, LLMRouter)
        assert isinstance(router, BaseQueryRouter)

    def test_llm_provider_without_llm_raises(self):
        settings = QueryRoutingSettings(
            enabled=True, provider="llm", routes=_SAMPLE_ROUTES,
        )
        with pytest.raises(ValueError, match="llm"):
            QueryRouterFactory.create_from_settings(settings, llm=None)

    def test_unknown_provider_raises(self):
        settings = QueryRoutingSettings(
            enabled=True, provider="unknown", routes=_SAMPLE_ROUTES,
        )
        with pytest.raises(ValueError, match="Unknown query router provider"):
            QueryRouterFactory.create_from_settings(settings)

    def test_create_default_returns_none_router(self):
        router = QueryRouterFactory.create_default()
        assert isinstance(router, NoneRouter)
