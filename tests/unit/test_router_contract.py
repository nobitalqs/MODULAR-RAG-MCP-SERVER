"""Contract tests for BaseQueryRouter implementations.

Every router must satisfy the same behavioral contract.
Uses @pytest.mark.parametrize to run identical assertions on all providers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.settings import RouteConfig
from src.libs.query_router.base_router import BaseQueryRouter, RouteDecision
from src.libs.query_router.none_router import NoneRouter
from src.libs.query_router.llm_router import LLMRouter


_SAMPLE_ROUTES = [
    RouteConfig(name="knowledge_search", description="Search the knowledge base"),
    RouteConfig(name="direct_answer", description="Answer directly from LLM"),
]


def _make_none_router() -> BaseQueryRouter:
    return NoneRouter()


def _make_llm_router() -> BaseQueryRouter:
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(
        content='{"route": "knowledge_search", "confidence": 0.9, "tool_name": null, "reasoning": "Needs knowledge"}'
    )
    return LLMRouter(llm=mock_llm, routes=_SAMPLE_ROUTES)


ALL_PROVIDERS = [
    pytest.param(_make_none_router, id="NoneRouter"),
    pytest.param(_make_llm_router, id="LLMRouter"),
]


@pytest.mark.parametrize("factory", ALL_PROVIDERS)
class TestRouterContract:
    """Contract: all BaseQueryRouter implementations must satisfy these."""

    def test_route_returns_route_decision(self, factory):
        router = factory()
        result = router.route("What is Python?")
        assert isinstance(result, RouteDecision)

    def test_route_decision_has_route_name(self, factory):
        router = factory()
        result = router.route("What is Python?")
        assert isinstance(result.route, str)
        assert len(result.route) > 0

    def test_route_decision_has_confidence(self, factory):
        router = factory()
        result = router.route("What is Python?")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_route_decision_tool_name_is_optional(self, factory):
        router = factory()
        result = router.route("What is Python?")
        assert result.tool_name is None or isinstance(result.tool_name, str)

    def test_route_decision_is_frozen(self, factory):
        router = factory()
        result = router.route("test")
        with pytest.raises(AttributeError):
            result.route = "mutated"

    def test_isinstance_base_router(self, factory):
        router = factory()
        assert isinstance(router, BaseQueryRouter)
