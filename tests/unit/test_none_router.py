"""Unit tests for BaseQueryRouter ABC, RouteDecision, and NoneRouter."""

from dataclasses import FrozenInstanceError

import pytest

from src.libs.query_router.base_router import BaseQueryRouter, RouteDecision
from src.libs.query_router.none_router import NoneRouter


class TestRouteDecision:
    def test_fields(self):
        d = RouteDecision(
            route="knowledge_search",
            confidence=0.95,
            tool_name=None,
            reasoning="High relevance",
        )
        assert d.route == "knowledge_search"
        assert d.confidence == 0.95
        assert d.tool_name is None
        assert d.reasoning == "High relevance"

    def test_frozen(self):
        d = RouteDecision(
            route="knowledge_search",
            confidence=1.0,
            tool_name=None,
            reasoning=None,
        )
        with pytest.raises(FrozenInstanceError):
            d.route = "direct_answer"

    def test_with_tool_name(self):
        d = RouteDecision(
            route="tool_call",
            confidence=0.8,
            tool_name="calculator",
            reasoning="Math detected",
        )
        assert d.tool_name == "calculator"


class TestNoneRouter:
    def test_is_subclass(self):
        assert issubclass(NoneRouter, BaseQueryRouter)

    def test_route_returns_knowledge_search(self):
        router = NoneRouter()
        result = router.route("What is RAG?")
        assert isinstance(result, RouteDecision)
        assert result.route == "knowledge_search"
        assert result.confidence == 1.0
        assert result.tool_name is None
        assert result.reasoning is None

    def test_route_ignores_any_query(self):
        router = NoneRouter()
        result = router.route("Calculate 2+2")
        assert result.route == "knowledge_search"

    def test_route_with_kwargs(self):
        router = NoneRouter()
        result = router.route("test", conversation_history=["prior"])
        assert result.route == "knowledge_search"
