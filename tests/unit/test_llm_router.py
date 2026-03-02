"""Unit tests for LLMRouter — mock LLM to test route classification."""

import json
from unittest.mock import MagicMock

import pytest

from src.core.settings import RouteConfig
from src.libs.llm.base_llm import ChatResponse
from src.libs.query_router.base_router import BaseQueryRouter, RouteDecision
from src.libs.query_router.llm_router import LLMRouter


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def routes():
    return [
        RouteConfig(name="knowledge_search", description="Search the knowledge base"),
        RouteConfig(name="direct_answer", description="Answer directly without retrieval"),
        RouteConfig(name="tool_call", description="Use a tool to compute an answer"),
    ]


class TestLLMRouterInterface:
    def test_is_subclass(self):
        assert issubclass(LLMRouter, BaseQueryRouter)


class TestLLMRouterKnowledgeSearch:
    def test_routes_to_knowledge_search(self, mock_llm, routes):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({
                "route": "knowledge_search",
                "confidence": 0.95,
                "tool_name": None,
                "reasoning": "User asking a factual question",
            }),
            model="test-model",
        )
        router = LLMRouter(llm=mock_llm, routes=routes)
        result = router.route("What is RAG?")

        assert isinstance(result, RouteDecision)
        assert result.route == "knowledge_search"
        assert result.confidence == 0.95
        assert result.tool_name is None
        assert result.reasoning == "User asking a factual question"


class TestLLMRouterDirectAnswer:
    def test_routes_to_direct_answer(self, mock_llm, routes):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({
                "route": "direct_answer",
                "confidence": 0.9,
                "tool_name": None,
                "reasoning": "Simple greeting",
            }),
            model="test-model",
        )
        router = LLMRouter(llm=mock_llm, routes=routes)
        result = router.route("Hello!")
        assert result.route == "direct_answer"
        assert result.confidence == 0.9


class TestLLMRouterToolCall:
    def test_routes_to_tool_call(self, mock_llm, routes):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({
                "route": "tool_call",
                "confidence": 0.85,
                "tool_name": "calculator",
                "reasoning": "Math expression detected",
            }),
            model="test-model",
        )
        router = LLMRouter(llm=mock_llm, routes=routes)
        result = router.route("Calculate 2+2")
        assert result.route == "tool_call"
        assert result.tool_name == "calculator"


class TestLLMRouterErrorHandling:
    def test_invalid_json_falls_back(self, mock_llm, routes):
        mock_llm.chat.return_value = ChatResponse(
            content="not json {{{",
            model="test-model",
        )
        router = LLMRouter(llm=mock_llm, routes=routes)
        result = router.route("some query")
        assert result.route == "knowledge_search"
        assert result.confidence == 0.5

    def test_llm_exception_falls_back(self, mock_llm, routes):
        mock_llm.chat.side_effect = RuntimeError("LLM down")
        router = LLMRouter(llm=mock_llm, routes=routes)
        result = router.route("some query")
        assert result.route == "knowledge_search"
        assert result.confidence == 0.5

    def test_missing_route_key_falls_back(self, mock_llm, routes):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({"answer": "wrong format"}),
            model="test-model",
        )
        router = LLMRouter(llm=mock_llm, routes=routes)
        result = router.route("some query")
        assert result.route == "knowledge_search"

    def test_unknown_route_falls_back(self, mock_llm, routes):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({
                "route": "nonexistent_route",
                "confidence": 0.9,
                "tool_name": None,
                "reasoning": "test",
            }),
            model="test-model",
        )
        router = LLMRouter(llm=mock_llm, routes=routes)
        result = router.route("some query")
        assert result.route == "knowledge_search"
        assert result.confidence == 0.5


class TestLLMRouterPrompt:
    def test_prompt_includes_route_descriptions(self, mock_llm, routes):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({
                "route": "knowledge_search",
                "confidence": 0.9,
                "tool_name": None,
                "reasoning": "test",
            }),
            model="test-model",
        )
        router = LLMRouter(llm=mock_llm, routes=routes)
        router.route("test query")

        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        system_text = messages[0].content
        assert "knowledge_search" in system_text
        assert "direct_answer" in system_text
        assert "tool_call" in system_text
