"""Unit tests for HyDERewriter — hypothetical document embedding strategy."""

import json
from unittest.mock import MagicMock

import pytest

from src.libs.llm.base_llm import ChatResponse
from src.libs.query_rewriter.base_rewriter import BaseQueryRewriter, RewriteResult
from src.libs.query_rewriter.hyde_rewriter import HyDERewriter


@pytest.fixture
def mock_llm():
    return MagicMock()


class TestHyDERewriterInterface:
    def test_is_subclass(self):
        assert issubclass(HyDERewriter, BaseQueryRewriter)


class TestHyDERewriterBasic:
    def test_generates_hypothetical_document(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content="Retrieval-Augmented Generation (RAG) is a technique that "
            "combines document retrieval with language model generation...",
            model="test-model",
        )
        rewriter = HyDERewriter(llm=mock_llm)
        result = rewriter.rewrite("What is RAG?")

        assert isinstance(result, RewriteResult)
        assert result.original_query == "What is RAG?"
        assert len(result.rewritten_queries) == 1
        assert "Retrieval-Augmented Generation" in result.rewritten_queries[0]
        assert result.strategy == "hyde"
        mock_llm.chat.assert_called_once()

    def test_strategy_is_hyde(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content="Some hypothetical answer.",
            model="test-model",
        )
        rewriter = HyDERewriter(llm=mock_llm)
        result = rewriter.rewrite("test query")
        assert result.strategy == "hyde"

    def test_reasoning_included(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content="Hypothetical answer content.",
            model="test-model",
        )
        rewriter = HyDERewriter(llm=mock_llm)
        result = rewriter.rewrite("test query")
        assert result.reasoning is not None
        assert "hypothetical" in result.reasoning.lower()


class TestHyDERewriterPrompt:
    def test_prompt_asks_for_answer(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content="Answer content.",
            model="test-model",
        )
        rewriter = HyDERewriter(llm=mock_llm)
        rewriter.rewrite("How does attention work?")

        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        # System prompt should instruct to write a hypothetical answer
        system_text = messages[0].content
        assert "hypothetical" in system_text.lower() or "answer" in system_text.lower()


class TestHyDERewriterErrorHandling:
    def test_llm_failure_falls_back_to_original(self, mock_llm):
        mock_llm.chat.side_effect = RuntimeError("LLM unavailable")
        rewriter = HyDERewriter(llm=mock_llm)
        result = rewriter.rewrite("some query")
        assert result.rewritten_queries == ("some query",)
        assert result.strategy == "hyde"

    def test_empty_response_falls_back_to_original(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content="",
            model="test-model",
        )
        rewriter = HyDERewriter(llm=mock_llm)
        result = rewriter.rewrite("some query")
        assert result.rewritten_queries == ("some query",)

    def test_whitespace_only_response_falls_back(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content="   \n\t  ",
            model="test-model",
        )
        rewriter = HyDERewriter(llm=mock_llm)
        result = rewriter.rewrite("some query")
        assert result.rewritten_queries == ("some query",)
