"""Unit tests for LLMRewriter — mock LLM to test three rewrite modes."""

import json
from unittest.mock import MagicMock

import pytest

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.query_rewriter.base_rewriter import BaseQueryRewriter, RewriteResult
from src.libs.query_rewriter.llm_rewriter import LLMRewriter


@pytest.fixture
def mock_llm():
    return MagicMock()


class TestLLMRewriterInterface:
    def test_is_subclass(self):
        assert issubclass(LLMRewriter, BaseQueryRewriter)


class TestLLMRewriterSingleRewrite:
    def test_simple_query_single_rewrite(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({
                "queries": ["What is retrieval-augmented generation?"],
                "reasoning": "Expanded abbreviation",
            }),
            model="test-model",
        )
        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=3)
        result = rewriter.rewrite("What is RAG?")

        assert isinstance(result, RewriteResult)
        assert result.original_query == "What is RAG?"
        assert result.rewritten_queries == [
            "What is retrieval-augmented generation?",
        ]
        assert result.reasoning == "Expanded abbreviation"
        assert result.strategy == "llm"
        mock_llm.chat.assert_called_once()


class TestLLMRewriterDecomposition:
    def test_complex_query_decomposed(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({
                "queries": [
                    "What is the architecture of transformers?",
                    "How do transformers compare to RNNs?",
                ],
                "reasoning": "Decomposed multi-part question",
            }),
            model="test-model",
        )
        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=5)
        result = rewriter.rewrite(
            "Explain the architecture of transformers and how they compare to RNNs"
        )
        assert len(result.rewritten_queries) == 2
        assert result.strategy == "llm"

    def test_max_rewrites_limits_output(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({
                "queries": ["q1", "q2", "q3", "q4", "q5"],
                "reasoning": "Many queries",
            }),
            model="test-model",
        )
        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=2)
        result = rewriter.rewrite("complex query")
        assert len(result.rewritten_queries) == 2


class TestLLMRewriterWithHistory:
    def test_conversation_history_included_in_prompt(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({
                "queries": ["What is the attention mechanism in transformers?"],
                "reasoning": "Resolved pronoun from context",
            }),
            model="test-model",
        )
        history = [
            Message("user", "Tell me about transformers"),
            Message("assistant", "Transformers are a type of neural network..."),
        ]
        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=3)
        result = rewriter.rewrite("How does it handle attention?", conversation_history=history)

        assert result.rewritten_queries == [
            "What is the attention mechanism in transformers?",
        ]
        # Verify the prompt included history context
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        prompt_text = " ".join(m.content for m in messages)
        assert "transformers" in prompt_text.lower()


class TestLLMRewriterErrorHandling:
    def test_invalid_json_falls_back_to_original(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content="not valid json {{{",
            model="test-model",
        )
        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=3)
        result = rewriter.rewrite("some query")
        assert result.rewritten_queries == ["some query"]
        assert result.strategy == "llm"

    def test_llm_exception_falls_back_to_original(self, mock_llm):
        mock_llm.chat.side_effect = RuntimeError("LLM unavailable")
        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=3)
        result = rewriter.rewrite("some query")
        assert result.rewritten_queries == ["some query"]

    def test_missing_queries_key_falls_back(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({"answer": "wrong format"}),
            model="test-model",
        )
        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=3)
        result = rewriter.rewrite("some query")
        assert result.rewritten_queries == ["some query"]

    def test_empty_queries_list_falls_back(self, mock_llm):
        mock_llm.chat.return_value = ChatResponse(
            content=json.dumps({"queries": [], "reasoning": "none"}),
            model="test-model",
        )
        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=3)
        result = rewriter.rewrite("some query")
        assert result.rewritten_queries == ["some query"]
