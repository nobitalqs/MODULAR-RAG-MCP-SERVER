"""Unit tests for BaseQueryRewriter ABC and NoneRewriter."""

from dataclasses import FrozenInstanceError

import pytest

from src.libs.query_rewriter.base_rewriter import BaseQueryRewriter, RewriteResult
from src.libs.query_rewriter.none_rewriter import NoneRewriter


class TestRewriteResult:
    def test_fields(self):
        r = RewriteResult(
            original_query="hello",
            rewritten_queries=["hello"],
            reasoning=None,
            strategy="none",
        )
        assert r.original_query == "hello"
        assert r.rewritten_queries == ["hello"]
        assert r.reasoning is None
        assert r.strategy == "none"

    def test_frozen(self):
        r = RewriteResult(
            original_query="hello",
            rewritten_queries=["hello"],
            reasoning=None,
            strategy="none",
        )
        with pytest.raises(FrozenInstanceError):
            r.original_query = "changed"


class TestNoneRewriter:
    def test_is_subclass(self):
        assert issubclass(NoneRewriter, BaseQueryRewriter)

    def test_rewrite_returns_original(self):
        rewriter = NoneRewriter()
        result = rewriter.rewrite("What is RAG?")
        assert result.original_query == "What is RAG?"
        assert result.rewritten_queries == ["What is RAG?"]
        assert result.reasoning is None
        assert result.strategy == "none"

    def test_rewrite_with_conversation_history(self):
        rewriter = NoneRewriter()
        result = rewriter.rewrite("What is RAG?", conversation_history=["prior msg"])
        assert result.rewritten_queries == ["What is RAG?"]

    def test_rewrite_preserves_whitespace(self):
        rewriter = NoneRewriter()
        result = rewriter.rewrite("  spaced query  ")
        assert result.rewritten_queries == ["  spaced query  "]

    def test_empty_query(self):
        rewriter = NoneRewriter()
        result = rewriter.rewrite("")
        assert result.rewritten_queries == [""]
