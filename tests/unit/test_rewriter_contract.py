"""Contract tests for BaseQueryRewriter implementations.

Every rewriter must satisfy the same behavioral contract.
Uses @pytest.mark.parametrize to run identical assertions on all providers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.libs.query_rewriter.base_rewriter import BaseQueryRewriter, RewriteResult
from src.libs.query_rewriter.none_rewriter import NoneRewriter
from src.libs.query_rewriter.llm_rewriter import LLMRewriter
from src.libs.query_rewriter.hyde_rewriter import HyDERewriter


def _make_none_rewriter() -> BaseQueryRewriter:
    return NoneRewriter()


def _make_llm_rewriter() -> BaseQueryRewriter:
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(
        content="Rewritten: test query about Python"
    )
    return LLMRewriter(llm=mock_llm, max_rewrites=3)


def _make_hyde_rewriter() -> BaseQueryRewriter:
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(
        content="Hypothetical document about Python."
    )
    return HyDERewriter(llm=mock_llm)


ALL_PROVIDERS = [
    pytest.param(_make_none_rewriter, id="NoneRewriter"),
    pytest.param(_make_llm_rewriter, id="LLMRewriter"),
    pytest.param(_make_hyde_rewriter, id="HyDERewriter"),
]


@pytest.mark.parametrize("factory", ALL_PROVIDERS)
class TestRewriterContract:
    """Contract: all BaseQueryRewriter implementations must satisfy these."""

    def test_rewrite_returns_rewrite_result(self, factory):
        rewriter = factory()
        result = rewriter.rewrite("test query about Python")
        assert isinstance(result, RewriteResult)

    def test_rewrite_result_has_original_query(self, factory):
        rewriter = factory()
        result = rewriter.rewrite("test query about Python")
        assert result.original_query == "test query about Python"

    def test_rewrite_result_has_non_empty_rewritten_queries(self, factory):
        rewriter = factory()
        result = rewriter.rewrite("test query about Python")
        assert isinstance(result.rewritten_queries, list)
        assert len(result.rewritten_queries) >= 1

    def test_rewrite_result_has_strategy(self, factory):
        rewriter = factory()
        result = rewriter.rewrite("test query about Python")
        assert isinstance(result.strategy, str)
        assert len(result.strategy) > 0

    def test_isinstance_base_rewriter(self, factory):
        rewriter = factory()
        assert isinstance(rewriter, BaseQueryRewriter)

    def test_rewrite_result_is_frozen(self, factory):
        rewriter = factory()
        result = rewriter.rewrite("test query")
        with pytest.raises(AttributeError):
            result.original_query = "mutated"
