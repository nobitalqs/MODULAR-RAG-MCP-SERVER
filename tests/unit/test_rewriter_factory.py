"""Unit tests for QueryRewriterFactory."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import QueryRewritingSettings
from src.libs.query_rewriter.base_rewriter import BaseQueryRewriter
from src.libs.query_rewriter.none_rewriter import NoneRewriter
from src.libs.query_rewriter.rewriter_factory import QueryRewriterFactory


class TestQueryRewriterFactory:
    def test_disabled_returns_none_rewriter(self):
        settings = QueryRewritingSettings(
            enabled=False, provider="llm", max_rewrites=3,
        )
        rewriter = QueryRewriterFactory.create_from_settings(settings)
        assert isinstance(rewriter, NoneRewriter)

    def test_provider_none_returns_none_rewriter(self):
        settings = QueryRewritingSettings(
            enabled=True, provider="none", max_rewrites=3,
        )
        rewriter = QueryRewriterFactory.create_from_settings(settings)
        assert isinstance(rewriter, NoneRewriter)

    def test_provider_llm(self):
        settings = QueryRewritingSettings(
            enabled=True, provider="llm", max_rewrites=5,
        )
        mock_llm = MagicMock()
        rewriter = QueryRewriterFactory.create_from_settings(settings, llm=mock_llm)
        from src.libs.query_rewriter.llm_rewriter import LLMRewriter

        assert isinstance(rewriter, LLMRewriter)
        assert isinstance(rewriter, BaseQueryRewriter)

    def test_provider_hyde(self):
        settings = QueryRewritingSettings(
            enabled=True, provider="hyde", max_rewrites=1,
        )
        mock_llm = MagicMock()
        rewriter = QueryRewriterFactory.create_from_settings(settings, llm=mock_llm)
        from src.libs.query_rewriter.hyde_rewriter import HyDERewriter

        assert isinstance(rewriter, HyDERewriter)

    def test_llm_provider_without_llm_raises(self):
        settings = QueryRewritingSettings(
            enabled=True, provider="llm", max_rewrites=3,
        )
        with pytest.raises(ValueError, match="llm"):
            QueryRewriterFactory.create_from_settings(settings, llm=None)

    def test_hyde_provider_without_llm_raises(self):
        settings = QueryRewritingSettings(
            enabled=True, provider="hyde", max_rewrites=1,
        )
        with pytest.raises(ValueError, match="llm"):
            QueryRewriterFactory.create_from_settings(settings, llm=None)

    def test_unknown_provider_raises(self):
        settings = QueryRewritingSettings(
            enabled=True, provider="unknown", max_rewrites=3,
        )
        with pytest.raises(ValueError, match="Unknown query rewriter provider"):
            QueryRewriterFactory.create_from_settings(settings)

    def test_create_default_returns_none_rewriter(self):
        rewriter = QueryRewriterFactory.create_default()
        assert isinstance(rewriter, NoneRewriter)
