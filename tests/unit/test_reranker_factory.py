"""Tests for Reranker abstraction: BaseReranker, NoneReranker, RerankerFactory."""

from __future__ import annotations

from typing import Any

import pytest

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker


# ---------------------------------------------------------------------------
# Fake provider for testing
# ---------------------------------------------------------------------------

class FakeReranker(BaseReranker):
    """Minimal Reranker stub that reverses candidate order."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs
        self.call_count = 0

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)
        self.call_count += 1
        return list(reversed(candidates))


# ===========================================================================
# BaseReranker.validate_query
# ===========================================================================

class TestValidateQuery:
    """Tests for BaseReranker.validate_query."""

    def setup_method(self) -> None:
        self.reranker = NoneReranker()

    def test_valid_query(self) -> None:
        self.reranker.validate_query("hello world")  # should not raise

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            self.reranker.validate_query(123)  # type: ignore[arg-type]

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            self.reranker.validate_query("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            self.reranker.validate_query("   ")


# ===========================================================================
# BaseReranker.validate_candidates
# ===========================================================================

class TestValidateCandidates:
    """Tests for BaseReranker.validate_candidates."""

    def setup_method(self) -> None:
        self.reranker = NoneReranker()

    def test_valid_candidates(self) -> None:
        self.reranker.validate_candidates([{"id": "1"}, {"id": "2"}])

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self.reranker.validate_candidates([])

    def test_non_list_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            self.reranker.validate_candidates("not a list")  # type: ignore[arg-type]

    def test_non_dict_item_raises(self) -> None:
        with pytest.raises(ValueError, match="not a dict"):
            self.reranker.validate_candidates([{"id": "1"}, "bad"])  # type: ignore[list-item]


# ===========================================================================
# NoneReranker
# ===========================================================================

class TestNoneReranker:
    """Tests for NoneReranker (pass-through, preserves order)."""

    def setup_method(self) -> None:
        self.reranker = NoneReranker()

    def test_preserves_order(self) -> None:
        candidates = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.7}]
        result = self.reranker.rerank("query", candidates)
        assert result == candidates

    def test_returns_shallow_copy(self) -> None:
        candidates = [{"id": "a"}, {"id": "b"}]
        result = self.reranker.rerank("query", candidates)
        assert result is not candidates
        assert result == candidates

    def test_validates_query(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            self.reranker.rerank(123, [{"id": "a"}])  # type: ignore[arg-type]

    def test_validates_candidates(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self.reranker.rerank("query", [])

    def test_accepts_kwargs(self) -> None:
        candidates = [{"id": "a"}]
        result = self.reranker.rerank("query", candidates, top_k=5)
        assert result == candidates

    def test_accepts_settings_kwarg(self) -> None:
        reranker = NoneReranker(model="none", top_k=10)
        assert reranker.config == {"model": "none", "top_k": 10}


# ===========================================================================
# FakeReranker behavior
# ===========================================================================

class TestFakeReranker:
    """Tests for FakeReranker as a BaseReranker implementation."""

    def test_reverses_order(self) -> None:
        reranker = FakeReranker()
        candidates = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        result = reranker.rerank("query", candidates)
        assert result == [{"id": "c"}, {"id": "b"}, {"id": "a"}]

    def test_validates_input(self) -> None:
        reranker = FakeReranker()
        with pytest.raises(ValueError, match="empty or whitespace"):
            reranker.rerank("", [{"id": "a"}])

    def test_increments_call_count(self) -> None:
        reranker = FakeReranker()
        assert reranker.call_count == 0
        reranker.rerank("q1", [{"id": "a"}])
        assert reranker.call_count == 1
        reranker.rerank("q2", [{"id": "b"}])
        assert reranker.call_count == 2


# ===========================================================================
# RerankerFactory
# ===========================================================================

class TestRerankerFactory:
    """Tests for the Reranker factory routing logic."""

    def setup_method(self) -> None:
        from src.libs.reranker.reranker_factory import RerankerFactory

        self.factory = RerankerFactory()

    def test_register_and_create(self) -> None:
        self.factory.register_provider("fake", FakeReranker)
        reranker = self.factory.create("fake")
        assert isinstance(reranker, FakeReranker)

    def test_case_insensitive_registration(self) -> None:
        self.factory.register_provider("FaKe", FakeReranker)
        reranker = self.factory.create("fake")
        assert isinstance(reranker, FakeReranker)

    def test_case_insensitive_create(self) -> None:
        self.factory.register_provider("fake", FakeReranker)
        reranker = self.factory.create("FAKE")
        assert isinstance(reranker, FakeReranker)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Reranker provider"):
            self.factory.create("nonexistent")

    def test_error_lists_available_providers(self) -> None:
        self.factory.register_provider("fake", FakeReranker)
        with pytest.raises(ValueError, match="fake"):
            self.factory.create("missing")

    def test_register_non_subclass_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a subclass"):
            self.factory.register_provider("bad", dict)  # type: ignore[arg-type]

    def test_list_providers(self) -> None:
        self.factory.register_provider("beta", FakeReranker)
        self.factory.register_provider("alpha", FakeReranker)
        providers = self.factory.list_providers()
        assert providers == ["alpha", "beta"]

    def test_create_with_kwargs(self) -> None:
        self.factory.register_provider("fake", FakeReranker)
        reranker = self.factory.create("fake", top_k=5)
        assert isinstance(reranker, FakeReranker)
        assert reranker.config["top_k"] == 5

    def test_create_from_settings(self) -> None:
        from src.core.settings import RerankSettings

        self.factory.register_provider("fake", FakeReranker)

        settings = RerankSettings(
            enabled=True,
            provider="fake",
            model="test-model",
            top_k=10,
        )
        reranker = self.factory.create_from_settings(settings)
        assert isinstance(reranker, FakeReranker)

    def test_create_from_settings_forwards_fields(self) -> None:
        """create_from_settings should forward non-None fields as kwargs."""
        from src.core.settings import RerankSettings

        self.factory.register_provider("fake", FakeReranker)

        settings = RerankSettings(
            enabled=True,
            provider="fake",
            model="cross-encoder-model",
            top_k=5,
        )
        reranker = self.factory.create_from_settings(settings)
        assert isinstance(reranker, FakeReranker)
        assert reranker.config["model"] == "cross-encoder-model"
        assert reranker.config["top_k"] == 5

    def test_create_from_settings_disabled_returns_none_reranker(self) -> None:
        """When enabled=False, always return NoneReranker regardless of provider."""
        from src.core.settings import RerankSettings

        settings = RerankSettings(
            enabled=False,
            provider="cross_encoder",
            model="some-model",
            top_k=10,
        )
        reranker = self.factory.create_from_settings(settings)
        assert isinstance(reranker, NoneReranker)

    def test_create_from_settings_none_provider_returns_none_reranker(self) -> None:
        """When provider='none', return NoneReranker even if enabled=True."""
        from src.core.settings import RerankSettings

        settings = RerankSettings(
            enabled=True,
            provider="none",
            model="none",
            top_k=10,
        )
        reranker = self.factory.create_from_settings(settings)
        assert isinstance(reranker, NoneReranker)

    def test_empty_registry_lists_none(self) -> None:
        with pytest.raises(ValueError, match="\\(none\\)"):
            self.factory.create("anything")

    def test_duplicate_registration_overwrites(self) -> None:
        """Registering same name twice overwrites the previous provider."""
        self.factory.register_provider("fake", FakeReranker)
        self.factory.register_provider("fake", NoneReranker)
        reranker = self.factory.create("fake")
        assert isinstance(reranker, NoneReranker)


# ===========================================================================
# Boundary tests — reranker edge cases
# ===========================================================================


class TestRerankerBoundary:
    """Boundary tests for reranker edge cases."""

    def test_single_candidate_preserved(self) -> None:
        """Reranking a single candidate returns it unchanged."""
        reranker = NoneReranker()
        result = reranker.rerank("query", [{"id": "only"}])
        assert len(result) == 1
        assert result[0]["id"] == "only"

    def test_candidates_with_varying_fields(self) -> None:
        """Candidates can have different metadata shapes."""
        reranker = NoneReranker()
        candidates = [
            {"id": "a", "score": 0.9, "text": "hello"},
            {"id": "b"},
            {"id": "c", "metadata": {"src": "x.pdf"}},
        ]
        result = reranker.rerank("query", candidates)
        assert len(result) == 3
        assert [r["id"] for r in result] == ["a", "b", "c"]

    def test_fake_reranker_single_candidate(self) -> None:
        """FakeReranker reversing single candidate returns same."""
        reranker = FakeReranker()
        result = reranker.rerank("query", [{"id": "only"}])
        assert result == [{"id": "only"}]

    def test_unicode_query_accepted(self) -> None:
        """Unicode queries are valid."""
        reranker = NoneReranker()
        result = reranker.rerank("什么是 RAG？", [{"id": "a"}])
        assert len(result) == 1

    def test_validate_candidates_non_list_type(self) -> None:
        """Tuple of dicts is not accepted (must be list)."""
        reranker = NoneReranker()
        with pytest.raises(ValueError, match="must be a list"):
            reranker.validate_candidates(({"id": "a"},))  # type: ignore[arg-type]
