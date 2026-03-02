"""Tests for Embedding abstraction: BaseEmbedding, EmbeddingFactory."""

from __future__ import annotations

from typing import Any

import pytest

from src.libs.embedding.base_embedding import BaseEmbedding


# ---------------------------------------------------------------------------
# Fake provider for testing
# ---------------------------------------------------------------------------

class FakeEmbedding(BaseEmbedding):
    """Minimal Embedding stub that returns deterministic vectors."""

    def __init__(self, dimension: int = 4, **kwargs: Any) -> None:
        self.dimension = dimension
        self.config = kwargs
        self.call_count = 0

    def embed(
        self,
        texts: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        self.validate_texts(texts)
        self.call_count += 1
        return [
            [float(i + j) for j in range(self.dimension)]
            for i in range(len(texts))
        ]

    def get_dimension(self) -> int:
        return self.dimension


# ===========================================================================
# BaseEmbedding.validate_texts
# ===========================================================================

class TestBaseEmbeddingValidation:
    """Tests for BaseEmbedding.validate_texts."""

    def setup_method(self) -> None:
        self.emb = FakeEmbedding()

    def test_valid_texts(self) -> None:
        self.emb.validate_texts(["hello", "world"])  # should not raise

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self.emb.validate_texts([])

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not a string"):
            self.emb.validate_texts(["ok", 123])  # type: ignore[list-item]

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            self.emb.validate_texts(["ok", ""])

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            self.emb.validate_texts(["   "])

    def test_single_valid_text(self) -> None:
        self.emb.validate_texts(["single"])  # should not raise


# ===========================================================================
# BaseEmbedding.get_dimension
# ===========================================================================

class TestBaseEmbeddingGetDimension:
    """Tests for BaseEmbedding.get_dimension."""

    def test_dimension_implemented(self) -> None:
        emb = FakeEmbedding(dimension=1536)
        assert emb.get_dimension() == 1536

    def test_dimension_not_implemented(self) -> None:
        class IncompleteEmbedding(BaseEmbedding):
            def embed(
                self,
                texts: list[str],
                trace: Any = None,
                **kwargs: Any,
            ) -> list[list[float]]:
                return [[0.0]]

        incomplete = IncompleteEmbedding()
        with pytest.raises(NotImplementedError, match="must implement get_dimension"):
            incomplete.get_dimension()


# ===========================================================================
# FakeEmbedding behavior
# ===========================================================================

class TestFakeEmbedding:
    """Tests for FakeEmbedding as a BaseEmbedding implementation."""

    def test_embed_single_text(self) -> None:
        emb = FakeEmbedding(dimension=3)
        result = emb.embed(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 3
        assert result[0] == [0.0, 1.0, 2.0]

    def test_embed_multiple_texts(self) -> None:
        emb = FakeEmbedding(dimension=2)
        result = emb.embed(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == [0.0, 1.0]
        assert result[1] == [1.0, 2.0]
        assert result[2] == [2.0, 3.0]

    def test_embed_validates_input(self) -> None:
        emb = FakeEmbedding()
        with pytest.raises(ValueError, match="cannot be empty"):
            emb.embed([])

    def test_embed_increments_call_count(self) -> None:
        emb = FakeEmbedding()
        assert emb.call_count == 0
        emb.embed(["a"])
        assert emb.call_count == 1
        emb.embed(["b", "c"])
        assert emb.call_count == 2


# ===========================================================================
# EmbeddingFactory
# ===========================================================================

class TestEmbeddingFactory:
    """Tests for the Embedding factory routing logic."""

    def setup_method(self) -> None:
        from src.libs.embedding.embedding_factory import EmbeddingFactory

        self.factory = EmbeddingFactory()

    def test_register_and_create(self) -> None:
        self.factory.register_provider("fake", FakeEmbedding)
        emb = self.factory.create("fake")
        assert isinstance(emb, FakeEmbedding)

    def test_case_insensitive_registration(self) -> None:
        self.factory.register_provider("FaKe", FakeEmbedding)
        emb = self.factory.create("fake")
        assert isinstance(emb, FakeEmbedding)

    def test_case_insensitive_create(self) -> None:
        self.factory.register_provider("fake", FakeEmbedding)
        emb = self.factory.create("FAKE")
        assert isinstance(emb, FakeEmbedding)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Embedding provider"):
            self.factory.create("nonexistent")

    def test_error_lists_available_providers(self) -> None:
        self.factory.register_provider("fake", FakeEmbedding)
        with pytest.raises(ValueError, match="fake"):
            self.factory.create("missing")

    def test_register_non_subclass_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a subclass"):
            self.factory.register_provider("bad", dict)  # type: ignore[arg-type]

    def test_list_providers(self) -> None:
        self.factory.register_provider("beta", FakeEmbedding)
        self.factory.register_provider("alpha", FakeEmbedding)
        providers = self.factory.list_providers()
        assert providers == ["alpha", "beta"]

    def test_create_with_kwargs(self) -> None:
        self.factory.register_provider("fake", FakeEmbedding)
        emb = self.factory.create("fake", dimension=1536)
        assert isinstance(emb, FakeEmbedding)
        assert emb.dimension == 1536

    def test_create_from_settings(self) -> None:
        from src.core.settings import EmbeddingSettings

        self.factory.register_provider("fake", FakeEmbedding)

        settings = EmbeddingSettings(
            provider="fake",
            model="test-model",
            dimensions=384,
        )
        emb = self.factory.create_from_settings(settings)
        assert isinstance(emb, FakeEmbedding)

    def test_create_from_settings_forwards_fields(self) -> None:
        """create_from_settings should forward non-None fields as kwargs."""
        from src.core.settings import EmbeddingSettings

        self.factory.register_provider("fake", FakeEmbedding)

        settings = EmbeddingSettings(
            provider="fake",
            model="text-embedding-3-small",
            dimensions=1536,
            api_key="sk-test",
        )
        emb = self.factory.create_from_settings(settings)
        assert isinstance(emb, FakeEmbedding)
        # FakeEmbedding stores extra kwargs in self.config
        assert emb.config["model"] == "text-embedding-3-small"
        assert emb.config["dimensions"] == 1536
        assert emb.config["api_key"] == "sk-test"

    def test_empty_registry_lists_none(self) -> None:
        with pytest.raises(ValueError, match="\\(none\\)"):
            self.factory.create("anything")
