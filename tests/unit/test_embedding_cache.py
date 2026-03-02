"""Unit tests for EmbeddingCache — caching decorator for BaseEmbedding."""

from unittest.mock import MagicMock

import pytest

from src.libs.cache.embedding_cache import EmbeddingCache
from src.libs.cache.memory_cache import InMemoryCache
from src.libs.embedding.base_embedding import BaseEmbedding


@pytest.fixture
def mock_embedding():
    emb = MagicMock(spec=BaseEmbedding)
    emb.model = "test-model"
    emb.embed.side_effect = lambda texts, **kw: [[float(i)] * 3 for i in range(len(texts))]
    return emb


@pytest.fixture
def cache():
    return InMemoryCache(max_items=1000, default_ttl=3600)


@pytest.fixture
def cached_embedding(mock_embedding, cache):
    return EmbeddingCache(embedding=mock_embedding, cache=cache, model_name="test-model")


class TestEmbeddingCacheHitMiss:
    def test_all_miss_calls_underlying(self, cached_embedding, mock_embedding):
        result = cached_embedding.embed(["a", "b"])
        assert len(result) == 2
        mock_embedding.embed.assert_called_once_with(["a", "b"])

    def test_all_hit_skips_underlying(self, cached_embedding, mock_embedding):
        cached_embedding.embed(["a", "b"])
        mock_embedding.embed.reset_mock()
        result = cached_embedding.embed(["a", "b"])
        assert len(result) == 2
        mock_embedding.embed.assert_not_called()

    def test_partial_hit(self, cached_embedding, mock_embedding):
        cached_embedding.embed(["a"])  # cache "a"
        mock_embedding.embed.reset_mock()
        mock_embedding.embed.side_effect = lambda texts, **kw: [[9.0] * 3 for _ in texts]
        result = cached_embedding.embed(["a", "b"])
        # Only "b" should be sent to underlying
        mock_embedding.embed.assert_called_once_with(["b"])
        assert len(result) == 2

    def test_preserves_original_order(self, cached_embedding, mock_embedding):
        # Cache "b" first
        mock_embedding.embed.side_effect = lambda texts, **kw: [[99.0] for _ in texts]
        cached_embedding.embed(["b"])
        mock_embedding.embed.reset_mock()
        mock_embedding.embed.side_effect = lambda texts, **kw: [[1.0] for _ in texts]
        result = cached_embedding.embed(["a", "b", "c"])
        # "a" and "c" are misses, "b" is hit
        assert len(result) == 3
        assert result[1] == [99.0]  # "b" from cache

    def test_empty_list_returns_empty(self, cached_embedding, mock_embedding):
        result = cached_embedding.embed([])
        assert result == []
        mock_embedding.embed.assert_not_called()


class TestEmbeddingCacheInterface:
    def test_is_subclass_of_base_embedding(self):
        assert issubclass(EmbeddingCache, BaseEmbedding)

    def test_get_dimension_delegates(self, cached_embedding, mock_embedding):
        mock_embedding.get_dimension.return_value = 1536
        assert cached_embedding.get_dimension() == 1536

    def test_validate_texts_delegates(self, cached_embedding, mock_embedding):
        cached_embedding.validate_texts(["hello"])
        mock_embedding.validate_texts.assert_called_once_with(["hello"])
