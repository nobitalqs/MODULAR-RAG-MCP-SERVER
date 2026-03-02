"""Integration test: EmbeddingCache + InMemoryCache.

Verifies that embedding results are cached and served from cache on re-request,
reducing calls to the underlying embedding provider.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.libs.cache.memory_cache import InMemoryCache
from src.libs.cache.embedding_cache import EmbeddingCache


class TestEmbeddingCacheWithInMemoryCache:
    """EmbeddingCache wraps a real InMemoryCache — end-to-end caching."""

    def _make_mock_embedding(self, dimension: int = 3):
        mock = MagicMock()
        call_count = 0

        def side_effect(texts, **kwargs):
            nonlocal call_count
            call_count += 1
            return [[float(call_count)] * dimension for _ in texts]

        mock.embed.side_effect = side_effect
        mock.get_dimension.return_value = dimension
        mock.validate_texts.return_value = None
        mock._call_count = lambda: call_count
        return mock

    def test_cache_hit_avoids_provider_call(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        mock_emb = self._make_mock_embedding()
        cached_emb = EmbeddingCache(
            embedding=mock_emb, cache=cache, model_name="test-model",
        )

        # First call — cache miss, hits provider
        result1 = cached_emb.embed(["hello world"])
        assert len(result1) == 1
        assert mock_emb.embed.call_count == 1

        # Second call — cache hit, does NOT hit provider
        result2 = cached_emb.embed(["hello world"])
        assert result2 == result1
        assert mock_emb.embed.call_count == 1  # unchanged

    def test_partial_cache_hit(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        mock_emb = self._make_mock_embedding()
        cached_emb = EmbeddingCache(
            embedding=mock_emb, cache=cache, model_name="test-model",
        )

        # Pre-populate one text
        cached_emb.embed(["text-A"])
        assert mock_emb.embed.call_count == 1

        # Now request two texts — only text-B should call provider
        result = cached_emb.embed(["text-A", "text-B"])
        assert len(result) == 2
        assert mock_emb.embed.call_count == 2
        # text-B was passed as a single-item batch to the provider
        args = mock_emb.embed.call_args_list[-1]
        assert args[0][0] == ["text-B"]

    def test_empty_input_returns_empty(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        mock_emb = self._make_mock_embedding()
        cached_emb = EmbeddingCache(
            embedding=mock_emb, cache=cache, model_name="test-model",
        )

        result = cached_emb.embed([])
        assert result == []
        assert mock_emb.embed.call_count == 0

    def test_get_dimension_delegates(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        mock_emb = self._make_mock_embedding(dimension=768)
        cached_emb = EmbeddingCache(
            embedding=mock_emb, cache=cache, model_name="test-model",
        )

        assert cached_emb.get_dimension() == 768

    def test_different_models_have_separate_keys(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)

        mock_emb_a = MagicMock()
        mock_emb_a.embed.return_value = [[1.0, 0.0, 0.0]]
        mock_emb_b = MagicMock()
        mock_emb_b.embed.return_value = [[0.0, 1.0, 0.0]]

        cached_a = EmbeddingCache(
            embedding=mock_emb_a, cache=cache, model_name="model-A",
        )
        cached_b = EmbeddingCache(
            embedding=mock_emb_b, cache=cache, model_name="model-B",
        )

        # Same text, different model scopes
        result_a = cached_a.embed(["hello"])
        result_b = cached_b.embed(["hello"])

        # Both should have called their providers (separate cache keys)
        assert mock_emb_a.embed.call_count == 1
        assert mock_emb_b.embed.call_count == 1
        # Results differ because they come from different providers
        assert result_a != result_b
