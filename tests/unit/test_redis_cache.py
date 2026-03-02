"""Unit tests for RedisCache — uses mock to avoid Redis dependency."""

import pickle
from unittest.mock import MagicMock, patch

import pytest

from src.libs.cache.base_cache import BaseCache
from src.libs.cache.redis_cache import RedisCache


@pytest.fixture
def mock_redis():
    return MagicMock()


@pytest.fixture
def cache(mock_redis):
    with patch("src.libs.cache.redis_cache.redis.from_url", return_value=mock_redis):
        return RedisCache(redis_url="redis://localhost:6379/0", default_ttl=3600)


class TestRedisCacheBasic:
    def test_is_subclass_of_base_cache(self):
        assert issubclass(RedisCache, BaseCache)

    def test_get_existing(self, cache, mock_redis):
        mock_redis.get.return_value = pickle.dumps("v1")
        assert cache.get("k1") == "v1"
        mock_redis.get.assert_called_once_with("k1")

    def test_get_missing_returns_none(self, cache, mock_redis):
        mock_redis.get.return_value = None
        assert cache.get("missing") is None

    def test_set_with_default_ttl(self, cache, mock_redis):
        cache.set("k1", "v1")
        mock_redis.setex.assert_called_once_with("k1", 3600, pickle.dumps("v1"))

    def test_set_with_custom_ttl(self, cache, mock_redis):
        cache.set("k1", "v1", ttl=60)
        mock_redis.setex.assert_called_once_with("k1", 60, pickle.dumps("v1"))

    def test_delete_existing(self, cache, mock_redis):
        mock_redis.delete.return_value = 1
        assert cache.delete("k1") is True

    def test_delete_missing(self, cache, mock_redis):
        mock_redis.delete.return_value = 0
        assert cache.delete("missing") is False

    def test_exists_true(self, cache, mock_redis):
        mock_redis.exists.return_value = 1
        assert cache.exists("k1") is True

    def test_exists_false(self, cache, mock_redis):
        mock_redis.exists.return_value = 0
        assert cache.exists("missing") is False

    def test_stores_complex_types(self, cache, mock_redis):
        vec = [0.1, 0.2, 0.3]
        cache.set("vec", vec)
        mock_redis.setex.assert_called_once_with("vec", 3600, pickle.dumps(vec))
        mock_redis.get.return_value = pickle.dumps(vec)
        assert cache.get("vec") == vec
