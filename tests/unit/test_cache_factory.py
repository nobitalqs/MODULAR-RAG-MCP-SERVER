"""Unit tests for CacheFactory."""

from unittest.mock import patch

import pytest

from src.core.settings import CacheSettings
from src.libs.cache.base_cache import BaseCache
from src.libs.cache.cache_factory import CacheFactory
from src.libs.cache.memory_cache import InMemoryCache


class TestCacheFactory:
    def test_create_memory_provider(self):
        settings = CacheSettings(
            provider="memory", default_ttl=3600, max_memory_items=1000,
        )
        cache = CacheFactory.create_from_settings(settings)
        assert isinstance(cache, InMemoryCache)
        assert isinstance(cache, BaseCache)

    def test_create_redis_provider(self):
        settings = CacheSettings(
            provider="redis", default_ttl=3600, max_memory_items=1000,
            redis_url="redis://localhost:6379/0",
        )
        with patch("src.libs.cache.redis_cache.redis.from_url"):
            cache = CacheFactory.create_from_settings(settings)
        from src.libs.cache.redis_cache import RedisCache
        assert isinstance(cache, RedisCache)

    def test_unknown_provider_raises(self):
        settings = CacheSettings(
            provider="unknown", default_ttl=3600, max_memory_items=1000,
        )
        with pytest.raises(ValueError, match="Unknown cache provider"):
            CacheFactory.create_from_settings(settings)

    def test_redis_without_url_raises(self):
        settings = CacheSettings(
            provider="redis", default_ttl=3600, max_memory_items=1000,
            redis_url=None,
        )
        with pytest.raises(ValueError, match="redis_url"):
            CacheFactory.create_from_settings(settings)

    def test_none_settings_returns_memory_default(self):
        cache = CacheFactory.create_default()
        assert isinstance(cache, InMemoryCache)
