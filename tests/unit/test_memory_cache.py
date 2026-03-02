"""Unit tests for InMemoryCache — LRU + TTL."""

import time

import pytest

from src.libs.cache.base_cache import BaseCache
from src.libs.cache.memory_cache import InMemoryCache


class TestInMemoryCacheBasic:
    def test_set_and_get(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_get_missing_returns_none(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        assert cache.get("missing") is None

    def test_delete_existing_returns_true(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        cache.set("k1", "v1")
        assert cache.delete("k1") is True
        assert cache.get("k1") is None

    def test_delete_missing_returns_false(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        assert cache.delete("missing") is False

    def test_exists(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        cache.set("k1", "v1")
        assert cache.exists("k1") is True
        assert cache.exists("missing") is False

    def test_set_overwrites(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        cache.set("k1", "v1")
        cache.set("k1", "v2")
        assert cache.get("k1") == "v2"

    def test_stores_complex_types(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        cache.set("vec", [0.1, 0.2, 0.3])
        assert cache.get("vec") == [0.1, 0.2, 0.3]

    def test_is_subclass_of_base_cache(self):
        assert issubclass(InMemoryCache, BaseCache)


class TestInMemoryCacheTTL:
    def test_expired_entry_returns_none(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        cache.set("k1", "v1", ttl=0)
        # ttl=0 means already expired on next access
        time.sleep(0.01)
        assert cache.get("k1") is None

    def test_custom_ttl_overrides_default(self):
        cache = InMemoryCache(max_items=100, default_ttl=0)
        cache.set("k1", "v1", ttl=3600)
        assert cache.get("k1") == "v1"

    def test_exists_false_for_expired(self):
        cache = InMemoryCache(max_items=100, default_ttl=3600)
        cache.set("k1", "v1", ttl=0)
        time.sleep(0.01)
        assert cache.exists("k1") is False


class TestInMemoryCacheLRU:
    def test_evicts_oldest_when_full(self):
        cache = InMemoryCache(max_items=2, default_ttl=3600)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")  # should evict k1
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"

    def test_access_refreshes_lru_order(self):
        cache = InMemoryCache(max_items=2, default_ttl=3600)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.get("k1")  # refresh k1
        cache.set("k3", "v3")  # should evict k2, not k1
        assert cache.get("k1") == "v1"
        assert cache.get("k2") is None
        assert cache.get("k3") == "v3"

    def test_set_existing_does_not_increase_size(self):
        cache = InMemoryCache(max_items=2, default_ttl=3600)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k1", "v1-updated")  # overwrite, not new entry
        cache.set("k3", "v3")  # should evict k2
        assert cache.get("k1") == "v1-updated"
        assert cache.get("k2") is None
