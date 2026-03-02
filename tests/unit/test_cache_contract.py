"""Contract tests for BaseCache implementations.

Every cache provider must satisfy the same behavioral contract.
Uses @pytest.mark.parametrize to run identical assertions on all providers.
"""

from __future__ import annotations

import pytest

from src.libs.cache.base_cache import BaseCache
from src.libs.cache.memory_cache import InMemoryCache


def _make_memory_cache() -> BaseCache:
    return InMemoryCache(max_items=100, default_ttl=3600)


ALL_PROVIDERS = [
    pytest.param(_make_memory_cache, id="InMemoryCache"),
]


@pytest.mark.parametrize("factory", ALL_PROVIDERS)
class TestCacheContract:
    """Contract: all BaseCache implementations must satisfy these behaviours."""

    def test_get_returns_none_for_missing_key(self, factory):
        cache = factory()
        assert cache.get("nonexistent") is None

    def test_set_and_get_roundtrip(self, factory):
        cache = factory()
        cache.set("k1", {"data": [1, 2, 3]})
        assert cache.get("k1") == {"data": [1, 2, 3]}

    def test_exists_true_after_set(self, factory):
        cache = factory()
        cache.set("k1", "value")
        assert cache.exists("k1") is True

    def test_exists_false_for_missing(self, factory):
        cache = factory()
        assert cache.exists("never_set") is False

    def test_delete_removes_entry(self, factory):
        cache = factory()
        cache.set("k1", "val")
        assert cache.delete("k1") is True
        assert cache.get("k1") is None
        assert cache.exists("k1") is False

    def test_delete_returns_false_for_missing(self, factory):
        cache = factory()
        assert cache.delete("nonexistent") is False

    def test_overwrite_replaces_value(self, factory):
        cache = factory()
        cache.set("k1", "old")
        cache.set("k1", "new")
        assert cache.get("k1") == "new"

    def test_isinstance_base_cache(self, factory):
        cache = factory()
        assert isinstance(cache, BaseCache)
