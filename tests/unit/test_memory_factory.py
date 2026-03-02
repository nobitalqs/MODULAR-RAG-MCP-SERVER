"""Unit tests for MemoryFactory."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import MemorySettings
from src.libs.memory.base_memory import BaseMemoryStore
from src.libs.memory.memory_factory import MemoryFactory
from src.libs.memory.memory_store import InMemoryStore


class TestMemoryFactory:
    def test_disabled_returns_in_memory(self):
        settings = MemorySettings(
            enabled=False, provider="redis", max_turns=20,
            summarize_threshold=10, summarize_enabled=True, session_ttl=3600,
        )
        store = MemoryFactory.create_from_settings(settings)
        assert isinstance(store, InMemoryStore)

    def test_provider_memory(self):
        settings = MemorySettings(
            enabled=True, provider="memory", max_turns=20,
            summarize_threshold=10, summarize_enabled=True, session_ttl=3600,
        )
        store = MemoryFactory.create_from_settings(settings)
        assert isinstance(store, InMemoryStore)
        assert isinstance(store, BaseMemoryStore)

    def test_provider_redis(self):
        settings = MemorySettings(
            enabled=True, provider="redis", max_turns=20,
            summarize_threshold=10, summarize_enabled=True, session_ttl=3600,
        )
        with patch("src.libs.memory.redis_memory.redis.from_url"):
            store = MemoryFactory.create_from_settings(
                settings, redis_url="redis://localhost:6379/0",
            )
        from src.libs.memory.redis_memory import RedisMemoryStore

        assert isinstance(store, RedisMemoryStore)

    def test_redis_without_url_raises(self):
        settings = MemorySettings(
            enabled=True, provider="redis", max_turns=20,
            summarize_threshold=10, summarize_enabled=True, session_ttl=3600,
        )
        with pytest.raises(ValueError, match="redis_url"):
            MemoryFactory.create_from_settings(settings)

    def test_unknown_provider_raises(self):
        settings = MemorySettings(
            enabled=True, provider="unknown", max_turns=20,
            summarize_threshold=10, summarize_enabled=True, session_ttl=3600,
        )
        with pytest.raises(ValueError, match="Unknown memory provider"):
            MemoryFactory.create_from_settings(settings)

    def test_create_default(self):
        store = MemoryFactory.create_default()
        assert isinstance(store, InMemoryStore)
