"""Cache Factory — configuration-driven provider routing."""

from __future__ import annotations

from src.core.settings import CacheSettings
from src.libs.cache.base_cache import BaseCache
from src.libs.cache.memory_cache import InMemoryCache


class CacheFactory:
    """Factory for creating cache instances from settings."""

    @staticmethod
    def create_from_settings(settings: CacheSettings) -> BaseCache:
        provider = settings.provider.lower()

        if provider == "memory":
            return InMemoryCache(
                max_items=settings.max_memory_items,
                default_ttl=settings.default_ttl,
            )
        elif provider == "redis":
            if not settings.redis_url:
                raise ValueError(
                    "redis_url is required when cache provider is 'redis'"
                )
            from src.libs.cache.redis_cache import RedisCache
            return RedisCache(
                redis_url=settings.redis_url,
                default_ttl=settings.default_ttl,
            )
        else:
            raise ValueError(
                f"Unknown cache provider '{settings.provider}'. "
                f"Available: memory, redis"
            )

    @staticmethod
    def create_default() -> BaseCache:
        """Create a default in-memory cache (for when no config provided)."""
        return InMemoryCache(max_items=10000, default_ttl=3600)
