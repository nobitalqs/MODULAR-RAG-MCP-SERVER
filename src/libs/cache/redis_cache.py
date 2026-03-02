"""Redis-backed cache with TTL."""

from __future__ import annotations

import logging
import pickle
from typing import Any

import redis

from src.libs.cache.base_cache import BaseCache

logger = logging.getLogger(__name__)


class RedisCache(BaseCache):
    """Cache backed by Redis. Values serialized with pickle.

    Args:
        redis_url: Redis connection URL.
        default_ttl: Default expiry in seconds.
    """

    def __init__(self, redis_url: str, default_ttl: int) -> None:
        self._default_ttl = default_ttl
        self._client = redis.from_url(redis_url, decode_responses=False)

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except (pickle.UnpicklingError, Exception) as exc:
            logger.warning("Cache unpickle failed for key=%s: %s", key, exc)
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        self._client.setex(key, effective_ttl, pickle.dumps(value))

    def delete(self, key: str) -> bool:
        return self._client.delete(key) > 0

    def exists(self, key: str) -> bool:
        return self._client.exists(key) > 0
