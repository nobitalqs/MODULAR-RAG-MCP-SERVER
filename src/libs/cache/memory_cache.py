"""In-memory LRU cache with per-entry TTL."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

from src.libs.cache.base_cache import BaseCache


class InMemoryCache(BaseCache):
    """Thread-safe LRU cache with TTL support.

    Uses OrderedDict for O(1) LRU eviction. Each entry stores
    (value, expire_at) where expire_at is a monotonic timestamp.

    Args:
        max_items: Maximum entries before LRU eviction.
        default_ttl: Default time-to-live in seconds.
    """

    def __init__(self, max_items: int, default_ttl: int) -> None:
        self._max_items = max_items
        self._default_ttl = default_ttl
        self._data: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expire_at = entry
            if time.monotonic() >= expire_at:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expire_at = time.monotonic() + effective_ttl
        with self._lock:
            if key in self._data:
                del self._data[key]
            self._data[key] = (value, expire_at)
            self._data.move_to_end(key)
            while len(self._data) > self._max_items:
                self._data.popitem(last=False)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            _, expire_at = entry
            if time.monotonic() >= expire_at:
                del self._data[key]
                return False
            return True
