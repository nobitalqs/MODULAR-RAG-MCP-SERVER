# Phase J: Advanced Features — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 6 advanced features (cache, rate limiter, circuit breaker, query rewriter, session memory, query router) to the Modular RAG MCP Server.

**Architecture:** All features follow the existing Registry + Factory + Null Object pattern. Redis is optional — every feature degrades to in-process implementation. New settings dataclasses are frozen/immutable and parsed in `Settings.from_dict()`.

**Tech Stack:** Python 3.10+, redis-py (optional), pytest, existing BaseLLM/BaseEmbedding interfaces.

**Design doc:** `docs/plans/2026-03-02-advanced-features-design.md`

---

## Phase J1: Cache Infrastructure

### Task 1: Settings Dataclass — CacheSettings

**Files:**
- Modify: `src/core/settings.py`
- Modify: `tests/unit/test_config_loading.py`

**Step 1: Write failing test for CacheSettings parsing**

Add to `tests/unit/test_config_loading.py`:

```python
class TestCacheSettings:
    """Tests for optional cache configuration section."""

    def test_cache_defaults_to_none(self, tmp_path):
        """When no cache section, settings.cache is None."""
        cfg = tmp_path / "s.yaml"
        cfg.write_text(yaml.dump(MINIMAL_CONFIG))
        s = load_settings(cfg)
        assert s.cache is None

    def test_load_cache_memory_provider(self, tmp_path):
        data = {**MINIMAL_CONFIG, "cache": {
            "provider": "memory",
            "default_ttl": 3600,
            "max_memory_items": 10000,
        }}
        cfg = tmp_path / "s.yaml"
        cfg.write_text(yaml.dump(data))
        s = load_settings(cfg)
        assert s.cache is not None
        assert s.cache.provider == "memory"
        assert s.cache.default_ttl == 3600
        assert s.cache.max_memory_items == 10000
        assert s.cache.redis_url is None

    def test_load_cache_redis_provider(self, tmp_path):
        data = {**MINIMAL_CONFIG, "cache": {
            "provider": "redis",
            "redis_url": "redis://localhost:6379/0",
            "default_ttl": 1800,
            "max_memory_items": 5000,
        }}
        cfg = tmp_path / "s.yaml"
        cfg.write_text(yaml.dump(data))
        s = load_settings(cfg)
        assert s.cache.provider == "redis"
        assert s.cache.redis_url == "redis://localhost:6379/0"

    def test_cache_settings_are_frozen(self, tmp_path):
        data = {**MINIMAL_CONFIG, "cache": {
            "provider": "memory", "default_ttl": 3600, "max_memory_items": 10000,
        }}
        cfg = tmp_path / "s.yaml"
        cfg.write_text(yaml.dump(data))
        s = load_settings(cfg)
        with pytest.raises(FrozenInstanceError):
            s.cache.provider = "redis"
```

Note: `MINIMAL_CONFIG` is the existing fixture dict in the test file — check its exact name before using. `FrozenInstanceError` must be imported from `dataclasses`.

**Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/test_config_loading.py::TestCacheSettings -v
```

Expected: FAIL — `Settings` has no `cache` attribute.

**Step 3: Add CacheSettings dataclass and parse in Settings.from_dict()**

In `src/core/settings.py`, add after `DashboardSettings`:

```python
@dataclass(frozen=True)
class CacheSettings:
    provider: str             # "memory" | "redis"
    default_ttl: int          # seconds
    max_memory_items: int     # LRU cap for in-memory provider
    redis_url: str | None = None
```

Add `cache: CacheSettings | None = None` to `Settings`.

In `Settings.from_dict()`, add after dashboard parsing:

```python
cache_settings = None
if "cache" in data:
    cache_data = _require_mapping(data, "cache", "settings")
    cache_settings = CacheSettings(
        provider=_require_str(cache_data, "provider", "cache"),
        default_ttl=_require_int(cache_data, "default_ttl", "cache"),
        max_memory_items=_require_int(cache_data, "max_memory_items", "cache"),
        redis_url=cache_data.get("redis_url"),
    )
```

Pass `cache=cache_settings` in the `cls(...)` return.

**Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/test_config_loading.py::TestCacheSettings -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/core/settings.py tests/unit/test_config_loading.py
git commit -m "feat(settings): add CacheSettings dataclass"
```

---

### Task 2: BaseCache ABC + InMemoryCache

**Files:**
- Create: `src/libs/cache/__init__.py`
- Create: `src/libs/cache/base_cache.py`
- Create: `src/libs/cache/memory_cache.py`
- Create: `tests/unit/test_memory_cache.py`

**Step 1: Write failing tests**

Create `tests/unit/test_memory_cache.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/test_memory_cache.py -v
```

Expected: FAIL — modules don't exist yet.

**Step 3: Implement BaseCache and InMemoryCache**

Create `src/libs/cache/__init__.py`:

```python
"""Cache infrastructure — pluggable caching with LRU + TTL."""
```

Create `src/libs/cache/base_cache.py`:

```python
"""Abstract base class for cache providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):
    """Pluggable cache interface.

    All cache providers must implement get/set/delete/exists.
    Values can be any picklable Python object.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None if missing or expired."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value. ttl overrides default_ttl if provided."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed, False otherwise."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
```

Create `src/libs/cache/memory_cache.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/test_memory_cache.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/libs/cache/ tests/unit/test_memory_cache.py
git commit -m "feat(cache): add BaseCache ABC and InMemoryCache with LRU+TTL"
```

---

### Task 3: RedisCache

**Files:**
- Create: `src/libs/cache/redis_cache.py`
- Create: `tests/unit/test_redis_cache.py`

**Step 1: Write tests**

Create `tests/unit/test_redis_cache.py`:

```python
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
```

**Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_redis_cache.py -v
```

**Step 3: Implement RedisCache**

Create `src/libs/cache/redis_cache.py`:

```python
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
```

**Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_redis_cache.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/libs/cache/redis_cache.py tests/unit/test_redis_cache.py
git commit -m "feat(cache): add RedisCache with pickle serialization"
```

---

### Task 4: CacheFactory

**Files:**
- Create: `src/libs/cache/cache_factory.py`
- Create: `tests/unit/test_cache_factory.py`

**Step 1: Write failing tests**

Create `tests/unit/test_cache_factory.py`:

```python
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
```

**Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_cache_factory.py -v
```

**Step 3: Implement CacheFactory**

Create `src/libs/cache/cache_factory.py`:

```python
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
```

**Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_cache_factory.py -v
```

**Step 5: Commit**

```bash
git add src/libs/cache/cache_factory.py tests/unit/test_cache_factory.py
git commit -m "feat(cache): add CacheFactory with memory/redis routing"
```

---

### Task 5: EmbeddingCache (Decorator)

**Files:**
- Create: `src/libs/cache/embedding_cache.py`
- Create: `tests/unit/test_embedding_cache.py`

**Step 1: Write failing tests**

Create `tests/unit/test_embedding_cache.py`:

```python
"""Unit tests for EmbeddingCache — caching decorator for BaseEmbedding."""

from unittest.mock import MagicMock, call

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
```

**Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_embedding_cache.py -v
```

**Step 3: Implement EmbeddingCache**

Create `src/libs/cache/embedding_cache.py`:

```python
"""Caching decorator for BaseEmbedding — avoids redundant embedding calls."""

from __future__ import annotations

import hashlib
from typing import Any

from src.libs.cache.base_cache import BaseCache
from src.libs.embedding.base_embedding import BaseEmbedding


class EmbeddingCache(BaseEmbedding):
    """Wraps a BaseEmbedding to cache embed() results.

    Cache key format: ``emb:{model_name}:{sha256(text)[:16]}``.

    Args:
        embedding: The underlying embedding provider.
        cache: A BaseCache instance for storage.
        model_name: Model identifier for cache key scoping.
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        cache: BaseCache,
        model_name: str,
    ) -> None:
        self._embedding = embedding
        self._cache = cache
        self._model_name = model_name

    def _cache_key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"emb:{self._model_name}:{text_hash}"

    def embed(
        self,
        texts: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(self._cache_key(text))
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            fresh_vectors = self._embedding.embed(miss_texts, **kwargs)
            for idx, vector in zip(miss_indices, fresh_vectors):
                results[idx] = vector
                self._cache.set(self._cache_key(texts[idx]), vector)

        return results  # type: ignore[return-value]

    def validate_texts(self, texts: list[str]) -> None:
        self._embedding.validate_texts(texts)

    def get_dimension(self) -> int:
        return self._embedding.get_dimension()
```

**Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_embedding_cache.py -v
```

**Step 5: Commit**

```bash
git add src/libs/cache/embedding_cache.py tests/unit/test_embedding_cache.py
git commit -m "feat(cache): add EmbeddingCache decorator with hit/miss splitting"
```

---

## Phase J2: Rate Limiter

### Task 6: Settings — RateLimitSettings

**Files:**
- Modify: `src/core/settings.py`
- Modify: `tests/unit/test_config_loading.py`

Same pattern as Task 1. Add `RateLimitSettings` dataclass:

```python
@dataclass(frozen=True)
class RateLimitSettings:
    enabled: bool
    provider: str             # "token_bucket" | "redis"
    requests_per_minute: int
    max_concurrent: int
    tokens_per_minute: int | None = None
    redis_url: str | None = None  # reuses cache.redis_url if not set
```

Add `rate_limit: RateLimitSettings | None = None` to `Settings`. Parse in `from_dict()`.

Write 3 tests: default None, load with values, frozen check. Commit.

---

### Task 7: TokenBucketLimiter

**Files:**
- Create: `src/libs/rate_limiter/__init__.py`
- Create: `src/libs/rate_limiter/base_limiter.py`
- Create: `src/libs/rate_limiter/token_bucket.py`
- Create: `tests/unit/test_token_bucket.py`

**Step 1: Write failing tests**

Create `tests/unit/test_token_bucket.py`:

```python
"""Unit tests for TokenBucketLimiter."""

import time
from unittest.mock import patch

import pytest

from src.libs.rate_limiter.base_limiter import BaseLimiter, RateLimitExceeded
from src.libs.rate_limiter.token_bucket import TokenBucketLimiter


class TestTokenBucketBasic:
    def test_is_subclass(self):
        assert issubclass(TokenBucketLimiter, BaseLimiter)

    def test_acquire_succeeds_when_tokens_available(self):
        limiter = TokenBucketLimiter(rpm=60, max_concurrent=10)
        assert limiter.acquire() is True

    def test_acquire_multiple_within_capacity(self):
        limiter = TokenBucketLimiter(rpm=60, max_concurrent=10)
        for _ in range(10):
            assert limiter.acquire() is True

    def test_acquire_fails_when_exhausted(self):
        limiter = TokenBucketLimiter(rpm=60, max_concurrent=2)
        limiter.acquire()
        limiter.acquire()
        with pytest.raises(RateLimitExceeded):
            limiter.acquire(timeout=0)

    def test_release_frees_concurrent_slot(self):
        limiter = TokenBucketLimiter(rpm=60, max_concurrent=1)
        limiter.acquire()
        limiter.release()
        assert limiter.acquire() is True

    def test_release_idempotent(self):
        limiter = TokenBucketLimiter(rpm=60, max_concurrent=1)
        limiter.acquire()
        limiter.release()
        limiter.release()  # should not raise


class TestTokenBucketRefill:
    def test_tokens_refill_over_time(self):
        limiter = TokenBucketLimiter(rpm=6000, max_concurrent=100)
        # Consume all initial tokens
        for _ in range(100):
            limiter.acquire()
            limiter.release()
        # After a small sleep, tokens should refill
        time.sleep(0.1)  # 6000/60 = 100/sec, 0.1s = ~10 tokens
        assert limiter.acquire() is True

    def test_capacity_does_not_exceed_max(self):
        limiter = TokenBucketLimiter(rpm=60, max_concurrent=10)
        time.sleep(0.1)  # let tokens accumulate
        # Should still be capped at max_concurrent
        count = 0
        for _ in range(20):
            try:
                limiter.acquire(timeout=0)
                count += 1
            except RateLimitExceeded:
                break
        assert count <= 10
```

**Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_token_bucket.py -v
```

**Step 3: Implement BaseLimiter and TokenBucketLimiter**

Create `src/libs/rate_limiter/__init__.py`:

```python
"""Rate limiting — token bucket and Redis sliding window."""
```

Create `src/libs/rate_limiter/base_limiter.py`:

```python
"""Abstract base class for rate limiters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and timeout is 0."""


class BaseLimiter(ABC):
    @abstractmethod
    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire a rate limit permit. Raises RateLimitExceeded if timeout=0 and unavailable."""

    @abstractmethod
    def release(self) -> None:
        """Release a concurrent permit (idempotent)."""
```

Create `src/libs/rate_limiter/token_bucket.py`:

```python
"""Token bucket rate limiter — in-process, thread-safe."""

from __future__ import annotations

import threading
import time

from src.libs.rate_limiter.base_limiter import BaseLimiter, RateLimitExceeded


class TokenBucketLimiter(BaseLimiter):
    """Token bucket with concurrent request tracking.

    Args:
        rpm: Requests per minute — determines refill rate.
        max_concurrent: Maximum simultaneous in-flight requests.
    """

    def __init__(self, rpm: int, max_concurrent: int) -> None:
        self._capacity = max_concurrent
        self._tokens = float(max_concurrent)
        self._refill_rate = rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._concurrent = 0
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._tokens + elapsed * self._refill_rate,
            float(self._capacity),
        )
        self._last_refill = now

    def acquire(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0 and self._concurrent < self._capacity:
                    self._tokens -= 1.0
                    self._concurrent += 1
                    return True
            if timeout == 0 or time.monotonic() >= deadline:
                raise RateLimitExceeded(
                    f"Rate limit exceeded (capacity={self._capacity})"
                )
            time.sleep(0.01)

    def release(self) -> None:
        with self._lock:
            if self._concurrent > 0:
                self._concurrent -= 1
```

**Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_token_bucket.py -v
```

**Step 5: Commit**

```bash
git add src/libs/rate_limiter/ tests/unit/test_token_bucket.py
git commit -m "feat(rate-limiter): add BaseLimiter ABC and TokenBucketLimiter"
```

---

### Task 8: RateLimiterFactory

**Files:**
- Create: `src/libs/rate_limiter/limiter_factory.py`
- Create: `tests/unit/test_limiter_factory.py`

Same factory pattern as CacheFactory. When `enabled=False`, return a `NullLimiter` (always allows, release is no-op). Support `token_bucket` and `redis` providers. Commit.

---

## Phase J3: Circuit Breaker + Provider Failover

### Task 9: CircuitBreaker

**Files:**
- Create: `src/libs/circuit_breaker/__init__.py`
- Create: `src/libs/circuit_breaker/circuit_breaker.py`
- Create: `tests/unit/test_circuit_breaker.py`

**Step 1: Write failing tests**

Create `tests/unit/test_circuit_breaker.py`:

```python
"""Unit tests for CircuitBreaker — three-state machine."""

import time
from unittest.mock import patch

import pytest

from src.libs.circuit_breaker.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreakerStates:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown=60)
        assert cb.state == CircuitState.CLOSED

    def test_closed_allows_requests(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown=60)
        assert cb.allow_request() is True

    def test_transitions_to_open_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_requests(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown=60)
        cb.record_failure()
        assert cb.allow_request() is False

    def test_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown=0.05)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_one_request(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.allow_request() is True

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()  # count restarted from 0
        assert cb.state == CircuitState.CLOSED

    def test_failure_count_below_threshold_stays_closed(self):
        cb = CircuitBreaker(failure_threshold=5, cooldown=60)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerDecorator:
    def test_decorator_passes_on_success(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown=60)

        @cb.protect
        def good_func():
            return "ok"

        assert good_func() == "ok"

    def test_decorator_records_failure(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown=60)

        @cb.protect
        def bad_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            bad_func()
        with pytest.raises(RuntimeError):
            bad_func()
        assert cb.state == CircuitState.OPEN

    def test_decorator_raises_circuit_open(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown=60)
        cb.record_failure()

        @cb.protect
        def func():
            return "ok"

        with pytest.raises(CircuitOpenError):
            func()
```

**Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_circuit_breaker.py -v
```

**Step 3: Implement CircuitBreaker**

Create `src/libs/circuit_breaker/__init__.py`:

```python
"""Circuit breaker pattern for LLM provider resilience."""
```

Create `src/libs/circuit_breaker/circuit_breaker.py`:

```python
"""Three-state circuit breaker with decorator support."""

from __future__ import annotations

import enum
import functools
import threading
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    """Three-state circuit breaker: CLOSED -> OPEN -> HALF_OPEN.

    Args:
        failure_threshold: Consecutive failures to trigger OPEN.
        cooldown: Seconds before transitioning OPEN -> HALF_OPEN.
    """

    def __init__(self, failure_threshold: int = 5, cooldown: float = 60.0) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self._cooldown:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def allow_request(self) -> bool:
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN

    def protect(self, func: F) -> F:
        """Decorator that wraps a function with circuit breaker logic."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.allow_request():
                raise CircuitOpenError(
                    f"Circuit is {self.state.value}, call rejected"
                )
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except CircuitOpenError:
                raise
            except Exception:
                self.record_failure()
                raise

        return wrapper  # type: ignore[return-value]
```

**Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_circuit_breaker.py -v
```

**Step 5: Commit**

```bash
git add src/libs/circuit_breaker/ tests/unit/test_circuit_breaker.py
git commit -m "feat(circuit-breaker): add three-state CircuitBreaker with decorator"
```

---

### Task 10: ProviderChain

**Files:**
- Create: `src/libs/circuit_breaker/provider_chain.py`
- Create: `tests/unit/test_provider_chain.py`

**Step 1: Write failing tests**

Create `tests/unit/test_provider_chain.py`:

```python
"""Unit tests for ProviderChain — multi-provider failover."""

from unittest.mock import MagicMock

import pytest

from src.libs.circuit_breaker.circuit_breaker import CircuitBreaker, CircuitState
from src.libs.circuit_breaker.provider_chain import (
    AllProvidersUnavailableError,
    ProviderChain,
)
from src.libs.llm.base_llm import ChatResponse, Message


def _make_llm(name: str, fail: bool = False) -> MagicMock:
    llm = MagicMock()
    llm.__class__.__name__ = name
    if fail:
        llm.chat.side_effect = RuntimeError(f"{name} down")
    else:
        llm.chat.return_value = ChatResponse(content=f"from {name}", model=name)
    return llm


class TestProviderChain:
    def test_first_provider_succeeds(self):
        llm1 = _make_llm("primary")
        chain = ProviderChain([(llm1, CircuitBreaker())])
        msgs = [Message("user", "hi")]
        result = chain.chat(msgs)
        assert result.content == "from primary"

    def test_failover_to_second(self):
        llm1 = _make_llm("primary", fail=True)
        llm2 = _make_llm("backup")
        chain = ProviderChain([
            (llm1, CircuitBreaker(failure_threshold=1)),
            (llm2, CircuitBreaker()),
        ])
        msgs = [Message("user", "hi")]
        result = chain.chat(msgs)
        assert result.content == "from backup"

    def test_skips_circuit_open_provider(self):
        llm1 = _make_llm("primary")
        cb1 = CircuitBreaker(failure_threshold=1)
        cb1.record_failure()  # opens circuit
        assert cb1.state == CircuitState.OPEN

        llm2 = _make_llm("backup")
        chain = ProviderChain([(llm1, cb1), (llm2, CircuitBreaker())])
        result = chain.chat([Message("user", "hi")])
        assert result.content == "from backup"
        llm1.chat.assert_not_called()

    def test_all_unavailable_raises(self):
        llm1 = _make_llm("p1", fail=True)
        llm2 = _make_llm("p2", fail=True)
        chain = ProviderChain([
            (llm1, CircuitBreaker(failure_threshold=1)),
            (llm2, CircuitBreaker(failure_threshold=1)),
        ])
        with pytest.raises(AllProvidersUnavailableError):
            chain.chat([Message("user", "hi")])

    def test_kwargs_forwarded(self):
        llm1 = _make_llm("primary")
        chain = ProviderChain([(llm1, CircuitBreaker())])
        chain.chat([Message("user", "hi")], temperature=0.5)
        llm1.chat.assert_called_once_with([Message("user", "hi")], temperature=0.5)
```

**Step 2: Run to verify failure, Step 3: Implement, Step 4: Run, Step 5: Commit**

Create `src/libs/circuit_breaker/provider_chain.py`:

```python
"""Multi-provider failover chain with circuit breaker integration."""

from __future__ import annotations

import logging
from typing import Any

from src.libs.circuit_breaker.circuit_breaker import CircuitBreaker
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message

logger = logging.getLogger(__name__)


class AllProvidersUnavailableError(Exception):
    """Raised when all providers in the chain are unavailable."""


class ProviderChain:
    """Tries LLM providers in priority order, skipping circuit-open ones.

    Args:
        providers: List of (BaseLLM, CircuitBreaker) tuples in priority order.
    """

    def __init__(self, providers: list[tuple[BaseLLM, CircuitBreaker]]) -> None:
        if not providers:
            raise ValueError("ProviderChain requires at least one provider")
        self._providers = providers

    def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        errors: list[tuple[str, Exception]] = []

        for llm, breaker in self._providers:
            if not breaker.allow_request():
                logger.info("Skipping %s (circuit open)", llm.__class__.__name__)
                continue
            try:
                result = llm.chat(messages, **kwargs)
                breaker.record_success()
                return result
            except Exception as exc:
                breaker.record_failure()
                name = llm.__class__.__name__
                logger.warning("Provider %s failed: %s", name, exc)
                errors.append((name, exc))

        raise AllProvidersUnavailableError(
            f"All {len(self._providers)} providers unavailable. "
            f"Errors: {errors}"
        )
```

Commit:

```bash
git add src/libs/circuit_breaker/provider_chain.py tests/unit/test_provider_chain.py
git commit -m "feat(circuit-breaker): add ProviderChain for multi-provider failover"
```

---

### Task 11: CircuitBreaker Settings + LLMFactory Integration

**Files:**
- Modify: `src/core/settings.py` — add `CircuitBreakerSettings`, `FallbackProviderSettings`, extend `LLMSettings`
- Modify: `src/libs/llm/llm_factory.py` — add `create_with_failover()`
- Tests for both

Add to `LLMSettings`:

```python
@dataclass(frozen=True)
class CircuitBreakerSettings:
    enabled: bool
    failure_threshold: int
    cooldown_seconds: float
    half_open_max_calls: int = 1

@dataclass(frozen=True)
class FallbackProviderSettings:
    provider: str
    model: str
    api_key: str | None = None
    azure_endpoint: str | None = None
    base_url: str | None = None
```

Add `circuit_breaker: CircuitBreakerSettings | None = None` and `fallback_providers: list[FallbackProviderSettings] | None = None` to `LLMSettings`.

Add `create_with_failover(settings, llm_factory) -> ProviderChain | BaseLLM` to `LLMFactory`. When `fallback_providers` is non-empty, create a ProviderChain. Otherwise wrap single provider with circuit breaker.

Commit.

---

## Phase J4: Query Rewriter

### Task 12: Settings — QueryRewritingSettings

Same pattern. Add dataclass, parse in `from_dict()`, tests, commit.

```python
@dataclass(frozen=True)
class QueryRewritingSettings:
    enabled: bool
    provider: str          # "none" | "llm" | "hyde"
    max_rewrites: int
    model: str | None = None
```

---

### Task 13: BaseQueryRewriter + NoneRewriter

**Files:**
- Create: `src/libs/query_rewriter/__init__.py`
- Create: `src/libs/query_rewriter/base_rewriter.py`
- Create: `src/libs/query_rewriter/none_rewriter.py`
- Create: `tests/unit/test_none_rewriter.py`

Implement `RewriteResult` (frozen dataclass) and `BaseQueryRewriter(ABC)`. `NoneRewriter` returns `RewriteResult(original_query=query, rewritten_queries=[query], reasoning=None, strategy="none")`.

Commit.

---

### Task 14: LLMRewriter

**Files:**
- Create: `src/libs/query_rewriter/llm_rewriter.py`
- Create: `tests/unit/test_llm_rewriter.py`

Mock `BaseLLM.chat()` to return JSON `{"queries": [...], "reasoning": "..."}`. Test three modes:
1. With `conversation_history` → context completion
2. Complex query → decomposition (multiple queries)
3. Simple query → single rewrite

Commit.

---

### Task 15: HyDERewriter

**Files:**
- Create: `src/libs/query_rewriter/hyde_rewriter.py`
- Create: `tests/unit/test_hyde_rewriter.py`

LLM generates hypothetical answer document. Returns `strategy="hyde"`. Commit.

---

### Task 16: QueryRewriterFactory

**Files:**
- Create: `src/libs/query_rewriter/rewriter_factory.py`
- Create: `tests/unit/test_rewriter_factory.py`

When `enabled=False` or `provider="none"`, return `NoneRewriter`. Otherwise route to `LLMRewriter` or `HyDERewriter`. Commit.

---

## Phase J5: Session Memory

### Task 17: Settings — MemorySettings

Same pattern. Commit.

```python
@dataclass(frozen=True)
class MemorySettings:
    enabled: bool
    provider: str           # "memory" | "redis"
    max_turns: int
    summarize_threshold: int
    summarize_enabled: bool
    session_ttl: int
```

---

### Task 18: BaseMemoryStore + InMemoryStore

**Files:**
- Create: `src/libs/memory/__init__.py`
- Create: `src/libs/memory/base_memory.py`
- Create: `src/libs/memory/memory_store.py`
- Create: `tests/unit/test_memory_store.py`

Implement `ConversationTurn`, `SessionContext` (frozen dataclasses), `BaseMemoryStore(ABC)`, `InMemoryStore`.

Key tests:
- add_turn → get_turns roundtrip
- TTL expiry (session_ttl)
- clear by session_id
- empty session returns []
- get_summary / set_summary

Commit.

---

### Task 19: RedisMemoryStore

**Files:**
- Create: `src/libs/memory/redis_memory.py`
- Create: `tests/unit/test_redis_memory.py`

Uses Redis Hash. Key = `session:{session_id}`. Fields: `turns` (JSON), `summary` (str). TTL via `EXPIRE`. Mock Redis in tests. Commit.

---

### Task 20: ConversationMemory (Business Logic)

**Files:**
- Create: `src/libs/memory/conversation_memory.py`
- Create: `tests/unit/test_conversation_memory.py`

Tests:
- `get_context` returns sliding window (last N turns)
- `add_turn` triggers `_compress` when > threshold
- `_compress` calls LLM with correct prompt
- `to_messages()` returns system summary + turn pairs
- `summarize_enabled=False` → truncate only
- `llm=None` → truncate only

Commit.

---

### Task 21: MemoryFactory

**Files:**
- Create: `src/libs/memory/memory_factory.py`
- Create: `tests/unit/test_memory_factory.py`

Commit.

---

## Phase J6: Query Router

### Task 22: Settings — QueryRoutingSettings

```python
@dataclass(frozen=True)
class RouteConfig:
    name: str
    description: str

@dataclass(frozen=True)
class QueryRoutingSettings:
    enabled: bool
    provider: str          # "none" | "llm"
    routes: list[RouteConfig]
    model: str | None = None
```

Commit.

---

### Task 23: BaseQueryRouter + NoneRouter

**Files:**
- Create: `src/libs/query_router/__init__.py`
- Create: `src/libs/query_router/base_router.py`
- Create: `src/libs/query_router/none_router.py`
- Create: `tests/unit/test_none_router.py`

`RouteDecision` (frozen dataclass). `NoneRouter` always returns `knowledge_search` with `confidence=1.0`. Commit.

---

### Task 24: LLMRouter

**Files:**
- Create: `src/libs/query_router/llm_router.py`
- Create: `tests/unit/test_llm_router.py`

Mock LLM returns JSON `{"route": "...", "confidence": 0.95, "tool_name": null, "reasoning": "..."}`. Test all three route types + invalid JSON fallback. Commit.

---

### Task 25: QueryRouterFactory

**Files:**
- Create: `src/libs/query_router/router_factory.py`
- Create: `tests/unit/test_router_factory.py`

When `enabled=False`, return `NoneRouter`. Commit.

---

## Phase J-Integration: Wire Everything Together

### Task 26: Integrate into query_knowledge_hub

**Files:**
- Modify: `src/mcp_server/tools/query_knowledge_hub.py`
- Modify: `tests/unit/test_query_knowledge_hub_advanced.py` (new)

Add optional `session_id` to input schema. Wire:
1. `ConversationMemory.get_context(session_id)` if session_id provided
2. `QueryRewriter.rewrite(query, conversation_history=context.to_messages())`
3. Existing `QueryProcessor.process()` + `HybridSearch` + `CoreReranker`
4. `RateLimiter.acquire()` / `release()` around LLM/embedding calls
5. `ConversationMemory.add_turn()` after response

Lazy-init all new components from Settings (same pattern as existing embedding/vector_store init).

Commit.

---

### Task 27: Contract Tests

**Files:**
- Create: `tests/unit/test_cache_contract.py`
- Create: `tests/unit/test_rewriter_contract.py`
- Create: `tests/unit/test_memory_store_contract.py`
- Create: `tests/unit/test_router_contract.py`

Use `@pytest.mark.parametrize` to run same assertions on all providers of each ABC. Commit.

---

### Task 28: Integration Tests

**Files:**
- Create: `tests/integration/test_cache_with_embedding.py`
- Create: `tests/integration/test_rewriter_with_hybrid_search.py`
- Create: `tests/integration/test_memory_with_rewriter.py`
- Create: `tests/integration/test_circuit_breaker_with_llm.py`

Commit.

---

### Task 29: Update config/settings.yaml.example

Add all new sections with commented defaults. Commit.

---

### Task 30: Final Verification

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
.venv/bin/python -m pytest tests/ --cov=src/libs/cache --cov=src/libs/rate_limiter --cov=src/libs/circuit_breaker --cov=src/libs/query_rewriter --cov=src/libs/memory --cov=src/libs/query_router --cov-report=term-missing
```

Verify:
- [ ] All tests pass
- [ ] Coverage ≥ 80% per new module
- [ ] No import errors
- [ ] `settings.yaml.example` includes all new sections

Commit all remaining changes.
