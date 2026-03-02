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
