"""Memory Factory — configuration-driven provider routing."""

from __future__ import annotations

from src.core.settings import MemorySettings
from src.libs.memory.base_memory import BaseMemoryStore
from src.libs.memory.memory_store import InMemoryStore

_DEFAULT_SESSION_TTL = 3600


class MemoryFactory:
    """Factory for creating memory store instances from settings."""

    @staticmethod
    def create_from_settings(
        settings: MemorySettings,
        redis_url: str | None = None,
    ) -> BaseMemoryStore:
        if not settings.enabled:
            return InMemoryStore(session_ttl=settings.session_ttl)

        provider = settings.provider.lower()

        if provider == "memory":
            return InMemoryStore(session_ttl=settings.session_ttl)

        elif provider == "redis":
            if not redis_url:
                raise ValueError(
                    "redis_url is required when memory provider is 'redis'"
                )
            from src.libs.memory.redis_memory import RedisMemoryStore

            return RedisMemoryStore(
                redis_url=redis_url,
                session_ttl=settings.session_ttl,
            )

        else:
            raise ValueError(
                f"Unknown memory provider '{settings.provider}'. "
                f"Available: memory, redis"
            )

    @staticmethod
    def create_default() -> BaseMemoryStore:
        """Create a default in-memory store (for when no config provided)."""
        return InMemoryStore(session_ttl=_DEFAULT_SESSION_TTL)
