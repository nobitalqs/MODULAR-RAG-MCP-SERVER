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
