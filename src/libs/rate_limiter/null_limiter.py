"""Null object limiter — always allows, used when rate limiting is disabled."""

from __future__ import annotations

from src.libs.rate_limiter.base_limiter import BaseLimiter


class NullLimiter(BaseLimiter):
    """No-op rate limiter. Always permits requests."""

    def acquire(self, timeout: float = 30.0) -> bool:
        return True

    def release(self) -> None:
        pass
