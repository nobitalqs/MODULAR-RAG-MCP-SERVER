"""Rate Limiter Factory — configuration-driven provider routing."""

from __future__ import annotations

from src.core.settings import RateLimitSettings
from src.libs.rate_limiter.base_limiter import BaseLimiter
from src.libs.rate_limiter.null_limiter import NullLimiter
from src.libs.rate_limiter.token_bucket import TokenBucketLimiter


class RateLimiterFactory:
    """Factory for creating rate limiter instances from settings."""

    @staticmethod
    def create_from_settings(settings: RateLimitSettings) -> BaseLimiter:
        if not settings.enabled:
            return NullLimiter()

        provider = settings.provider.lower()

        if provider == "token_bucket":
            return TokenBucketLimiter(
                rpm=settings.requests_per_minute,
                max_concurrent=settings.max_concurrent,
            )
        else:
            raise ValueError(
                f"Unknown rate limiter provider '{settings.provider}'. "
                f"Available: token_bucket"
            )

    @staticmethod
    def create_default() -> BaseLimiter:
        """Create a default no-op limiter (rate limiting disabled)."""
        return NullLimiter()
