"""Unit tests for RateLimiterFactory."""

import pytest

from src.core.settings import RateLimitSettings
from src.libs.rate_limiter.base_limiter import BaseLimiter
from src.libs.rate_limiter.limiter_factory import RateLimiterFactory
from src.libs.rate_limiter.null_limiter import NullLimiter
from src.libs.rate_limiter.token_bucket import TokenBucketLimiter


class TestRateLimiterFactory:
    def test_disabled_returns_null_limiter(self):
        settings = RateLimitSettings(
            enabled=False, provider="token_bucket",
            requests_per_minute=60, max_concurrent=10,
        )
        limiter = RateLimiterFactory.create_from_settings(settings)
        assert isinstance(limiter, NullLimiter)

    def test_create_token_bucket(self):
        settings = RateLimitSettings(
            enabled=True, provider="token_bucket",
            requests_per_minute=60, max_concurrent=10,
        )
        limiter = RateLimiterFactory.create_from_settings(settings)
        assert isinstance(limiter, TokenBucketLimiter)
        assert isinstance(limiter, BaseLimiter)

    def test_unknown_provider_raises(self):
        settings = RateLimitSettings(
            enabled=True, provider="unknown",
            requests_per_minute=60, max_concurrent=10,
        )
        with pytest.raises(ValueError, match="Unknown rate limiter provider"):
            RateLimiterFactory.create_from_settings(settings)

    def test_create_default_returns_null(self):
        limiter = RateLimiterFactory.create_default()
        assert isinstance(limiter, NullLimiter)


class TestNullLimiter:
    def test_is_subclass(self):
        assert issubclass(NullLimiter, BaseLimiter)

    def test_acquire_always_true(self):
        limiter = NullLimiter()
        assert limiter.acquire() is True

    def test_release_is_noop(self):
        limiter = NullLimiter()
        limiter.release()  # should not raise

    def test_acquire_with_zero_timeout(self):
        limiter = NullLimiter()
        assert limiter.acquire(timeout=0) is True
