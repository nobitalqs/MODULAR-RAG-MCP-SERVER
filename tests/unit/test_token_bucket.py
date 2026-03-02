"""Unit tests for TokenBucketLimiter."""

import time

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
