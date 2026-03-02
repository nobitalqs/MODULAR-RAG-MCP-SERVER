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
