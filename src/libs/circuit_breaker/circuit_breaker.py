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
