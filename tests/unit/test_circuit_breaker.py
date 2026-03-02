"""Unit tests for CircuitBreaker — three-state machine."""

import time

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
