"""Tests for retry_with_backoff decorator — TDD (write first, run RED, then GREEN)."""

from unittest.mock import patch

import pytest

from src.libs.resilience.retry import RetryableError, retry_with_backoff

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ServerError(RetryableError):
    """Simulates a 500-class error."""

    def __init__(self, code: int = 500):
        self.status_code = code
        super().__init__(f"server error {code}")


class NonRetryableError(RetryableError):
    """Simulates a 400 client error — should NOT be retried by default."""

    def __init__(self):
        self.status_code = 400
        super().__init__("client error 400")


class SimulatedTimeoutError(Exception):
    """Exception whose class name contains 'Timeout' — should be retried."""

    pass


# ---------------------------------------------------------------------------
# 1. No retry on immediate success
# ---------------------------------------------------------------------------


def test_no_retry_on_success():
    call_count = 0

    @retry_with_backoff(max_retries=3, backoff_base=0.001)
    def succeed():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = succeed()
    assert result == "ok"
    assert call_count == 1


# ---------------------------------------------------------------------------
# 2. Retries on retryable error, succeeds on 3rd attempt
# ---------------------------------------------------------------------------


def test_retries_on_retryable_error():
    call_count = 0

    @retry_with_backoff(max_retries=3, backoff_base=0.001)
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ServerError(500)
        return "recovered"

    with patch("time.sleep"):  # suppress actual sleeping
        result = flaky()

    assert result == "recovered"
    assert call_count == 3


# ---------------------------------------------------------------------------
# 3. Raises last exception after max_retries exhausted
# ---------------------------------------------------------------------------


def test_raises_after_max_retries():
    call_count = 0

    @retry_with_backoff(max_retries=3, backoff_base=0.001)
    def always_fail():
        nonlocal call_count
        call_count += 1
        raise ServerError(503)

    with patch("time.sleep"):
        with pytest.raises(ServerError):
            always_fail()

    # initial attempt + 3 retries = 4 total calls
    assert call_count == 4


# ---------------------------------------------------------------------------
# 4. No retry on non-retryable exception (plain ValueError)
# ---------------------------------------------------------------------------


def test_no_retry_on_non_retryable():
    call_count = 0

    @retry_with_backoff(max_retries=3, backoff_base=0.001)
    def bad_input():
        nonlocal call_count
        call_count += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        bad_input()

    assert call_count == 1


# ---------------------------------------------------------------------------
# 5. Custom retryable codes — only retry on specified codes
# ---------------------------------------------------------------------------


def test_custom_retryable_codes():
    call_count = 0

    @retry_with_backoff(max_retries=3, backoff_base=0.001, retryable_codes=frozenset({429}))
    def rate_limited():
        nonlocal call_count
        call_count += 1
        raise ServerError(429)

    with patch("time.sleep"):
        with pytest.raises(ServerError):
            rate_limited()

    # retried 3 times → 4 total calls
    assert call_count == 4


def test_custom_retryable_codes_non_matching():
    """Code 500 should NOT be retried when only 429 is in retryable_codes."""
    call_count = 0

    @retry_with_backoff(max_retries=3, backoff_base=0.001, retryable_codes=frozenset({429}))
    def server_error():
        nonlocal call_count
        call_count += 1
        raise ServerError(500)

    with pytest.raises(ServerError):
        server_error()

    assert call_count == 1


# ---------------------------------------------------------------------------
# 6. Timeout exception (class name contains "Timeout") is retried
# ---------------------------------------------------------------------------


def test_timeout_exception_retried():
    call_count = 0

    @retry_with_backoff(max_retries=3, backoff_base=0.001)
    def times_out():
        nonlocal call_count
        call_count += 1
        raise SimulatedTimeoutError("connection timed out")

    with patch("time.sleep"):
        with pytest.raises(SimulatedTimeoutError):
            times_out()

    assert call_count == 4  # initial + 3 retries


# ---------------------------------------------------------------------------
# 7. Backoff timing — delays are roughly exponential
# ---------------------------------------------------------------------------


def test_backoff_timing():
    """Verify sleep is called with exponentially growing delays."""
    sleep_calls: list[float] = []

    @retry_with_backoff(max_retries=3, backoff_base=0.5)
    def always_fail():
        raise ServerError(500)

    with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        with pytest.raises(ServerError):
            always_fail()

    # Expect 3 sleep calls for 3 retries (attempts 0, 1, 2)
    assert len(sleep_calls) == 3
    # Exponential: 0.5*2^0=0.5, 0.5*2^1=1.0, 0.5*2^2=2.0
    assert sleep_calls[0] == pytest.approx(0.5, rel=1e-3)
    assert sleep_calls[1] == pytest.approx(1.0, rel=1e-3)
    assert sleep_calls[2] == pytest.approx(2.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 8. RetryableError.status_code defaults to 0
# ---------------------------------------------------------------------------


def test_retryable_error_default_status_code():
    err = RetryableError("bare error")
    assert err.status_code == 0


# ---------------------------------------------------------------------------
# 9. Decorator preserves wrapped function name and docstring
# ---------------------------------------------------------------------------


def test_decorator_preserves_metadata():
    @retry_with_backoff()
    def my_function():
        """My docstring."""
        return 42

    assert my_function.__name__ == "my_function"
    assert "My docstring" in (my_function.__doc__ or "")
