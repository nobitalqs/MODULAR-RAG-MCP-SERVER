"""retry_with_backoff — exponential-backoff retry decorator.

Usage::

    from src.libs.resilience.retry import RetryableError, retry_with_backoff

    class OpenAIError(RetryableError):
        pass

    @retry_with_backoff(max_retries=3, backoff_base=1.0)
    def call_llm(prompt: str) -> str:
        ...
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

_DEFAULT_RETRYABLE_CODES: frozenset[int] = frozenset({429, 500, 502, 503})

F = TypeVar("F", bound=Callable)


class RetryableError(Exception):
    """Base exception for errors that carry an HTTP-like status_code.

    Subclasses should set ``status_code`` to allow the retry decorator to
    decide whether the error is transient.
    """

    status_code: int = 0


def _is_retryable(exc: BaseException, retryable_codes: frozenset[int]) -> bool:
    """Return True when *exc* warrants a retry attempt."""
    # Class name contains "timeout" (case-insensitive)
    if "timeout" in type(exc).__name__.lower():
        return True

    # RetryableError subclasses with a matching status code
    if isinstance(exc, RetryableError):
        return exc.status_code in retryable_codes

    return False


def retry_with_backoff(
    max_retries: int = 3,
    backoff_base: float = 1.0,
    retryable_codes: frozenset[int] | None = None,
) -> Callable[[F], F]:
    """Decorator factory: retry *func* up to *max_retries* times with exponential backoff.

    Args:
        max_retries: Number of additional attempts after the first failure.
        backoff_base: Base seconds for exponential delay: ``backoff_base * 2^attempt``.
        retryable_codes: Status codes to retry.  Defaults to {429, 500, 502, 503}.

    Returns:
        A decorator that wraps the target function with retry logic.
    """
    codes = retryable_codes if retryable_codes is not None else _DEFAULT_RETRYABLE_CODES

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except BaseException as exc:
                    if not _is_retryable(exc, codes):
                        # Non-retryable — propagate immediately
                        raise

                    last_exc = exc

                    if attempt < max_retries:
                        delay = backoff_base * (2**attempt)
                        logger.warning(
                            "Retry %d/%d for %s after %.3fs — %s: %s",
                            attempt + 1,
                            max_retries,
                            func.__qualname__,
                            delay,
                            type(exc).__name__,
                            exc,
                        )
                        time.sleep(delay)

            # All retries exhausted
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
