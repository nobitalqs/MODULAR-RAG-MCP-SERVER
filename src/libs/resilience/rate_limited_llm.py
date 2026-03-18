"""RateLimitedLLM — decorator-pattern wrapper that enforces rate limiting.

Wraps any BaseLLM implementation and calls acquire()/release() on the
provided BaseLimiter around every chat() call. The release() is always
called in a finally block so permits are returned even when the inner
LLM raises.
"""

from __future__ import annotations

import logging
from typing import Any

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.rate_limiter.base_limiter import BaseLimiter

logger = logging.getLogger(__name__)


class RateLimitedLLM(BaseLLM):
    """BaseLLM wrapper that acquires a rate-limit permit for every call.

    Args:
        llm: The inner LLM to delegate to.
        limiter: The rate limiter that controls access.
    """

    def __init__(self, llm: BaseLLM, limiter: BaseLimiter) -> None:
        self._llm = llm
        self._limiter = limiter

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send messages to the inner LLM, guarded by the rate limiter.

        Calls ``limiter.acquire()`` before forwarding to the inner LLM and
        ``limiter.release()`` in a finally block so the permit is always
        returned, even when the inner call raises.

        Args:
            messages: Conversation messages in chronological order.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific overrides forwarded verbatim.

        Returns:
            The ChatResponse from the inner LLM.

        Raises:
            RateLimitExceeded: If the limiter cannot acquire a permit.
            Any exception raised by the inner LLM.
        """
        self._limiter.acquire()
        try:
            return self._llm.chat(messages, trace=trace, **kwargs)
        finally:
            self._limiter.release()

    def validate_messages(self, messages: list[Message]) -> None:
        """Delegate message validation to the inner LLM.

        Args:
            messages: Messages to validate.

        Raises:
            ValueError: Propagated from the inner LLM's validate_messages.
            TypeError: Propagated from the inner LLM's validate_messages.
        """
        self._llm.validate_messages(messages)
