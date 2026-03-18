"""LLM abstract base class and data types.

Defines the pluggable interface for Large Language Model providers.
All concrete LLM implementations (Azure, OpenAI, Ollama, DeepSeek)
must inherit from ``BaseLLM`` and implement the ``chat`` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

VALID_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class Message:
    """A single chat message.

    Attributes:
        role: One of ``system``, ``user``, or ``assistant``.
        content: The text content of the message.
    """

    role: str
    content: str


@dataclass(frozen=True)
class ChatResponse:
    """Unified response from any LLM provider.

    Attributes:
        content: Generated text.
        model: Model identifier used for this response.
        usage: Optional token usage statistics.
        raw_response: Optional provider-specific raw response for debugging.
    """

    content: str
    model: str
    usage: dict[str, int] | None = None
    raw_response: Any = None


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement :meth:`chat`. The base class provides
    :meth:`validate_messages` for input validation.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send messages to the LLM and return a response.

        Args:
            messages: Conversation messages in chronological order.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific overrides (temperature, etc.).

        Returns:
            A ChatResponse with the generated text.
        """

    def validate_messages(self, messages: list[Message]) -> None:
        """Validate a list of messages before sending to the provider.

        Args:
            messages: Messages to validate.

        Raises:
            ValueError: If the list is empty, a role is invalid,
                or content is empty/whitespace-only.
            TypeError: If any item is not a Message instance.
        """
        if not messages:
            raise ValueError("messages must contain at least one message")

        for msg in messages:
            if not isinstance(msg, Message):
                raise TypeError(f"Expected Message instance, got {type(msg).__name__}")
            if msg.role not in VALID_ROLES:
                raise ValueError(
                    f"Invalid role '{msg.role}'. Must be one of: {', '.join(sorted(VALID_ROLES))}"
                )
            if not msg.content or not msg.content.strip():
                raise ValueError(f"Message with role '{msg.role}' has empty content")
