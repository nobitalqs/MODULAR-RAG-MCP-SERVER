"""Abstract base class and data types for session memory stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in a conversation.

    Attributes:
        role: Either ``"user"`` or ``"assistant"``.
        content: The text content of the turn.
    """

    role: str
    content: str


@dataclass(frozen=True)
class SessionContext:
    """Snapshot of a conversation session.

    Attributes:
        session_id: Unique identifier for the session.
        turns: Ordered list of conversation turns.
        summary: Optional compressed summary of earlier turns.
    """

    session_id: str
    turns: list[ConversationTurn]
    summary: str | None


class BaseMemoryStore(ABC):
    """Pluggable interface for conversation memory storage.

    Stores conversation turns per session, with optional summary.
    """

    @abstractmethod
    def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a turn to a session."""

    @abstractmethod
    def get_turns(self, session_id: str) -> list[ConversationTurn]:
        """Retrieve all turns for a session. Returns [] if expired/missing."""

    @abstractmethod
    def get_summary(self, session_id: str) -> str | None:
        """Retrieve the session summary. Returns None if unset/expired."""

    @abstractmethod
    def set_summary(self, session_id: str, summary: str) -> None:
        """Set or replace the session summary."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Remove all data for a session."""

    @abstractmethod
    def get_context(self, session_id: str) -> SessionContext:
        """Get full session context (turns + summary) as a snapshot."""
