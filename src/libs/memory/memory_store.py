"""In-memory session store with per-session TTL."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.libs.memory.base_memory import (
    BaseMemoryStore,
    ConversationTurn,
    SessionContext,
)


@dataclass
class _SessionData:
    """Mutable internal state for a single session."""

    turns: list[ConversationTurn] = field(default_factory=list)
    summary: str | None = None
    last_access: float = field(default_factory=time.monotonic)


class InMemoryStore(BaseMemoryStore):
    """Thread-safe in-memory session store with TTL expiry.

    Sessions that have not been accessed within ``session_ttl`` seconds
    are treated as expired and their data is lazily cleaned up on access.

    Args:
        session_ttl: Time-to-live in seconds for idle sessions.
    """

    def __init__(self, session_ttl: int) -> None:
        self._session_ttl = session_ttl
        self._sessions: dict[str, _SessionData] = {}
        self._lock = threading.Lock()

    def _get_session(self, session_id: str) -> _SessionData | None:
        """Return session data if it exists and is not expired."""
        data = self._sessions.get(session_id)
        if data is None:
            return None
        if time.monotonic() - data.last_access >= self._session_ttl:
            del self._sessions[session_id]
            return None
        data.last_access = time.monotonic()
        return data

    def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            data = self._get_session(session_id)
            if data is None:
                data = _SessionData()
                self._sessions[session_id] = data
            data.turns.append(turn)
            data.last_access = time.monotonic()

    def get_turns(self, session_id: str) -> tuple[ConversationTurn, ...]:
        with self._lock:
            data = self._get_session(session_id)
            if data is None:
                return ()
            return tuple(data.turns)

    def get_summary(self, session_id: str) -> str | None:
        with self._lock:
            data = self._get_session(session_id)
            if data is None:
                return None
            return data.summary

    def set_summary(self, session_id: str, summary: str) -> None:
        with self._lock:
            data = self._get_session(session_id)
            if data is None:
                data = _SessionData()
                self._sessions[session_id] = data
            data.summary = summary
            data.last_access = time.monotonic()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def get_context(self, session_id: str) -> SessionContext:
        with self._lock:
            data = self._get_session(session_id)
            if data is None:
                return SessionContext(
                    session_id=session_id, turns=(), summary=None,
                )
            return SessionContext(
                session_id=session_id,
                turns=tuple(data.turns),
                summary=data.summary,
            )
