"""Conversation memory — business logic for sliding window + LLM compression."""

from __future__ import annotations

import logging
from typing import Any

from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.memory.base_memory import (
    BaseMemoryStore,
    ConversationTurn,
    SessionContext,
)

logger = logging.getLogger(__name__)

_COMPRESS_PROMPT = (
    "Summarise the following conversation into a concise paragraph that "
    "preserves key facts, decisions, and context needed for continuing "
    "the conversation. Write only the summary — no preamble."
)


class ConversationMemory:
    """Business logic layer over a memory store.

    Provides sliding-window retrieval and optional LLM-based compression
    when the number of turns exceeds a threshold.

    Args:
        store: Underlying storage provider.
        max_turns: Maximum turns returned in get_context / to_messages.
        summarize_threshold: Trigger compression when total turns exceed this.
        summarize_enabled: Whether LLM compression is active.
        llm: Optional LLM for generating summaries.
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        max_turns: int,
        summarize_threshold: int,
        summarize_enabled: bool = True,
        llm: BaseLLM | None = None,
    ) -> None:
        self._store = store
        self._max_turns = max_turns
        self._summarize_threshold = summarize_threshold
        self._summarize_enabled = summarize_enabled
        self._llm = llm

    def get_context(self, session_id: str) -> SessionContext:
        """Get windowed session context (last N turns + summary)."""
        ctx = self._store.get_context(session_id)
        windowed_turns = ctx.turns[-self._max_turns:]
        return SessionContext(
            session_id=ctx.session_id,
            turns=tuple(windowed_turns),
            summary=ctx.summary,
        )

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Add a turn and trigger compression if threshold exceeded."""
        self._store.add_turn(session_id, ConversationTurn(role=role, content=content))
        all_turns = self._store.get_turns(session_id)
        if (
            self._summarize_enabled
            and self._llm is not None
            and len(all_turns) > self._summarize_threshold
        ):
            self._compress(session_id, all_turns)

    def to_messages(self, session_id: str) -> list[Message]:
        """Convert windowed context to a list of LLM Message objects.

        If a summary exists, it is prepended as a system message.
        """
        ctx = self.get_context(session_id)
        if not ctx.turns and ctx.summary is None:
            return []

        messages: list[Message] = []
        if ctx.summary:
            messages.append(
                Message("system", f"Conversation summary: {ctx.summary}")
            )
        for turn in ctx.turns:
            messages.append(Message(turn.role, turn.content))
        return messages

    def _compress(
        self, session_id: str, turns: list[ConversationTurn],
    ) -> None:
        """Summarise older turns via LLM and store the result."""
        try:
            conversation_text = "\n".join(
                f"{t.role}: {t.content}" for t in turns
            )
            messages = [
                Message("system", _COMPRESS_PROMPT),
                Message("user", conversation_text),
            ]
            response = self._llm.chat(messages)
            summary = response.content.strip()
            if summary:
                self._store.set_summary(session_id, summary)
        except Exception as exc:
            logger.warning(
                "Failed to compress session %s: %s", session_id, exc,
            )
