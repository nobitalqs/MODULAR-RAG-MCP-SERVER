"""Redis-backed session memory store using Hash per session."""

from __future__ import annotations

import json
import logging
from typing import Any

import redis

from src.libs.memory.base_memory import (
    BaseMemoryStore,
    ConversationTurn,
    SessionContext,
)

logger = logging.getLogger(__name__)

_KEY_PREFIX = "session:"


class RedisMemoryStore(BaseMemoryStore):
    """Session memory backed by Redis Hash.

    Each session is stored as a Redis Hash at key ``session:{session_id}``
    with fields ``turns`` (JSON list) and ``summary`` (string).
    TTL is refreshed on every write operation.

    Args:
        redis_url: Redis connection URL.
        session_ttl: Time-to-live in seconds for idle sessions.
    """

    def __init__(self, redis_url: str, session_ttl: int) -> None:
        self._session_ttl = session_ttl
        self._client = redis.from_url(redis_url, decode_responses=False)

    def _key(self, session_id: str) -> str:
        return f"{_KEY_PREFIX}{session_id}"

    def _refresh_ttl(self, session_id: str) -> None:
        self._client.expire(self._key(session_id), self._session_ttl)

    def _decode_turns(self, raw: bytes | None) -> tuple[ConversationTurn, ...]:
        if raw is None:
            return ()
        try:
            data = json.loads(raw)
            return tuple(ConversationTurn(role=t["role"], content=t["content"]) for t in data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to decode turns: %s", exc)
            return ()

    @staticmethod
    def _encode_turns(turns: tuple[ConversationTurn, ...]) -> str:
        return json.dumps([{"role": t.role, "content": t.content} for t in turns])

    def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        key = self._key(session_id)
        raw = self._client.hget(key, "turns")
        existing = self._decode_turns(raw)
        updated = (*existing, turn)
        self._client.hset(key, "turns", self._encode_turns(updated))
        self._refresh_ttl(session_id)

    def get_turns(self, session_id: str) -> tuple[ConversationTurn, ...]:
        raw = self._client.hget(self._key(session_id), "turns")
        return self._decode_turns(raw)

    def get_summary(self, session_id: str) -> str | None:
        raw = self._client.hget(self._key(session_id), "summary")
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else raw

    def set_summary(self, session_id: str, summary: str) -> None:
        self._client.hset(self._key(session_id), "summary", summary)
        self._refresh_ttl(session_id)

    def clear(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))

    def get_context(self, session_id: str) -> SessionContext:
        key = self._key(session_id)
        raw_turns = self._client.hget(key, "turns")
        raw_summary = self._client.hget(key, "summary")
        return SessionContext(
            session_id=session_id,
            turns=self._decode_turns(raw_turns),
            summary=raw_summary.decode() if isinstance(raw_summary, bytes) else raw_summary,
        )
