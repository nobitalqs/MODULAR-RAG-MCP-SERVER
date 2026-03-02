"""Unit tests for RedisMemoryStore — uses mock to avoid Redis dependency."""

import json
from unittest.mock import MagicMock, call, patch

import pytest

from src.libs.memory.base_memory import (
    BaseMemoryStore,
    ConversationTurn,
    SessionContext,
)
from src.libs.memory.redis_memory import RedisMemoryStore


@pytest.fixture
def mock_redis():
    return MagicMock()


@pytest.fixture
def store(mock_redis):
    with patch("src.libs.memory.redis_memory.redis.from_url", return_value=mock_redis):
        return RedisMemoryStore(redis_url="redis://localhost:6379/0", session_ttl=3600)


class TestRedisMemoryStoreInterface:
    def test_is_subclass(self):
        assert issubclass(RedisMemoryStore, BaseMemoryStore)


class TestRedisMemoryStoreAddTurn:
    def test_add_turn_to_empty_session(self, store, mock_redis):
        mock_redis.hget.return_value = None
        turn = ConversationTurn("user", "hello")
        store.add_turn("s1", turn)

        # Should set turns field with JSON list
        set_call = mock_redis.hset.call_args
        assert set_call[0][0] == "session:s1"
        assert set_call[0][1] == "turns"
        stored = json.loads(set_call[0][2])
        assert len(stored) == 1
        assert stored[0]["role"] == "user"
        assert stored[0]["content"] == "hello"
        # Should refresh TTL
        mock_redis.expire.assert_called_with("session:s1", 3600)

    def test_add_turn_appends_to_existing(self, store, mock_redis):
        existing = json.dumps([{"role": "user", "content": "first"}])
        mock_redis.hget.return_value = existing.encode()
        store.add_turn("s1", ConversationTurn("assistant", "reply"))

        set_call = mock_redis.hset.call_args
        stored = json.loads(set_call[0][2])
        assert len(stored) == 2
        assert stored[1]["role"] == "assistant"


class TestRedisMemoryStoreGetTurns:
    def test_get_turns_existing(self, store, mock_redis):
        data = json.dumps([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        mock_redis.hget.return_value = data.encode()
        turns = store.get_turns("s1")
        assert len(turns) == 2
        assert turns[0] == ConversationTurn("user", "hi")
        assert turns[1] == ConversationTurn("assistant", "hello")

    def test_get_turns_missing_returns_empty(self, store, mock_redis):
        mock_redis.hget.return_value = None
        assert store.get_turns("nonexistent") == []


class TestRedisMemoryStoreSummary:
    def test_set_summary(self, store, mock_redis):
        store.set_summary("s1", "a summary")
        mock_redis.hset.assert_called_once_with("session:s1", "summary", "a summary")
        mock_redis.expire.assert_called_with("session:s1", 3600)

    def test_get_summary_existing(self, store, mock_redis):
        mock_redis.hget.return_value = b"a summary"
        assert store.get_summary("s1") == "a summary"

    def test_get_summary_missing(self, store, mock_redis):
        mock_redis.hget.return_value = None
        assert store.get_summary("s1") is None


class TestRedisMemoryStoreClear:
    def test_clear_deletes_key(self, store, mock_redis):
        store.clear("s1")
        mock_redis.delete.assert_called_once_with("session:s1")


class TestRedisMemoryStoreGetContext:
    def test_get_context_full(self, store, mock_redis):
        turns_data = json.dumps([{"role": "user", "content": "hi"}])
        mock_redis.hget.side_effect = lambda key, field: {
            "turns": turns_data.encode(),
            "summary": b"a summary",
        }.get(field)

        ctx = store.get_context("s1")
        assert isinstance(ctx, SessionContext)
        assert ctx.session_id == "s1"
        assert len(ctx.turns) == 1
        assert ctx.summary == "a summary"

    def test_get_context_empty(self, store, mock_redis):
        mock_redis.hget.return_value = None
        ctx = store.get_context("s1")
        assert ctx.turns == []
        assert ctx.summary is None
