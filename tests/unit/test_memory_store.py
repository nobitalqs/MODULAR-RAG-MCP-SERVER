"""Unit tests for BaseMemoryStore ABC and InMemoryStore."""

import time
from dataclasses import FrozenInstanceError

import pytest

from src.libs.memory.base_memory import (
    BaseMemoryStore,
    ConversationTurn,
    SessionContext,
)
from src.libs.memory.memory_store import InMemoryStore


class TestConversationTurn:
    def test_fields(self):
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"

    def test_frozen(self):
        turn = ConversationTurn(role="user", content="hello")
        with pytest.raises(FrozenInstanceError):
            turn.content = "changed"


class TestSessionContext:
    def test_fields(self):
        turns = (ConversationTurn("user", "hi"), ConversationTurn("assistant", "hello"))
        ctx = SessionContext(session_id="s1", turns=turns, summary=None)
        assert ctx.session_id == "s1"
        assert len(ctx.turns) == 2
        assert ctx.summary is None

    def test_frozen(self):
        ctx = SessionContext(session_id="s1", turns=(), summary=None)
        with pytest.raises(FrozenInstanceError):
            ctx.session_id = "s2"


class TestInMemoryStoreInterface:
    def test_is_subclass(self):
        assert issubclass(InMemoryStore, BaseMemoryStore)


class TestInMemoryStoreBasic:
    def test_add_turn_and_get_turns(self):
        store = InMemoryStore(session_ttl=3600)
        store.add_turn("s1", ConversationTurn("user", "hello"))
        store.add_turn("s1", ConversationTurn("assistant", "hi there"))
        turns = store.get_turns("s1")
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"

    def test_get_turns_empty_session(self):
        store = InMemoryStore(session_ttl=3600)
        turns = store.get_turns("nonexistent")
        assert turns == ()

    def test_multiple_sessions_isolated(self):
        store = InMemoryStore(session_ttl=3600)
        store.add_turn("s1", ConversationTurn("user", "session 1"))
        store.add_turn("s2", ConversationTurn("user", "session 2"))
        assert len(store.get_turns("s1")) == 1
        assert len(store.get_turns("s2")) == 1
        assert store.get_turns("s1")[0].content == "session 1"

    def test_clear_session(self):
        store = InMemoryStore(session_ttl=3600)
        store.add_turn("s1", ConversationTurn("user", "hello"))
        store.clear("s1")
        assert store.get_turns("s1") == ()

    def test_clear_nonexistent_does_not_raise(self):
        store = InMemoryStore(session_ttl=3600)
        store.clear("nonexistent")  # should not raise


class TestInMemoryStoreSummary:
    def test_get_summary_default_none(self):
        store = InMemoryStore(session_ttl=3600)
        assert store.get_summary("s1") is None

    def test_set_and_get_summary(self):
        store = InMemoryStore(session_ttl=3600)
        store.set_summary("s1", "This is a summary")
        assert store.get_summary("s1") == "This is a summary"

    def test_summary_survives_add_turn(self):
        store = InMemoryStore(session_ttl=3600)
        store.set_summary("s1", "summary text")
        store.add_turn("s1", ConversationTurn("user", "new msg"))
        assert store.get_summary("s1") == "summary text"

    def test_clear_removes_summary(self):
        store = InMemoryStore(session_ttl=3600)
        store.set_summary("s1", "summary")
        store.clear("s1")
        assert store.get_summary("s1") is None


class TestInMemoryStoreTTL:
    def test_expired_session_returns_empty(self):
        store = InMemoryStore(session_ttl=0)  # immediate expiry
        store.add_turn("s1", ConversationTurn("user", "hello"))
        time.sleep(0.01)
        assert store.get_turns("s1") == ()

    def test_expired_summary_returns_none(self):
        store = InMemoryStore(session_ttl=0)
        store.set_summary("s1", "summary")
        time.sleep(0.01)
        assert store.get_summary("s1") is None

    def test_access_refreshes_ttl(self):
        store = InMemoryStore(session_ttl=3600)
        store.add_turn("s1", ConversationTurn("user", "hello"))
        # Access keeps session alive
        turns = store.get_turns("s1")
        assert len(turns) == 1


class TestInMemoryStoreGetContext:
    def test_get_context_returns_session_context(self):
        store = InMemoryStore(session_ttl=3600)
        store.add_turn("s1", ConversationTurn("user", "hello"))
        store.add_turn("s1", ConversationTurn("assistant", "hi"))
        store.set_summary("s1", "greeting exchange")
        ctx = store.get_context("s1")
        assert isinstance(ctx, SessionContext)
        assert ctx.session_id == "s1"
        assert len(ctx.turns) == 2
        assert ctx.summary == "greeting exchange"

    def test_get_context_empty_session(self):
        store = InMemoryStore(session_ttl=3600)
        ctx = store.get_context("nonexistent")
        assert ctx.session_id == "nonexistent"
        assert ctx.turns == ()
        assert ctx.summary is None
