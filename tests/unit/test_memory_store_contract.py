"""Contract tests for BaseMemoryStore implementations.

Every memory store must satisfy the same behavioral contract.
Uses @pytest.mark.parametrize to run identical assertions on all providers.
"""

from __future__ import annotations

import pytest

from src.libs.memory.base_memory import (
    BaseMemoryStore,
    ConversationTurn,
    SessionContext,
)
from src.libs.memory.memory_store import InMemoryStore


def _make_memory_store() -> BaseMemoryStore:
    return InMemoryStore(session_ttl=3600)


ALL_PROVIDERS = [
    pytest.param(_make_memory_store, id="InMemoryStore"),
]


@pytest.mark.parametrize("factory", ALL_PROVIDERS)
class TestMemoryStoreContract:
    """Contract: all BaseMemoryStore implementations must satisfy these."""

    def test_get_turns_empty_for_new_session(self, factory):
        store = factory()
        assert store.get_turns("sess-new") == ()

    def test_add_turn_and_retrieve(self, factory):
        store = factory()
        turn = ConversationTurn(role="user", content="Hello")
        store.add_turn("sess-1", turn)
        turns = store.get_turns("sess-1")
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "Hello"

    def test_multiple_turns_preserve_order(self, factory):
        store = factory()
        store.add_turn("sess-1", ConversationTurn(role="user", content="Q1"))
        store.add_turn("sess-1", ConversationTurn(role="assistant", content="A1"))
        store.add_turn("sess-1", ConversationTurn(role="user", content="Q2"))
        turns = store.get_turns("sess-1")
        assert len(turns) == 3
        assert [t.content for t in turns] == ["Q1", "A1", "Q2"]

    def test_get_summary_returns_none_by_default(self, factory):
        store = factory()
        assert store.get_summary("sess-1") is None

    def test_set_and_get_summary(self, factory):
        store = factory()
        store.set_summary("sess-1", "This is a summary")
        assert store.get_summary("sess-1") == "This is a summary"

    def test_clear_removes_all_data(self, factory):
        store = factory()
        store.add_turn("sess-1", ConversationTurn(role="user", content="Hi"))
        store.set_summary("sess-1", "Summary")
        store.clear("sess-1")
        assert store.get_turns("sess-1") == ()
        assert store.get_summary("sess-1") is None

    def test_get_context_returns_session_context(self, factory):
        store = factory()
        store.add_turn("sess-1", ConversationTurn(role="user", content="Hi"))
        store.set_summary("sess-1", "Summary text")
        ctx = store.get_context("sess-1")
        assert isinstance(ctx, SessionContext)
        assert ctx.session_id == "sess-1"
        assert len(ctx.turns) == 1
        assert ctx.summary == "Summary text"

    def test_session_isolation(self, factory):
        store = factory()
        store.add_turn("sess-A", ConversationTurn(role="user", content="A"))
        store.add_turn("sess-B", ConversationTurn(role="user", content="B"))
        assert len(store.get_turns("sess-A")) == 1
        assert store.get_turns("sess-A")[0].content == "A"
        assert len(store.get_turns("sess-B")) == 1
        assert store.get_turns("sess-B")[0].content == "B"

    def test_isinstance_base_memory_store(self, factory):
        store = factory()
        assert isinstance(store, BaseMemoryStore)
