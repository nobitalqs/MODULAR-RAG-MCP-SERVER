"""Unit tests for ConversationMemory — business logic layer."""

from unittest.mock import MagicMock

import pytest

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.memory.base_memory import ConversationTurn, SessionContext
from src.libs.memory.conversation_memory import ConversationMemory
from src.libs.memory.memory_store import InMemoryStore


@pytest.fixture
def store():
    return InMemoryStore(session_ttl=3600)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat.return_value = ChatResponse(
        content="Summary of conversation so far.",
        model="test-model",
    )
    return llm


class TestGetContext:
    def test_returns_session_context(self, store):
        mem = ConversationMemory(store=store, max_turns=10, summarize_threshold=5)
        store.add_turn("s1", ConversationTurn("user", "hi"))
        store.add_turn("s1", ConversationTurn("assistant", "hello"))
        ctx = mem.get_context("s1")
        assert isinstance(ctx, SessionContext)
        assert len(ctx.turns) == 2

    def test_sliding_window_limits_turns(self, store):
        mem = ConversationMemory(store=store, max_turns=3, summarize_threshold=10)
        for i in range(6):
            store.add_turn("s1", ConversationTurn("user", f"msg {i}"))
        ctx = mem.get_context("s1")
        # Should return only the last 3 turns
        assert len(ctx.turns) == 3
        assert ctx.turns[0].content == "msg 3"
        assert ctx.turns[2].content == "msg 5"

    def test_empty_session(self, store):
        mem = ConversationMemory(store=store, max_turns=10, summarize_threshold=5)
        ctx = mem.get_context("nonexistent")
        assert ctx.turns == []
        assert ctx.summary is None


class TestAddTurn:
    def test_add_turn_stores_in_underlying(self, store):
        mem = ConversationMemory(store=store, max_turns=10, summarize_threshold=5)
        mem.add_turn("s1", "user", "hello")
        turns = store.get_turns("s1")
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "hello"

    def test_add_turn_triggers_compress_when_over_threshold(self, store, mock_llm):
        mem = ConversationMemory(
            store=store,
            max_turns=10,
            summarize_threshold=3,
            summarize_enabled=True,
            llm=mock_llm,
        )
        mem.add_turn("s1", "user", "msg 1")
        mem.add_turn("s1", "assistant", "reply 1")
        mem.add_turn("s1", "user", "msg 2")
        # At threshold=3, the 3rd turn should trigger compression
        mock_llm.chat.assert_not_called()
        mem.add_turn("s1", "assistant", "reply 2")  # 4th turn > threshold
        mock_llm.chat.assert_called_once()
        # Summary should be set
        assert store.get_summary("s1") == "Summary of conversation so far."

    def test_no_compress_when_summarize_disabled(self, store, mock_llm):
        mem = ConversationMemory(
            store=store,
            max_turns=10,
            summarize_threshold=2,
            summarize_enabled=False,
            llm=mock_llm,
        )
        for i in range(5):
            mem.add_turn("s1", "user", f"msg {i}")
        mock_llm.chat.assert_not_called()

    def test_no_compress_when_llm_is_none(self, store):
        mem = ConversationMemory(
            store=store,
            max_turns=10,
            summarize_threshold=2,
            summarize_enabled=True,
            llm=None,
        )
        for i in range(5):
            mem.add_turn("s1", "user", f"msg {i}")
        # Should not raise — gracefully degrades to truncation only


class TestToMessages:
    def test_converts_turns_to_messages(self, store):
        mem = ConversationMemory(store=store, max_turns=10, summarize_threshold=5)
        store.add_turn("s1", ConversationTurn("user", "hi"))
        store.add_turn("s1", ConversationTurn("assistant", "hello"))
        messages = mem.to_messages("s1")
        assert len(messages) == 2
        assert all(isinstance(m, Message) for m in messages)
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_includes_summary_as_system_message(self, store):
        mem = ConversationMemory(store=store, max_turns=10, summarize_threshold=5)
        store.add_turn("s1", ConversationTurn("user", "hi"))
        store.set_summary("s1", "Previous context summary")
        messages = mem.to_messages("s1")
        assert messages[0].role == "system"
        assert "Previous context summary" in messages[0].content
        assert messages[1].role == "user"

    def test_empty_session_returns_empty(self, store):
        mem = ConversationMemory(store=store, max_turns=10, summarize_threshold=5)
        messages = mem.to_messages("nonexistent")
        assert messages == []

    def test_respects_max_turns_window(self, store):
        mem = ConversationMemory(store=store, max_turns=2, summarize_threshold=10)
        for i in range(5):
            store.add_turn("s1", ConversationTurn("user", f"msg {i}"))
        messages = mem.to_messages("s1")
        assert len(messages) == 2
        assert messages[0].content == "msg 3"


class TestCompress:
    def test_compress_calls_llm_with_turns(self, store, mock_llm):
        mem = ConversationMemory(
            store=store,
            max_turns=10,
            summarize_threshold=2,
            summarize_enabled=True,
            llm=mock_llm,
        )
        store.add_turn("s1", ConversationTurn("user", "What is RAG?"))
        store.add_turn("s1", ConversationTurn("assistant", "RAG stands for..."))
        store.add_turn("s1", ConversationTurn("user", "How does it work?"))
        # Manually trigger compress
        mem.add_turn("s1", "assistant", "It works by...")
        # LLM should receive messages containing conversation content
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        prompt_text = " ".join(m.content for m in messages)
        assert "RAG" in prompt_text

    def test_compress_handles_llm_failure(self, store, mock_llm):
        mock_llm.chat.side_effect = RuntimeError("LLM down")
        mem = ConversationMemory(
            store=store,
            max_turns=10,
            summarize_threshold=2,
            summarize_enabled=True,
            llm=mock_llm,
        )
        store.add_turn("s1", ConversationTurn("user", "msg1"))
        store.add_turn("s1", ConversationTurn("assistant", "reply1"))
        # Should not raise
        mem.add_turn("s1", "user", "msg2")
        # Summary should remain None
        assert store.get_summary("s1") is None
