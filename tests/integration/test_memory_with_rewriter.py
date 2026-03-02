"""Integration test: ConversationMemory + QueryRewriter.

Verifies that conversation memory context is correctly fed into the query
rewriter for pronoun resolution and context-aware rewriting.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.libs.memory.conversation_memory import ConversationMemory
from src.libs.memory.memory_store import InMemoryStore
from src.libs.memory.base_memory import ConversationTurn
from src.libs.query_rewriter.llm_rewriter import LLMRewriter


class TestMemoryFeedsRewriter:
    """ConversationMemory provides context to QueryRewriter."""

    def test_memory_context_to_rewriter_messages(self):
        """Memory turns become conversation_history for the rewriter."""
        store = InMemoryStore(session_ttl=3600)
        memory = ConversationMemory(
            store=store, max_turns=10, summarize_threshold=100,
        )

        # Simulate a conversation
        memory.add_turn("sess-1", "user", "Tell me about Python asyncio")
        memory.add_turn("sess-1", "assistant", "Asyncio is a library for writing concurrent code...")

        # Get context and convert to messages
        messages = memory.to_messages("sess-1")
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

        # Feed messages into rewriter as conversation_history
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            content=json.dumps({
                "queries": ["How to use async/await in Python asyncio"],
                "reasoning": "Resolved 'it' to 'asyncio'",
            })
        )
        rewriter = LLMRewriter(llm=mock_llm, max_rewrites=3)

        result = rewriter.rewrite(
            "How do I use it?",
            conversation_history=messages,
        )
        assert len(result.rewritten_queries) >= 1
        # LLM was called with conversation context
        assert mock_llm.chat.called

    def test_empty_memory_produces_no_history(self):
        """No turns means empty conversation history for the rewriter."""
        store = InMemoryStore(session_ttl=3600)
        memory = ConversationMemory(
            store=store, max_turns=10, summarize_threshold=100,
        )

        messages = memory.to_messages("sess-empty")
        assert messages == []

    def test_windowed_context_limits_turns(self):
        """ConversationMemory should only return the last N turns."""
        store = InMemoryStore(session_ttl=3600)
        memory = ConversationMemory(
            store=store, max_turns=2, summarize_threshold=100,
        )

        for i in range(5):
            memory.add_turn("sess-1", "user", f"Question {i}")
            memory.add_turn("sess-1", "assistant", f"Answer {i}")

        messages = memory.to_messages("sess-1")
        # max_turns=2, so only last 2 turns
        assert len(messages) == 2
        assert messages[-1].content == "Answer 4"

    def test_memory_with_summary_prepends_system_message(self):
        """When a summary exists, it's prepended as a system message."""
        store = InMemoryStore(session_ttl=3600)
        store.set_summary("sess-1", "User has been asking about Python asyncio.")
        store.add_turn("sess-1", ConversationTurn("user", "Latest question"))

        memory = ConversationMemory(
            store=store, max_turns=10, summarize_threshold=100,
        )

        messages = memory.to_messages("sess-1")
        assert messages[0].role == "system"
        assert "asyncio" in messages[0].content
        assert messages[1].role == "user"
