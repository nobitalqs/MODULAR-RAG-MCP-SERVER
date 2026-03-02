"""Tests for LLM abstraction: BaseLLM, Message, ChatResponse, LLMFactory."""

from __future__ import annotations

from typing import Any

import pytest

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


# ---------------------------------------------------------------------------
# Fake provider for testing
# ---------------------------------------------------------------------------

class FakeLLM(BaseLLM):
    """Minimal LLM stub that echoes the last user message."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.validate_messages(messages)
        last = messages[-1].content
        return ChatResponse(
            content=f"echo: {last}",
            model="fake-model",
            usage={"prompt_tokens": 5, "completion_tokens": 3},
        )


# ===========================================================================
# Message dataclass
# ===========================================================================

class TestMessage:
    """Tests for the Message dataclass."""

    def test_create_message(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_message_immutable(self) -> None:
        msg = Message(role="user", content="hello")
        with pytest.raises(AttributeError):
            msg.role = "system"  # type: ignore[misc]


# ===========================================================================
# ChatResponse dataclass
# ===========================================================================

class TestChatResponse:
    """Tests for the ChatResponse dataclass."""

    def test_create_response(self) -> None:
        resp = ChatResponse(content="hi", model="gpt-4o")
        assert resp.content == "hi"
        assert resp.model == "gpt-4o"
        assert resp.usage is None
        assert resp.raw_response is None

    def test_response_with_usage(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        resp = ChatResponse(content="hi", model="m", usage=usage)
        assert resp.usage == usage


# ===========================================================================
# BaseLLM.validate_messages
# ===========================================================================

class TestBaseLLMValidation:
    """Tests for BaseLLM.validate_messages."""

    def setup_method(self) -> None:
        self.llm = FakeLLM()

    def test_valid_messages(self) -> None:
        msgs = [Message(role="user", content="hello")]
        self.llm.validate_messages(msgs)  # should not raise

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one message"):
            self.llm.validate_messages([])

    def test_invalid_role_raises(self) -> None:
        msgs = [Message(role="invalid_role", content="x")]
        with pytest.raises(ValueError, match="Invalid role"):
            self.llm.validate_messages(msgs)

    def test_empty_content_raises(self) -> None:
        msgs = [Message(role="user", content="")]
        with pytest.raises(ValueError, match="empty content"):
            self.llm.validate_messages(msgs)

    def test_whitespace_content_raises(self) -> None:
        msgs = [Message(role="user", content="   ")]
        with pytest.raises(ValueError, match="empty content"):
            self.llm.validate_messages(msgs)

    def test_non_message_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected Message"):
            self.llm.validate_messages([{"role": "user", "content": "hi"}])  # type: ignore[list-item]

    def test_all_valid_roles(self) -> None:
        for role in ("system", "user", "assistant"):
            msgs = [Message(role=role, content="ok")]
            self.llm.validate_messages(msgs)  # should not raise


# ===========================================================================
# FakeLLM chat
# ===========================================================================

class TestFakeLLMChat:
    """Tests for FakeLLM as a BaseLLM implementation."""

    def test_chat_returns_response(self) -> None:
        llm = FakeLLM()
        msgs = [Message(role="user", content="ping")]
        resp = llm.chat(msgs)
        assert isinstance(resp, ChatResponse)
        assert resp.content == "echo: ping"
        assert resp.model == "fake-model"

    def test_chat_validates_messages(self) -> None:
        llm = FakeLLM()
        with pytest.raises(ValueError):
            llm.chat([])


# ===========================================================================
# LLMFactory
# ===========================================================================

class TestLLMFactory:
    """Tests for the LLM factory routing logic."""

    def setup_method(self) -> None:
        from src.libs.llm.llm_factory import LLMFactory

        self.factory = LLMFactory()

    def test_register_and_create(self) -> None:
        self.factory.register_provider("fake", FakeLLM)
        llm = self.factory.create("fake")
        assert isinstance(llm, FakeLLM)

    def test_case_insensitive_registration(self) -> None:
        self.factory.register_provider("FaKe", FakeLLM)
        llm = self.factory.create("fake")
        assert isinstance(llm, FakeLLM)

    def test_case_insensitive_create(self) -> None:
        self.factory.register_provider("fake", FakeLLM)
        llm = self.factory.create("FAKE")
        assert isinstance(llm, FakeLLM)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            self.factory.create("nonexistent")

    def test_error_lists_available_providers(self) -> None:
        self.factory.register_provider("fake", FakeLLM)
        with pytest.raises(ValueError, match="fake"):
            self.factory.create("missing")

    def test_register_non_subclass_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a subclass"):
            self.factory.register_provider("bad", dict)  # type: ignore[arg-type]

    def test_list_providers(self) -> None:
        self.factory.register_provider("alpha", FakeLLM)
        self.factory.register_provider("beta", FakeLLM)
        providers = self.factory.list_providers()
        assert "alpha" in providers
        assert "beta" in providers

    def test_create_with_kwargs(self) -> None:
        """Factory should forward kwargs to constructor."""

        class ConfigurableFake(BaseLLM):
            def __init__(self, **kwargs: Any) -> None:
                self.extra = kwargs

            def chat(
                self,
                messages: list[Message],
                trace: Any = None,
                **kwargs: Any,
            ) -> ChatResponse:
                return ChatResponse(content="ok", model="m")

        self.factory.register_provider("cfgfake", ConfigurableFake)
        llm = self.factory.create("cfgfake", model="gpt-4o", temperature=0.5)
        assert llm.extra["model"] == "gpt-4o"
        assert llm.extra["temperature"] == 0.5

    def test_create_from_settings(self) -> None:
        """Factory should extract provider from LLMSettings and forward fields."""
        from src.core.settings import LLMSettings

        self.factory.register_provider("fake", FakeLLM)

        settings = LLMSettings(
            provider="fake",
            model="test-model",
            temperature=0.7,
            max_tokens=100,
        )
        llm = self.factory.create_from_settings(settings)
        assert isinstance(llm, FakeLLM)
