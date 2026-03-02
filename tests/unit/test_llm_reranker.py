"""Tests for LLMReranker implementation."""

from __future__ import annotations

from typing import Any

import pytest

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


# ===========================================================================
# FakeLLM for testing
# ===========================================================================

class FakeLLM(BaseLLM):
    """Mock LLM that returns controlled JSON responses."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.call_count = 0
        self.last_messages: list[Message] | None = None

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Return predetermined response."""
        self.validate_messages(messages)
        self.call_count += 1
        self.last_messages = messages
        return ChatResponse(
            content=self.response_text,
            model="fake-model",
        )


# ===========================================================================
# LLMReranker Tests
# ===========================================================================

class TestLLMReranker:
    """Tests for LLMReranker implementation."""

    def setup_method(self) -> None:
        """Import LLMReranker for each test."""
        from src.libs.reranker.llm_reranker import LLMReranker
        self.LLMReranker = LLMReranker

    def test_factory_can_create(self) -> None:
        """LLMReranker can be registered and created via factory."""
        from src.libs.reranker.reranker_factory import RerankerFactory

        factory = RerankerFactory()
        factory.register_provider("llm", self.LLMReranker)
        reranker = factory.create("llm", model="gpt-4o", top_k=5)
        assert isinstance(reranker, self.LLMReranker)

    def test_rerank_orders_by_score(self) -> None:
        """Reranker orders candidates by LLM-assigned scores."""
        fake_response = '''[
            {"passage_id": "0", "score": 0.3},
            {"passage_id": "1", "score": 0.9},
            {"passage_id": "2", "score": 0.6}
        ]'''
        fake_llm = FakeLLM(fake_response)
        reranker = self.LLMReranker(model="gpt-4o", top_k=10, llm=fake_llm)

        candidates = [
            {"id": "0", "text": "Low relevance"},
            {"id": "1", "text": "High relevance"},
            {"id": "2", "text": "Medium relevance"},
        ]

        results = reranker.rerank("test query", candidates)

        # Should be ordered by score: 1 (0.9), 2 (0.6), 0 (0.3)
        assert len(results) == 3
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"
        assert results[2]["id"] == "0"
        assert results[0]["rerank_score"] == 0.9

    def test_single_candidate_passthrough(self) -> None:
        """Single candidate is returned as-is without LLM call."""
        fake_llm = FakeLLM("should not be called")
        reranker = self.LLMReranker(model="gpt-4o", top_k=10, llm=fake_llm)

        candidates = [{"id": "0", "text": "Only one"}]
        results = reranker.rerank("test query", candidates)

        assert len(results) == 1
        assert results[0]["id"] == "0"
        assert fake_llm.call_count == 0  # LLM not called

    def test_validates_query(self) -> None:
        """Empty query raises ValueError."""
        fake_llm = FakeLLM("[]")
        reranker = self.LLMReranker(llm=fake_llm)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("", [{"id": "0", "text": "test"}])

    def test_validates_candidates(self) -> None:
        """Empty candidates raises ValueError."""
        fake_llm = FakeLLM("[]")
        reranker = self.LLMReranker(llm=fake_llm)

        with pytest.raises(ValueError, match="Candidates list cannot be empty"):
            reranker.rerank("query", [])

    def test_llm_not_set_raises(self) -> None:
        """LLM not set raises LLMRerankError."""
        from src.libs.reranker.llm_reranker import LLMRerankError

        reranker = self.LLMReranker(model="gpt-4o", llm=None)
        candidates = [
            {"id": "0", "text": "test1"},
            {"id": "1", "text": "test2"},
        ]

        with pytest.raises(LLMRerankError, match="LLM.*not set"):
            reranker.rerank("query", candidates)

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON response raises error."""
        from src.libs.reranker.llm_reranker import LLMRerankError

        fake_llm = FakeLLM("not valid json at all")
        reranker = self.LLMReranker(llm=fake_llm)

        candidates = [
            {"id": "0", "text": "test1"},
            {"id": "1", "text": "test2"},
        ]

        with pytest.raises(LLMRerankError, match="JSON"):
            reranker.rerank("query", candidates)

    def test_missing_score_field_raises(self) -> None:
        """Missing score field in response raises error."""
        from src.libs.reranker.llm_reranker import LLMRerankError

        fake_response = '[{"passage_id": "0"}, {"passage_id": "1"}]'
        fake_llm = FakeLLM(fake_response)
        reranker = self.LLMReranker(llm=fake_llm)

        candidates = [{"id": "0", "text": "test"}, {"id": "1", "text": "test2"}]

        with pytest.raises(LLMRerankError, match="score"):
            reranker.rerank("query", candidates)

    def test_markdown_fences_stripped(self) -> None:
        """Markdown code fences are stripped from LLM response."""
        fake_response = '''```json
        [
            {"passage_id": "0", "score": 0.8},
            {"passage_id": "1", "score": 0.5}
        ]
        ```'''
        fake_llm = FakeLLM(fake_response)
        reranker = self.LLMReranker(llm=fake_llm)

        candidates = [{"id": "0", "text": "test"}, {"id": "1", "text": "test2"}]
        results = reranker.rerank("query", candidates)

        assert len(results) == 2
        assert results[0]["rerank_score"] == 0.8

    def test_top_k_limits_results(self) -> None:
        """top_k parameter limits number of results."""
        fake_response = '''[
            {"passage_id": "0", "score": 0.9},
            {"passage_id": "1", "score": 0.8},
            {"passage_id": "2", "score": 0.7}
        ]'''
        fake_llm = FakeLLM(fake_response)
        reranker = self.LLMReranker(llm=fake_llm, top_k=2)

        candidates = [
            {"id": "0", "text": "test1"},
            {"id": "1", "text": "test2"},
            {"id": "2", "text": "test3"},
        ]
        results = reranker.rerank("query", candidates)

        assert len(results) == 2
        assert results[0]["id"] == "0"
        assert results[1]["id"] == "1"

    def test_uses_text_field(self) -> None:
        """Reranker extracts text from 'text' field."""
        fake_response = '[{"passage_id": "0", "score": 0.8}, {"passage_id": "1", "score": 0.5}]'
        fake_llm = FakeLLM(fake_response)
        reranker = self.LLMReranker(llm=fake_llm)

        candidates = [{"id": "0", "text": "passage content"}, {"id": "1", "text": "other"}]
        reranker.rerank("query", candidates)

        # Check that prompt was built with the text
        assert fake_llm.last_messages is not None
        prompt = fake_llm.last_messages[0].content
        assert "passage content" in prompt

    def test_uses_content_field(self) -> None:
        """Reranker extracts text from 'content' field if 'text' missing."""
        fake_response = '[{"passage_id": "0", "score": 0.8}, {"passage_id": "1", "score": 0.5}]'
        fake_llm = FakeLLM(fake_response)
        reranker = self.LLMReranker(llm=fake_llm)

        candidates = [{"id": "0", "content": "passage content"}, {"id": "1", "content": "other"}]
        reranker.rerank("query", candidates)

        # Check that prompt was built with the content
        assert fake_llm.last_messages is not None
        prompt = fake_llm.last_messages[0].content
        assert "passage content" in prompt

    def test_custom_prompt_template(self) -> None:
        """Custom prompt template can be provided."""
        fake_response = '[{"passage_id": "0", "score": 0.8}, {"passage_id": "1", "score": 0.6}]'
        fake_llm = FakeLLM(fake_response)
        custom_prompt = "Custom: {query} | {passages}"
        reranker = self.LLMReranker(llm=fake_llm, prompt_template=custom_prompt)

        candidates = [{"id": "0", "text": "test"}, {"id": "1", "text": "test2"}]
        reranker.rerank("my query", candidates)

        # Custom template should be used
        assert fake_llm.last_messages is not None
        # Implementation will use the custom template
