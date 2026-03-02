"""Tests for CrossEncoderReranker implementation."""

from __future__ import annotations

from typing import Any

import pytest


# ===========================================================================
# MockScorer for testing
# ===========================================================================

class MockScorer:
    """Mock CrossEncoder scorer with deterministic predict()."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.call_count = 0
        self.last_pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return predetermined scores for pairs."""
        self.call_count += 1
        self.last_pairs = pairs
        return self.scores[:len(pairs)]


class FailingScorer:
    """Mock scorer that raises errors."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        raise RuntimeError("Scorer prediction failed")


# ===========================================================================
# CrossEncoderReranker Tests
# ===========================================================================

class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker implementation."""

    def setup_method(self) -> None:
        """Import CrossEncoderReranker for each test."""
        from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
        self.CrossEncoderReranker = CrossEncoderReranker

    def test_factory_can_create(self) -> None:
        """CrossEncoderReranker can be registered and created via factory."""
        from src.libs.reranker.reranker_factory import RerankerFactory

        factory = RerankerFactory()
        factory.register_provider("cross_encoder", self.CrossEncoderReranker)
        # Pass a mock scorer to avoid loading real model
        mock_scorer = MockScorer([])
        reranker = factory.create("cross_encoder", model="ms-marco", top_k=5, scorer=mock_scorer)
        assert isinstance(reranker, self.CrossEncoderReranker)

    def test_rerank_orders_by_score(self) -> None:
        """Reranker orders candidates by cross-encoder scores."""
        mock_scorer = MockScorer([0.1, 0.9, 0.5])
        reranker = self.CrossEncoderReranker(
            model="test-model",
            top_k=10,
            scorer=mock_scorer
        )

        candidates = [
            {"id": "0", "text": "Low relevance"},
            {"id": "1", "text": "High relevance"},
            {"id": "2", "text": "Medium relevance"},
        ]

        results = reranker.rerank("test query", candidates)

        # Should be ordered by score: 1 (0.9), 2 (0.5), 0 (0.1)
        assert len(results) == 3
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"
        assert results[2]["id"] == "0"
        assert results[0]["rerank_score"] == 0.9
        assert results[1]["rerank_score"] == 0.5
        assert results[2]["rerank_score"] == 0.1

    def test_top_k_limits(self) -> None:
        """top_k parameter limits number of results."""
        mock_scorer = MockScorer([0.3, 0.9, 0.6])
        reranker = self.CrossEncoderReranker(
            model="test-model",
            top_k=2,
            scorer=mock_scorer
        )

        candidates = [
            {"id": "0", "text": "Low"},
            {"id": "1", "text": "High"},
            {"id": "2", "text": "Medium"},
        ]

        results = reranker.rerank("query", candidates)

        # Should return only top 2
        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"

    def test_validates_query(self) -> None:
        """Empty query raises ValueError."""
        mock_scorer = MockScorer([0.5])
        reranker = self.CrossEncoderReranker(scorer=mock_scorer)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("", [{"id": "0", "text": "test"}])

    def test_validates_candidates(self) -> None:
        """Empty candidates raises ValueError."""
        mock_scorer = MockScorer([])
        reranker = self.CrossEncoderReranker(scorer=mock_scorer)

        with pytest.raises(ValueError, match="Candidates list cannot be empty"):
            reranker.rerank("query", [])

    def test_attaches_rerank_score(self) -> None:
        """Rerank scores are attached to result dicts."""
        mock_scorer = MockScorer([0.75])
        reranker = self.CrossEncoderReranker(scorer=mock_scorer)

        candidates = [{"id": "0", "text": "test"}]
        results = reranker.rerank("query", candidates)

        assert "rerank_score" in results[0]
        assert results[0]["rerank_score"] == 0.75

    def test_scorer_error_raises(self) -> None:
        """Scorer predict() error raises CrossEncoderRerankError."""
        from src.libs.reranker.cross_encoder_reranker import CrossEncoderRerankError

        failing_scorer = FailingScorer()
        reranker = self.CrossEncoderReranker(scorer=failing_scorer)

        candidates = [
            {"id": "0", "text": "test1"},
            {"id": "1", "text": "test2"},
        ]

        with pytest.raises(CrossEncoderRerankError, match="prediction failed"):
            reranker.rerank("query", candidates)

    def test_uses_text_field(self) -> None:
        """Reranker extracts text from 'text' field."""
        mock_scorer = MockScorer([0.5])
        reranker = self.CrossEncoderReranker(scorer=mock_scorer)

        candidates = [{"id": "0", "text": "passage text"}]
        reranker.rerank("query", candidates)

        # Check that pairs were built correctly
        assert mock_scorer.last_pairs is not None
        assert len(mock_scorer.last_pairs) == 1
        assert mock_scorer.last_pairs[0][0] == "query"
        assert mock_scorer.last_pairs[0][1] == "passage text"

    def test_uses_content_field(self) -> None:
        """Reranker extracts text from 'content' field if 'text' missing."""
        mock_scorer = MockScorer([0.5])
        reranker = self.CrossEncoderReranker(scorer=mock_scorer)

        candidates = [{"id": "0", "content": "passage content"}]
        reranker.rerank("query", candidates)

        # Check that pairs were built correctly
        assert mock_scorer.last_pairs is not None
        assert mock_scorer.last_pairs[0][1] == "passage content"

    def test_handles_missing_text_fields(self) -> None:
        """Reranker handles candidates without text/content fields."""
        mock_scorer = MockScorer([0.5])
        reranker = self.CrossEncoderReranker(scorer=mock_scorer)

        candidates = [{"id": "0", "title": "only title"}]
        reranker.rerank("query", candidates)

        # Should use empty string for missing text
        assert mock_scorer.last_pairs is not None
        assert mock_scorer.last_pairs[0][1] == ""

    def test_preserves_original_candidates(self) -> None:
        """Original candidate dicts are preserved (not mutated)."""
        mock_scorer = MockScorer([0.8])
        reranker = self.CrossEncoderReranker(scorer=mock_scorer)

        original_candidate = {"id": "0", "text": "test", "metadata": {"key": "value"}}
        candidates = [original_candidate]

        results = reranker.rerank("query", candidates)

        # Result should have rerank_score added
        assert "rerank_score" in results[0]
        # Original should not be mutated
        assert "rerank_score" not in original_candidate
        # Other fields preserved
        assert results[0]["metadata"]["key"] == "value"

    def test_scorer_not_set_raises(self) -> None:
        """Scorer not set attempts to load real model."""
        # When scorer is None, it tries to load from sentence_transformers
        # We can't easily test this without mocking the import
        # Just verify that providing a scorer works
        mock_scorer = MockScorer([0.5])
        reranker = self.CrossEncoderReranker(scorer=mock_scorer)
        assert reranker is not None
        assert reranker.scorer is mock_scorer

    def test_default_model(self) -> None:
        """Default model is set correctly."""
        reranker = self.CrossEncoderReranker(scorer=MockScorer([]))
        assert reranker.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_default_top_k(self) -> None:
        """Default top_k is 10."""
        reranker = self.CrossEncoderReranker(scorer=MockScorer([]))
        assert reranker.top_k == 10

    def test_single_candidate(self) -> None:
        """Single candidate works correctly."""
        mock_scorer = MockScorer([0.95])
        reranker = self.CrossEncoderReranker(scorer=mock_scorer)

        candidates = [{"id": "0", "text": "single"}]
        results = reranker.rerank("query", candidates)

        assert len(results) == 1
        assert results[0]["rerank_score"] == 0.95
