"""Tests for CohereReranker — Azure Cohere Rerank API provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCohereRerankerInit:
    """Constructor validation tests."""

    def test_raises_without_api_key(self):
        """Missing api_key and no env var raises ValueError."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="api_key"):
                CohereReranker(
                    model="Cohere-rerank-v4.0-fast",
                    api_key="",
                    azure_endpoint="https://example.com",
                )

    def test_raises_without_azure_endpoint(self):
        """Missing azure_endpoint and no env var raises ValueError."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="azure_endpoint"):
                CohereReranker(
                    model="Cohere-rerank-v4.0-fast",
                    api_key="test-key",
                    azure_endpoint="",
                )

    def test_builds_correct_url(self):
        """URL is constructed as {endpoint}/providers/cohere/v2/rerank."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker(
            model="Cohere-rerank-v4.0-fast",
            api_key="test-key",
            azure_endpoint="https://myresource.services.ai.azure.com",
        )
        assert reranker._url == (
            "https://myresource.services.ai.azure.com/providers/cohere/v2/rerank"
        )

    def test_deployment_name_defaults_to_model(self):
        """When deployment_name is not set, it defaults to model."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker(
            model="Cohere-rerank-v4.0-fast",
            api_key="test-key",
            azure_endpoint="https://example.com",
        )
        assert reranker.deployment_name == "Cohere-rerank-v4.0-fast"

    def test_reads_env_vars_as_fallback(self):
        """api_key and azure_endpoint fall back to env vars."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        env = {
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": "https://env-endpoint.azure.com",
        }
        with patch.dict("os.environ", env, clear=True):
            reranker = CohereReranker(model="test-model")
            assert reranker.api_key == "env-key"
            assert "env-endpoint.azure.com" in reranker._url

    def test_cohere_endpoint_env_takes_priority(self):
        """AZURE_COHERE_ENDPOINT takes priority over AZURE_OPENAI_ENDPOINT."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        env = {
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_COHERE_ENDPOINT": "https://cohere-specific.azure.com",
            "AZURE_OPENAI_ENDPOINT": "https://openai-generic.azure.com",
        }
        with patch.dict("os.environ", env, clear=True):
            reranker = CohereReranker(model="test-model")
            assert "cohere-specific.azure.com" in reranker._url


class TestCohereRerankerRerank:
    """Rerank method tests with mocked HTTP."""

    def _make_reranker(self):
        """Create a CohereReranker with test credentials."""
        from src.libs.reranker.cohere_reranker import CohereReranker

        return CohereReranker(
            model="Cohere-rerank-v4.0-fast",
            top_k=2,
            api_key="test-key",
            azure_endpoint="https://example.com",
        )

    def test_rerank_sends_correct_payload(self):
        """HTTP request contains model, query, documents, and top_n."""
        reranker = self._make_reranker()
        candidates = [
            {"id": "c1", "text": "Python is great"},
            {"id": "c2", "text": "Java is verbose"},
            {"id": "c3", "text": "Rust is fast"},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 2, "relevance_score": 0.80},
            ]
        }

        with patch(
            "src.libs.reranker.cohere_reranker.httpx.post", return_value=mock_response
        ) as mock_post:
            reranker.rerank("best programming language", candidates)

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["model"] == "Cohere-rerank-v4.0-fast"
            assert payload["query"] == "best programming language"
            assert payload["documents"] == [
                "Python is great",
                "Java is verbose",
                "Rust is fast",
            ]
            assert payload["top_n"] == 2

    def test_rerank_returns_scored_candidates(self):
        """Returned candidates have rerank_score and correct ordering."""
        reranker = self._make_reranker()
        candidates = [
            {"id": "c1", "text": "Python is great", "score": 0.5},
            {"id": "c2", "text": "Java is verbose", "score": 0.7},
            {"id": "c3", "text": "Rust is fast", "score": 0.3},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
                {"index": 1, "relevance_score": 0.10},
            ]
        }

        with patch("src.libs.reranker.cohere_reranker.httpx.post", return_value=mock_response):
            result = reranker.rerank("fast language", candidates)

        # top_k=2, so only 2 results
        assert len(result) == 2
        # First result should be highest score (index 2 = Rust)
        assert result[0]["id"] == "c3"
        assert result[0]["rerank_score"] == 0.95
        # Second result (index 0 = Python)
        assert result[1]["id"] == "c1"
        assert result[1]["rerank_score"] == 0.80

    def test_rerank_does_not_mutate_original_candidates(self):
        """Original candidate dicts are not modified."""
        reranker = self._make_reranker()
        candidates = [
            {"id": "c1", "text": "Hello"},
            {"id": "c2", "text": "World"},
        ]
        original_keys = [set(c.keys()) for c in candidates]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
            ]
        }

        with patch("src.libs.reranker.cohere_reranker.httpx.post", return_value=mock_response):
            reranker.rerank("test query", candidates)

        # Original candidates should be unchanged
        for c, orig_keys in zip(candidates, original_keys):
            assert set(c.keys()) == orig_keys

    def test_rerank_raises_on_http_error(self):
        """HTTP error raises CohereRerankError."""
        import httpx as httpx_mod

        from src.libs.reranker.cohere_reranker import CohereRerankError

        reranker = self._make_reranker()
        candidates = [{"id": "c1", "text": "Hello"}]

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
            "Rate limited",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("src.libs.reranker.cohere_reranker.httpx.post", return_value=mock_response):
            with pytest.raises(CohereRerankError, match="Cohere"):
                reranker.rerank("test", candidates)

    def test_rerank_validates_empty_query(self):
        """Empty query raises ValueError."""
        reranker = self._make_reranker()
        with pytest.raises(ValueError, match="empty"):
            reranker.rerank("", [{"id": "c1", "text": "Hello"}])

    def test_rerank_validates_empty_candidates(self):
        """Empty candidates list raises ValueError."""
        reranker = self._make_reranker()
        with pytest.raises(ValueError, match="empty"):
            reranker.rerank("test", [])

    def test_uses_content_field_fallback(self):
        """Falls back to 'content' field when 'text' is missing."""
        reranker = self._make_reranker()
        candidates = [{"id": "c1", "content": "Using content field"}]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.9}]}

        with patch(
            "src.libs.reranker.cohere_reranker.httpx.post", return_value=mock_response
        ) as mock_post:
            reranker.rerank("test", candidates)
            payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
            assert payload["documents"] == ["Using content field"]


class TestCohereRerankerFactory:
    """Factory integration tests."""

    def test_factory_can_create(self):
        """CohereReranker can be registered and created via factory."""
        from src.libs.reranker.cohere_reranker import CohereReranker
        from src.libs.reranker.reranker_factory import RerankerFactory

        factory = RerankerFactory()
        factory.register_provider("cohere", CohereReranker)
        reranker = factory.create(
            "cohere",
            model="Cohere-rerank-v4.0-fast",
            api_key="test-key",
            azure_endpoint="https://example.com",
        )
        assert isinstance(reranker, CohereReranker)

    def test_is_base_reranker_subclass(self):
        """CohereReranker is a proper BaseReranker subclass."""
        from src.libs.reranker.base_reranker import BaseReranker
        from src.libs.reranker.cohere_reranker import CohereReranker

        assert issubclass(CohereReranker, BaseReranker)
