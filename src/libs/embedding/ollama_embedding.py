"""Ollama Embedding provider.

Uses local Ollama instance to generate embeddings via the /api/embed
HTTP endpoint. Does not require API keys and runs fully local.
"""

from __future__ import annotations

import os
from typing import Any

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.resilience.retry import RetryableError, retry_with_backoff


class OllamaEmbeddingError(RuntimeError, RetryableError):
    """Raised when Ollama embedding API call fails."""


class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding provider.

    Uses local Ollama server's ``/api/embed`` endpoint. No API key needed.
    Supports local models like "nomic-embed-text".

    Args:
        model: Ollama model name (default: "nomic-embed-text").
        dimensions: Embedding vector dimension (default: 768).
        base_url: Ollama server URL. Falls back to OLLAMA_BASE_URL env var
            or "http://localhost:11434".
        **kwargs: Additional arguments (ignored).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.dimensions = dimensions

        # Resolve base URL: parameter > env var > default
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )

    @retry_with_backoff(max_retries=3, backoff_base=1.0)
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama /api/embed endpoint.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            OllamaEmbeddingError: With status_code set for retryable HTTP errors.
            httpx.TimeoutException: If the request times out (also retried).
        """
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is not installed. "
                "Install with: pip install httpx"
            ) from e

        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["embeddings"]
        except httpx.HTTPStatusError as e:
            err = OllamaEmbeddingError(
                f"Ollama HTTP error {e.response.status_code}: {e.response.text}"
            )
            err.status_code = e.response.status_code
            raise err from e
        except httpx.RequestError as e:
            raise OllamaEmbeddingError(
                f"Ollama connection error: {e}"
            ) from e
        except KeyError as e:
            raise OllamaEmbeddingError(
                f"Ollama unexpected response format (missing 'embeddings' key)"
            ) from e
        except Exception as e:
            raise OllamaEmbeddingError(
                f"Ollama API call failed: {e}"
            ) from e

    def embed(
        self,
        texts: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings for texts using Ollama.

        Args:
            texts: Non-empty list of text strings to embed.
            trace: Optional TraceContext for observability (unused).
            **kwargs: Provider-specific overrides (unused).

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            ValueError: If texts list is invalid.
            OllamaEmbeddingError: If API call fails.
        """
        self.validate_texts(texts)
        return self._call_api(texts)

    def get_dimension(self) -> int:
        """Return embedding vector dimensionality.

        Returns:
            Configured dimension size.
        """
        return self.dimensions
