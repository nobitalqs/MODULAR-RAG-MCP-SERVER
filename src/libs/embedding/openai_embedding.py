"""OpenAI Embedding provider.

Uses the OpenAI embeddings API to generate dense vectors for text.
Supports configurable models (e.g., text-embedding-3-small) and
dimensionality reduction.
"""

from __future__ import annotations

import os
from typing import Any

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.resilience.retry import RetryableError, retry_with_backoff


class OpenAIEmbeddingError(RuntimeError, RetryableError):
    """Raised when OpenAI embedding API call fails."""


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding provider.

    Uses OpenAI's ``embeddings.create`` API. Requires OPENAI_API_KEY
    environment variable or explicit api_key parameter.

    Args:
        model: OpenAI model name (e.g., "text-embedding-3-small").
        dimensions: Embedding vector dimension (default: 1536).
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        base_url: Optional custom base URL for OpenAI API.
        **kwargs: Additional arguments (ignored).

    Raises:
        ValueError: If api_key is None and OPENAI_API_KEY is not set.
    """

    def __init__(
        self,
        model: str,
        dimensions: int = 1536,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url

        # Resolve API key from parameter or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

    def _create_client(self) -> Any:
        """Create OpenAI client (lazy import).

        Returns:
            openai.OpenAI client instance.
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK is not installed. "
                "Install with: pip install openai"
            ) from e

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        return openai.OpenAI(**client_kwargs)

    @retry_with_backoff(max_retries=3, backoff_base=1.0)
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API. Separated for retry and test mocking.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            OpenAIEmbeddingError: With status_code set for retryable HTTP errors.
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK is not installed. "
                "Install with: pip install openai"
            ) from e

        client = self._create_client()

        try:
            response = client.embeddings.create(
                input=texts,
                model=self.model,
                dimensions=self.dimensions,
            )
            return [item.embedding for item in response.data]
        except openai.APIStatusError as e:
            err = OpenAIEmbeddingError(f"OpenAI embedding API call failed: {e}")
            err.status_code = e.status_code
            raise err from e
        except Exception as e:
            raise OpenAIEmbeddingError(
                f"OpenAI embedding API call failed: {e}"
            ) from e

    def embed(
        self,
        texts: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings for texts using OpenAI API.

        Args:
            texts: Non-empty list of text strings to embed.
            trace: Optional TraceContext for observability (unused).
            **kwargs: Provider-specific overrides (unused).

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            ValueError: If texts list is invalid.
            OpenAIEmbeddingError: If API call fails.
        """
        self.validate_texts(texts)
        return self._call_api(texts)

    def get_dimension(self) -> int:
        """Return embedding vector dimensionality.

        Returns:
            Configured dimension size.
        """
        return self.dimensions
