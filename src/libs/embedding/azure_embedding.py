"""Azure OpenAI Embedding provider.

Uses Azure OpenAI Service embeddings API to generate dense vectors
for text. Requires Azure-specific configuration including endpoint,
API version, and deployment name.
"""

from __future__ import annotations

import os
from typing import Any

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.resilience.retry import RetryableError, retry_with_backoff


class AzureEmbeddingError(RuntimeError, RetryableError):
    """Raised when Azure OpenAI embedding API call fails."""


class AzureEmbedding(BaseEmbedding):
    """Azure OpenAI Embedding provider.

    Uses Azure OpenAI's ``embeddings.create`` API. Requires
    AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables
    or explicit parameters.

    Args:
        model: Azure OpenAI model name.
        dimensions: Embedding vector dimension (default: 1536).
        api_key: Azure OpenAI API key. Falls back to AZURE_OPENAI_API_KEY.
        azure_endpoint: Azure endpoint URL. Falls back to AZURE_OPENAI_ENDPOINT.
        api_version: Azure API version (e.g., "2024-02-01").
        deployment_name: Azure deployment name (optional, defaults to model).
        **kwargs: Additional arguments (ignored).

    Raises:
        ValueError: If api_key or azure_endpoint is missing.
    """

    def __init__(
        self,
        model: str,
        dimensions: int = 1536,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        deployment_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.api_version = api_version
        self.deployment_name = deployment_name or model

        # Resolve API key and endpoint from parameters or environment
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key is required. "
                "Set AZURE_OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint is required. "
                "Set AZURE_OPENAI_ENDPOINT environment variable or pass azure_endpoint parameter."
            )

    def _create_client(self) -> Any:
        """Create Azure OpenAI client (lazy import).

        Returns:
            openai.AzureOpenAI client instance.
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK is not installed. "
                "Install with: pip install openai"
            ) from e

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "azure_endpoint": self.azure_endpoint,
        }
        if self.api_version:
            client_kwargs["api_version"] = self.api_version

        return openai.AzureOpenAI(**client_kwargs)

    @retry_with_backoff(max_retries=3, backoff_base=1.0)
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the Azure OpenAI embeddings API. Separated for retry and test mocking.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            AzureEmbeddingError: With status_code set for retryable HTTP errors.
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
                model=self.deployment_name,
                dimensions=self.dimensions,
            )
            return [item.embedding for item in response.data]
        except openai.APIStatusError as e:
            err = AzureEmbeddingError(f"Azure OpenAI embedding API call failed: {e}")
            err.status_code = e.status_code
            raise err from e
        except Exception as e:
            raise AzureEmbeddingError(
                f"Azure OpenAI embedding API call failed: {e}"
            ) from e

    def embed(
        self,
        texts: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings for texts using Azure OpenAI API.

        Args:
            texts: Non-empty list of text strings to embed.
            trace: Optional TraceContext for observability (unused).
            **kwargs: Provider-specific overrides (unused).

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            ValueError: If texts list is invalid.
            AzureEmbeddingError: If API call fails.
        """
        self.validate_texts(texts)
        return self._call_api(texts)

    def get_dimension(self) -> int:
        """Return embedding vector dimensionality.

        Returns:
            Configured dimension size.
        """
        return self.dimensions
