"""Azure OpenAI LLM provider implementation.

Uses Azure's OpenAI Service API with deployment-based routing
and api-key authentication (not Bearer token).
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.resilience.retry import RetryableError, retry_with_backoff


class AzureLLMError(RuntimeError, RetryableError):
    """Raised when the Azure OpenAI API returns an error."""


class AzureLLM(BaseLLM):
    """Azure OpenAI Service LLM provider.

    Requires deployment name and Azure-specific endpoint configuration.

    Args:
        model: Model identifier (e.g., "gpt-4o").
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens in the response.
        api_key: Azure OpenAI API key. Falls back to AZURE_OPENAI_API_KEY env var.
        azure_endpoint: Azure endpoint URL. Falls back to AZURE_OPENAI_ENDPOINT env var.
        api_version: Azure API version (e.g., "2024-02-15-preview").
        deployment_name: Azure deployment name for the model.
        **kwargs: Additional arguments (ignored).

    Raises:
        ValueError: If required fields (api_key, azure_endpoint, deployment_name) are missing.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        deployment_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.deployment_name = deployment_name

        if not self.api_key:
            raise ValueError(
                "[Azure] Missing API key. Set AZURE_OPENAI_API_KEY env var "
                "or pass api_key parameter."
            )
        if not self.azure_endpoint:
            raise ValueError(
                "[Azure] Missing endpoint. Set AZURE_OPENAI_ENDPOINT env var "
                "or pass azure_endpoint parameter."
            )
        if not self.deployment_name:
            raise ValueError("[Azure] Missing deployment_name parameter.")

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send chat messages to Azure OpenAI and return the response.

        Args:
            messages: Conversation messages.
            trace: Optional TraceContext for observability.
            **kwargs: Runtime overrides (temperature, max_tokens, etc.).

        Returns:
            ChatResponse with generated text.

        Raises:
            AzureLLMError: If the API call fails.
        """
        self.validate_messages(messages)

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response_data = self._call_api(payload)
        except Exception as exc:
            # Re-raise with provider-specific error, never leak api_key
            error_msg = str(exc).replace(self.api_key, "[REDACTED]") if self.api_key else str(exc)
            raise AzureLLMError(f"[Azure] API call failed: {error_msg}") from exc

        try:
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            usage_dict = (
                {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                if usage
                else None
            )

            return ChatResponse(
                content=content,
                model=response_data.get("model", self.model),
                usage=usage_dict,
                raw_response=response_data,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise AzureLLMError(f"[Azure] Unexpected response format: {exc}") from exc

    @retry_with_backoff(max_retries=3, backoff_base=1.0)
    def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Internal method to call the Azure OpenAI API. Separated for test mocking.

        Args:
            payload: JSON payload for the API request.

        Returns:
            Parsed JSON response.

        Raises:
            AzureLLMError: With status_code set for retryable HTTP errors.
            httpx.TimeoutException: If the request times out (also retried).
            httpx.ConnectError: If connection fails.
        """
        # Azure uses deployment-based URL structure
        url = (
            f"{self.azure_endpoint}/openai/deployments/{self.deployment_name}"
            f"/chat/completions?api-version={self.api_version}"
        )
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            err = AzureLLMError(str(exc))
            err.status_code = exc.response.status_code
            raise err from exc
