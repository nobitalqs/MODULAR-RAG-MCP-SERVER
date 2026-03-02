"""DeepSeek LLM provider implementation.

DeepSeek provides an OpenAI-compatible API, so this implementation
reuses the same request/response patterns as OpenAI.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class DeepSeekLLMError(RuntimeError):
    """Raised when the DeepSeek API returns an error."""


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM provider (OpenAI-compatible API).

    Args:
        model: Model identifier (e.g., "deepseek-chat", "deepseek-coder").
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens in the response.
        api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
        base_url: Base URL for the API endpoint.
        **kwargs: Additional arguments (ignored).

    Raises:
        ValueError: If api_key is not provided and DEEPSEEK_API_KEY is not set.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url or "https://api.deepseek.com/v1"

        if not self.api_key:
            raise ValueError(
                "[DeepSeek] Missing API key. Set DEEPSEEK_API_KEY env var "
                "or pass api_key parameter."
            )

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send chat messages to DeepSeek and return the response.

        Args:
            messages: Conversation messages.
            trace: Optional TraceContext for observability.
            **kwargs: Runtime overrides (temperature, max_tokens, etc.).

        Returns:
            ChatResponse with generated text.

        Raises:
            DeepSeekLLMError: If the API call fails.
        """
        self.validate_messages(messages)

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response_data = self._call_api(payload)
        except Exception as exc:
            # Re-raise with provider-specific error, never leak api_key
            error_msg = str(exc).replace(self.api_key, "[REDACTED]") if self.api_key else str(exc)
            raise DeepSeekLLMError(f"[DeepSeek] API call failed: {error_msg}") from exc

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
            raise DeepSeekLLMError(
                f"[DeepSeek] Unexpected response format: {exc}"
            ) from exc

    def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Internal method to call the DeepSeek API. Separated for test mocking.

        Args:
            payload: JSON payload for the API request.

        Returns:
            Parsed JSON response.

        Raises:
            httpx.TimeoutException: If the request times out.
            httpx.ConnectError: If connection fails.
            httpx.HTTPStatusError: If the API returns an error status.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
