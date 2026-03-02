"""Ollama LLM provider implementation.

Ollama is a local LLM runtime that provides a REST API. Unlike cloud
providers, it requires no API key and uses a different request format.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class OllamaLLMError(RuntimeError):
    """Raised when the Ollama API returns an error."""


class OllamaLLM(BaseLLM):
    """Ollama local LLM provider.

    Supports any model available in your local Ollama installation.

    Args:
        model: Model identifier (e.g., "llama2", "mistral", "codellama").
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens in the response (Ollama uses num_predict).
        base_url: Base URL for the Ollama API. Falls back to OLLAMA_BASE_URL
            env var, then "http://localhost:11434".
        **kwargs: Additional arguments (ignored).
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send chat messages to Ollama and return the response.

        Args:
            messages: Conversation messages.
            trace: Optional TraceContext for observability.
            **kwargs: Runtime overrides (temperature, max_tokens, etc.).

        Returns:
            ChatResponse with generated text.

        Raises:
            OllamaLLMError: If the API call fails.
        """
        self.validate_messages(messages)

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Ollama uses a different payload structure
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response_data = self._call_api(payload)
        except Exception as exc:
            raise OllamaLLMError(f"[Ollama] API call failed: {exc}") from exc

        try:
            # Ollama response format differs from OpenAI
            content = response_data["message"]["content"]

            # Ollama may provide usage stats in some versions
            usage_dict = None
            if "prompt_eval_count" in response_data or "eval_count" in response_data:
                usage_dict = {
                    "prompt_tokens": response_data.get("prompt_eval_count", 0),
                    "completion_tokens": response_data.get("eval_count", 0),
                    "total_tokens": (
                        response_data.get("prompt_eval_count", 0)
                        + response_data.get("eval_count", 0)
                    ),
                }

            return ChatResponse(
                content=content,
                model=response_data.get("model", self.model),
                usage=usage_dict,
                raw_response=response_data,
            )
        except (KeyError, TypeError) as exc:
            raise OllamaLLMError(
                f"[Ollama] Unexpected response format: {exc}"
            ) from exc

    def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Internal method to call the Ollama API. Separated for test mocking.

        Args:
            payload: JSON payload for the API request.

        Returns:
            Parsed JSON response.

        Raises:
            httpx.TimeoutException: If the request times out.
            httpx.ConnectError: If connection fails.
            httpx.HTTPStatusError: If the API returns an error status.
        """
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}

        # Longer timeout for local models (they may be slower than cloud APIs)
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
