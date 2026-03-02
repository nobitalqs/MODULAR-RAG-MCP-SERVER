"""Azure OpenAI Vision LLM provider implementation.

Supports GPT-4o and GPT-4-Vision-Preview for multimodal interactions
(text + image). Uses httpx for HTTP calls with Azure-specific
deployment-based routing and api-key authentication.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import httpx

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class AzureVisionLLMError(RuntimeError):
    """Raised when the Azure Vision API call fails."""


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision LLM provider.

    Args:
        model: Model identifier (e.g., ``gpt-4o``).
        api_key: Azure API key. Falls back to ``AZURE_OPENAI_API_KEY`` env var.
        azure_endpoint: Azure endpoint URL. Falls back to ``AZURE_OPENAI_ENDPOINT``.
        deployment_name: Azure deployment name for the vision model.
        api_version: API version (default ``2024-02-15-preview``).
        max_image_size: Maximum image dimension in pixels (default 2048).
        temperature: Sampling temperature (default 0.7).
        max_tokens: Maximum tokens in response (default 1024).
        **kwargs: Additional configuration (ignored).

    Raises:
        ValueError: If api_key, azure_endpoint, or deployment_name is missing.
    """

    DEFAULT_API_VERSION = "2024-02-15-preview"
    DEFAULT_MAX_IMAGE_SIZE = 2048

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        deployment_name: str | None = None,
        api_version: str | None = None,
        max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_image_size = max_image_size
        self.api_version = api_version or self.DEFAULT_API_VERSION

        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "[Azure Vision] Missing API key. Set AZURE_OPENAI_API_KEY "
                "env var or pass api_key parameter."
            )

        self.endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not self.endpoint:
            raise ValueError(
                "[Azure Vision] Missing endpoint. Set AZURE_OPENAI_ENDPOINT "
                "env var or pass azure_endpoint parameter."
            )

        self.deployment_name = deployment_name
        if not self.deployment_name:
            raise ValueError("[Azure Vision] Missing deployment_name parameter.")

    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: list[Message] | None = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a response from text + image using Azure Vision API.

        Args:
            text: Text prompt or question about the image.
            image: Image input (path, bytes, or base64).
            messages: Optional conversation history.
            trace: Optional TraceContext for observability.
            **kwargs: Overrides (temperature, max_tokens).

        Returns:
            ChatResponse with generated text.

        Raises:
            ValueError: If text or image is invalid.
            AzureVisionLLMError: If the API call fails.
        """
        self.validate_text(text)
        self.validate_image(image)

        image_base64 = self._get_image_base64(image)

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Build API messages
        api_messages: list[dict[str, Any]] = []
        if messages:
            api_messages.extend(
                {"role": m.role, "content": m.content} for m in messages
            )

        # Append current text + image as multimodal user message
        api_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.mime_type};base64,{image_base64}",
                    },
                },
            ],
        })

        payload = {
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response_data = self._call_api(payload)
        except Exception as exc:
            if isinstance(exc, AzureVisionLLMError):
                raise
            error_msg = str(exc)
            if self.api_key:
                error_msg = error_msg.replace(self.api_key, "[REDACTED]")
            raise AzureVisionLLMError(
                f"[Azure Vision] API call failed: {error_msg}"
            ) from exc

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
            raise AzureVisionLLMError(
                f"[Azure Vision] Unexpected response format: {exc}"
            ) from exc

    def _get_image_base64(self, image: ImageInput) -> str:
        """Convert ImageInput to base64 string.

        Args:
            image: The image to convert.

        Returns:
            Base64-encoded string.

        Raises:
            AzureVisionLLMError: If encoding fails.
        """
        try:
            if image.base64:
                return image.base64
            if image.data:
                return base64.b64encode(image.data).decode("utf-8")
            if image.path:
                raw = Path(image.path).read_bytes()
                return base64.b64encode(raw).decode("utf-8")
            raise ValueError("ImageInput has no valid data source")
        except Exception as exc:
            if isinstance(exc, AzureVisionLLMError):
                raise
            raise AzureVisionLLMError(
                f"[Azure Vision] Failed to encode image: {exc}"
            ) from exc

    def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make HTTP request to Azure OpenAI Vision API. Separated for test mocking.

        Args:
            payload: JSON request body.

        Returns:
            Parsed JSON response.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
        """
        endpoint = self.endpoint.rstrip("/")
        url = (
            f"{endpoint}/openai/deployments/{self.deployment_name}"
            f"/chat/completions?api-version={self.api_version}"
        )
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
