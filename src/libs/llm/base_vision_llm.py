"""Abstract base class for Vision LLM providers.

Defines the pluggable interface for Vision Language Model providers,
enabling multimodal interactions (text + image). All concrete Vision LLM
implementations must inherit from ``BaseVisionLLM`` and implement
the ``chat_with_image`` method.

Data types:
    ``ImageInput`` — validated container supporting path, bytes, or base64.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.libs.llm.base_llm import ChatResponse, Message


@dataclass
class ImageInput:
    """Image input for Vision LLM, supporting three mutually exclusive formats.

    Exactly one of ``path``, ``data``, or ``base64`` must be provided.

    Attributes:
        path: Path to a local image file.
        data: Raw image bytes (already loaded).
        base64: Base64-encoded image string.
        mime_type: MIME type of the image (default ``image/png``).
    """

    path: str | Path | None = None
    data: bytes | None = None
    base64: str | None = None
    mime_type: str = "image/png"

    def __post_init__(self) -> None:
        """Validate that exactly one input format is provided."""
        provided = sum(
            [
                self.path is not None,
                self.data is not None,
                self.base64 is not None,
            ]
        )
        if provided == 0:
            raise ValueError("Must provide one of: path, data, or base64")
        if provided > 1:
            raise ValueError("Must provide exactly one of: path, data, or base64")


class BaseVisionLLM(ABC):
    """Abstract base class for Vision LLM providers.

    Subclasses must implement :meth:`chat_with_image`. The base class
    provides validation helpers and a default pass-through preprocessor.
    """

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: list[Message] | None = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a response from text + image input.

        Args:
            text: Text prompt or question about the image.
            image: The image input (path, bytes, or base64).
            messages: Optional conversation history for context.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific overrides.

        Returns:
            ChatResponse with generated text.
        """

    def validate_text(self, text: str) -> None:
        """Validate text prompt.

        Raises:
            ValueError: If text is not a non-empty string.
        """
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text).__name__}")
        if not text or not text.strip():
            raise ValueError("Text prompt cannot be empty")

    def validate_image(self, image: ImageInput) -> None:
        """Validate image input.

        Raises:
            ValueError: If image is not an ImageInput instance.
        """
        if not isinstance(image, ImageInput):
            raise ValueError(f"Image must be an ImageInput instance, got {type(image).__name__}")

    def preprocess_image(
        self,
        image: ImageInput,
        max_size: tuple[int, int] | None = None,
    ) -> ImageInput:
        """Preprocess image before sending to Vision LLM.

        Default implementation returns the image unchanged.
        Subclasses may override for compression, resizing, etc.

        Args:
            image: The input image to preprocess.
            max_size: Optional maximum dimensions (width, height) in pixels.

        Returns:
            Preprocessed ImageInput (same instance by default).
        """
        return image
