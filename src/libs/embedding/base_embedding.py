"""Embedding abstract base class and validation.

Defines the pluggable interface for text embedding providers.
All concrete implementations (OpenAI, Azure, Ollama) must inherit
from ``BaseEmbedding`` and implement the ``embed`` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """Abstract base class for Embedding providers.

    Subclasses must implement :meth:`embed`. The base class provides
    :meth:`validate_texts` for input validation and :meth:`get_dimension`
    for reporting vector dimensionality.
    """

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: Non-empty list of text strings to embed.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific overrides.

        Returns:
            List of embedding vectors (one per input text).
        """

    def validate_texts(self, texts: list[str]) -> None:
        """Validate input text list before embedding.

        Args:
            texts: Texts to validate.

        Raises:
            ValueError: If the list is empty, an element is not a string,
                or a string is empty/whitespace-only.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Text at index {i} is not a string "
                    f"(type: {type(text).__name__})"
                )
            if not text.strip():
                raise ValueError(
                    f"Text at index {i} is empty or whitespace-only"
                )

    def get_dimension(self) -> int:
        """Return the dimensionality of produced embedding vectors.

        Subclasses should override this to return their specific dimension.

        Raises:
            NotImplementedError: If the subclass does not override.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_dimension()"
        )
