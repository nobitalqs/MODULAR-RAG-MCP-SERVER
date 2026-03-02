"""Splitter abstract base class and validation.

Defines the pluggable interface for text splitting strategies.
All concrete implementations (Recursive, Semantic, FixedLength) must
inherit from ``BaseSplitter`` and implement the ``split_text`` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """Abstract base class for text splitters.

    Subclasses must implement :meth:`split_text`. The base class provides
    :meth:`validate_text` for input validation and :meth:`validate_chunks`
    for output validation.
    """

    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Any = None,
        **kwargs: Any,
    ) -> list[str]:
        """Split input text into a list of chunks.

        Args:
            text: Non-empty string to split.
            trace: Optional TraceContext for observability.
            **kwargs: Strategy-specific overrides (chunk_size, overlap, etc.).

        Returns:
            Ordered list of text chunks preserving original sequence.
        """

    def validate_text(self, text: str) -> None:
        """Validate input text before splitting.

        Args:
            text: Text to validate.

        Raises:
            ValueError: If text is not a non-empty string.
        """
        if not isinstance(text, str):
            raise ValueError(
                f"Input text must be a string, got {type(text).__name__}"
            )
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only")

    def validate_chunks(self, chunks: list[str]) -> None:
        """Validate output chunks after splitting.

        Args:
            chunks: List of chunk strings to validate.

        Raises:
            ValueError: If chunks list is invalid.
        """
        if not isinstance(chunks, list):
            raise ValueError("Chunks must be a list of strings")
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                raise ValueError(
                    f"Chunk at index {i} is not a string "
                    f"(type: {type(chunk).__name__})"
                )
            if not chunk.strip():
                raise ValueError(
                    f"Chunk at index {i} is empty or whitespace-only"
                )
