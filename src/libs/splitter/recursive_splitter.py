"""Recursive Character Text Splitter implementation.

Uses langchain's RecursiveCharacterTextSplitter to split text on a hierarchy
of separators (paragraphs, then sentences, then words).
"""

from __future__ import annotations

from typing import Any

from src.libs.splitter.base_splitter import BaseSplitter

# Lazy import with fallback
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment, misc]


class RecursiveSplitter(BaseSplitter):
    """Recursive character text splitter.

    Splits text using a hierarchy of separators, trying paragraph boundaries
    first, then sentences, then words, then characters. Preserves semantic
    coherence better than fixed-length splitting.

    Args:
        chunk_size: Target size for each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        separators: List of separator strings to try in order. If None,
            uses default hierarchy: ["\\n\\n", "\\n", ". ", "! ", "? ",
            "; ", ", ", " ", ""].
        **kwargs: Additional arguments (ignored).

    Raises:
        ImportError: If langchain_text_splitters is not installed.
        ValueError: If chunk_size <= 0 or chunk_overlap < 0 or
            chunk_overlap >= chunk_size.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "[RecursiveSplitter] langchain_text_splitters is not installed. "
                "Install with: pip install langchain-text-splitters"
            )

        # Validate parameters
        if chunk_size <= 0:
            raise ValueError(
                f"[RecursiveSplitter] chunk_size must be > 0, got {chunk_size}"
            )
        if chunk_overlap < 0:
            raise ValueError(
                f"[RecursiveSplitter] chunk_overlap must be >= 0, got {chunk_overlap}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"[RecursiveSplitter] chunk_overlap ({chunk_overlap}) must be "
                f"< chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators: try paragraph, then newline, then sentences, etc.
        if separators is None:
            separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

        self.separators = separators

        # Create the underlying langchain splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def split_text(
        self,
        text: str,
        trace: Any = None,
        **kwargs: Any,
    ) -> list[str]:
        """Split text into chunks using recursive separator hierarchy.

        Args:
            text: Non-empty string to split.
            trace: Optional TraceContext for observability.
            **kwargs: Strategy-specific overrides (ignored).

        Returns:
            Ordered list of text chunks.

        Raises:
            ValueError: If text is empty or not a string.
        """
        self.validate_text(text)

        # Use langchain's recursive splitter
        chunks = self._splitter.split_text(text)

        self.validate_chunks(chunks)
        return chunks
