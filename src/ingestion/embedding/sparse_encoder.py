"""Sparse Encoder: BM25 term statistics extraction from text chunks.

Extracts term-level statistics needed for BM25 indexing.  The actual
index construction is handled downstream by BM25Indexer (C11).

Output structure per chunk::

    {
        "chunk_id": str,
        "term_frequencies": dict[str, int],   # term → count
        "doc_length": int,                     # total terms
        "unique_terms": int                    # vocabulary size
    }

Design Principles:
    - Stateless: no internal state between encode() calls
    - Deterministic: same input → same output
    - Clear Contracts: output usable by BM25Indexer without transformation
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.core.types import Chunk

# Tokenizer: word-boundary tokens including hyphens and underscores
_TOKEN_RE = re.compile(r"\b[\w-]+\b")


class SparseEncoder:
    """Encode text chunks into BM25 term statistics.

    Args:
        min_term_length: Minimum character length for a term (default 2).
        lowercase: Whether to lowercase terms (default True).

    Raises:
        ValueError: If *min_term_length* < 1.
    """

    def __init__(
        self,
        min_term_length: int = 2,
        lowercase: bool = True,
    ) -> None:
        if min_term_length < 1:
            raise ValueError(f"min_term_length must be >= 1, got {min_term_length}")

        self.min_term_length = min_term_length
        self.lowercase = lowercase

    # ── public interface ──────────────────────────────────────────────

    def encode(
        self,
        chunks: list[Chunk],
        trace: Any = None,
    ) -> list[dict[str, Any]]:
        """Encode chunks into BM25 term statistics.

        Args:
            chunks: Non-empty list of Chunk objects.
            trace: Optional TraceContext (reserved for Stage F).

        Returns:
            List of statistics dicts, one per chunk in the same order.

        Raises:
            ValueError: If *chunks* is empty or contains blank text.
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")

        results: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            if not chunk.text or not chunk.text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunk.id}) "
                    f"has empty or whitespace-only text"
                )

            terms = self._tokenize(chunk.text)
            tf = Counter(terms)

            results.append({
                "chunk_id": chunk.id,
                "term_frequencies": dict(tf),
                "doc_length": len(terms),
                "unique_terms": len(tf),
            })

        return results

    def get_corpus_stats(
        self,
        encoded_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate corpus-level statistics from encoded chunks.

        Computes average document length and document frequency (DF)
        for each term — inputs needed by BM25Indexer's IDF calculation.

        Args:
            encoded_chunks: Output of :meth:`encode`.

        Returns:
            ``{"num_docs", "avg_doc_length", "document_frequency"}``.
        """
        if not encoded_chunks:
            return {"num_docs": 0, "avg_doc_length": 0.0, "document_frequency": {}}

        num_docs = len(encoded_chunks)
        total_length = sum(c["doc_length"] for c in encoded_chunks)
        avg_doc_length = total_length / num_docs

        doc_freq: dict[str, int] = {}
        for chunk_stats in encoded_chunks:
            for term in chunk_stats["term_frequencies"]:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        return {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "document_frequency": doc_freq,
        }

    # ── tokenizer ─────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Split text into terms via regex, optionally lowercase, filter short."""
        tokens = _TOKEN_RE.findall(text)

        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        return [t for t in tokens if len(t) >= self.min_term_length]
