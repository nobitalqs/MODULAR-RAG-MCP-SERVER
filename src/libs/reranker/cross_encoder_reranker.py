"""Cross-Encoder Reranker implementation.

Uses a cross-encoder model from sentence-transformers to score query-passage
pairs for relevance. Cross-encoders jointly encode the query and passage,
providing more accurate relevance scores than bi-encoders.
"""

from __future__ import annotations

from typing import Any

from src.libs.reranker.base_reranker import BaseReranker

# Lazy import with fallback
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore[assignment, misc]


class CrossEncoderRerankError(RuntimeError):
    """Raised when cross-encoder reranking fails."""


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranker using sentence-transformers.

    Uses a pre-trained cross-encoder model to score query-passage pairs.
    Cross-encoders provide more accurate relevance scores than bi-encoders
    (dense embeddings) because they jointly encode both texts.

    Args:
        model: Model identifier (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").
        top_k: Maximum number of results to return after reranking.
        scorer: Optional CrossEncoder instance for testing. If None,
            creates one from sentence_transformers.
        **kwargs: Additional arguments (ignored).

    Raises:
        ImportError: If sentence_transformers is not installed.
        CrossEncoderRerankError: If scoring fails.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        scorer: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.top_k = top_k

        if scorer is not None:
            # Use injected scorer (for testing)
            self.scorer = scorer
        else:
            # Create real scorer
            if CrossEncoder is None:
                raise ImportError(
                    "[CrossEncoderReranker] sentence_transformers is not installed. "
                    "Install with: pip install sentence-transformers"
                )
            self.scorer = CrossEncoder(model)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using cross-encoder scoring.

        Args:
            query: User query string.
            candidates: List of candidate dicts to rerank.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters (ignored).

        Returns:
            Candidates sorted by cross-encoder score, limited to top_k.

        Raises:
            CrossEncoderRerankError: If scoring fails.
            ValueError: If query or candidates are invalid.
        """
        self.validate_query(query)
        self.validate_candidates(candidates)

        # Prepare query-passage pairs
        pairs = []
        for candidate in candidates:
            # Extract text from 'text' or 'content' field
            text = candidate.get("text") or candidate.get("content") or ""
            pairs.append((query, text))

        # Score pairs using cross-encoder
        try:
            scores = self.scorer.predict(pairs)
        except Exception as e:
            raise CrossEncoderRerankError(
                f"[CrossEncoderReranker] Cross-encoder prediction failed: {e}"
            ) from e

        # Attach scores to candidates (make copies to avoid mutation)
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            new_candidate = dict(candidate)
            new_candidate["rerank_score"] = float(score)
            scored_candidates.append(new_candidate)

        # Sort by score descending and take top_k
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[: self.top_k]
