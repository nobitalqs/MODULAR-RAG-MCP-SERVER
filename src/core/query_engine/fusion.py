"""Reciprocal Rank Fusion (RRF) for combining multiple retrieval results.

This module implements the RRF fusion algorithm that combines ranking lists from
Dense and Sparse retrievers into a unified ranking. RRF is a simple yet effective
rank aggregation method that doesn't require score normalization.

Reference:
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
    SIGIR '09.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class RRFFusion:
    """Reciprocal Rank Fusion (RRF) for combining multiple ranking lists.

    RRF combines rankings from multiple sources using the formula::

        RRF_score(d) = Σ 1 / (k + rank(d))

    where:
        - d is a document (chunk)
        - k is a smoothing constant (typically 60)
        - rank(d) is the 1-based rank of document d in a ranking list

    Key Properties:
    - Deterministic: Same inputs always produce same output ordering
    - Score-agnostic: Uses only rank positions, not raw scores
    - No normalization needed: Works with heterogeneous score scales
    - Handles missing documents: Documents in only one list still contribute

    Attributes:
        k: Smoothing constant for RRF formula (default: 60).

    Example:
        >>> fusion = RRFFusion(k=60)
        >>> fused = fusion.fuse([dense_results, sparse_results], top_k=5)
    """

    # Default smoothing constant as recommended in the original RRF paper
    DEFAULT_K = 60

    def __init__(self, k: int = DEFAULT_K) -> None:
        """Initialize RRF fusion with configurable smoothing constant.

        Args:
            k: Smoothing constant for RRF formula (default: 60).
               Must be a positive integer.

        Raises:
            ValueError: If k is not a positive integer.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        self.k = k
        logger.info("RRFFusion initialized with k=%d", k)

    def fuse(
        self,
        ranking_lists: List[List[RetrievalResult]],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Fuse multiple ranking lists using Reciprocal Rank Fusion.

        Args:
            ranking_lists: List of ranking lists, each containing RetrievalResult
                objects sorted by relevance (descending).
                Typically ``[dense_results, sparse_results]``.
            top_k: Maximum number of results to return. If None, returns all.
            trace: Optional TraceContext for observability (reserved for Stage F).

        Returns:
            List of RetrievalResult objects, sorted by fused RRF score (descending).
            The score field contains the RRF score, not the original retrieval score.
            Text and metadata are preserved from the first occurrence of each chunk.

        Raises:
            ValueError: If ranking_lists is empty.
        """
        if not ranking_lists:
            raise ValueError("ranking_lists cannot be empty")

        # Filter out empty lists
        non_empty_lists = [lst for lst in ranking_lists if lst]

        if not non_empty_lists:
            logger.debug("All ranking lists are empty, returning empty result")
            return []

        logger.debug(
            "Fusing %d ranking lists with sizes %s",
            len(non_empty_lists),
            [len(lst) for lst in non_empty_lists],
        )

        # Step 1: Calculate RRF scores for each unique chunk
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, RetrievalResult] = {}  # Preserve text/metadata

        for ranking_list in non_empty_lists:
            for rank, result in enumerate(ranking_list, start=1):
                chunk_id = result.chunk_id

                # Calculate RRF contribution: 1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank)

                # Accumulate scores
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    # Store first occurrence's data (text, metadata)
                    chunk_data[chunk_id] = result

                rrf_scores[chunk_id] += rrf_contribution

        logger.debug("Computed RRF scores for %d unique chunks", len(rrf_scores))

        # Step 2: Create fused results with RRF scores
        fused_results = [
            RetrievalResult(
                chunk_id=chunk_id,
                score=rrf_score_val,
                text=chunk_data[chunk_id].text,
                metadata=chunk_data[chunk_id].metadata.copy(),
            )
            for chunk_id, rrf_score_val in rrf_scores.items()
        ]

        # Step 3: Sort by RRF score (descending), then by chunk_id for stability
        fused_results.sort(key=lambda r: (-r.score, r.chunk_id))

        # Step 4: Apply top_k limit if specified
        if top_k is not None and top_k > 0:
            fused_results = fused_results[:top_k]

        logger.debug(
            "Fusion complete: %d results (top_k=%s)",
            len(fused_results),
            top_k if top_k else "all",
        )

        return fused_results

    def fuse_with_weights(
        self,
        ranking_lists: List[List[RetrievalResult]],
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Fuse multiple ranking lists with optional per-list weights.

        Extended version of :meth:`fuse` that allows weighting different
        ranking sources. For example, giving more weight to dense retrieval
        for semantic queries.

        Args:
            ranking_lists: List of ranking lists.
            weights: Optional list of weights for each ranking list (default: uniform).
                Must have same length as ranking_lists if provided.
            top_k: Maximum number of results to return. If None, returns all.
            trace: Optional TraceContext for observability (reserved for Stage F).

        Returns:
            List of RetrievalResult objects, sorted by weighted RRF score (descending).

        Raises:
            ValueError: If ranking_lists is empty or weights length doesn't match.
        """
        if not ranking_lists:
            raise ValueError("ranking_lists cannot be empty")

        # Default to uniform weights
        if weights is None:
            weights = [1.0] * len(ranking_lists)

        if len(weights) != len(ranking_lists):
            raise ValueError(
                f"weights length ({len(weights)}) must match "
                f"ranking_lists length ({len(ranking_lists)})"
            )

        # Validate weights
        for i, w in enumerate(weights):
            if not isinstance(w, (int, float)) or w < 0:
                raise ValueError(
                    f"Weight at index {i} must be non-negative, got {w}"
                )

        # Filter out empty lists (keep their weights aligned)
        filtered = [(lst, w) for lst, w in zip(ranking_lists, weights) if lst]

        if not filtered:
            logger.debug("All ranking lists are empty, returning empty result")
            return []

        non_empty_lists, filtered_weights = zip(*filtered)

        logger.debug(
            "Fusing %d ranking lists with weights=%s",
            len(non_empty_lists),
            list(filtered_weights),
        )

        # Calculate weighted RRF scores
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, RetrievalResult] = {}

        for ranking_list, weight in zip(non_empty_lists, filtered_weights):
            for rank, result in enumerate(ranking_list, start=1):
                chunk_id = result.chunk_id

                # Weighted RRF contribution
                rrf_contribution = weight * (1.0 / (self.k + rank))

                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    chunk_data[chunk_id] = result

                rrf_scores[chunk_id] += rrf_contribution

        # Create and sort results
        fused_results = [
            RetrievalResult(
                chunk_id=chunk_id,
                score=rrf_score_val,
                text=chunk_data[chunk_id].text,
                metadata=chunk_data[chunk_id].metadata.copy(),
            )
            for chunk_id, rrf_score_val in rrf_scores.items()
        ]

        fused_results.sort(key=lambda r: (-r.score, r.chunk_id))

        if top_k is not None and top_k > 0:
            fused_results = fused_results[:top_k]

        return fused_results


def rrf_score(rank: int, k: int = RRFFusion.DEFAULT_K) -> float:
    """Calculate RRF score contribution for a single rank position.

    Utility function for calculating individual RRF contributions.

    Args:
        rank: 1-based rank position (1 = highest rank).
        k: Smoothing constant (default: 60).

    Returns:
        RRF score contribution: ``1 / (k + rank)``.

    Raises:
        ValueError: If rank is not a positive integer or k is not positive.

    Example:
        >>> rrf_score(1, k=60)  # Top-ranked document
        0.01639344262295082
    """
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"rank must be a positive integer, got {rank}")
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    return 1.0 / (k + rank)
