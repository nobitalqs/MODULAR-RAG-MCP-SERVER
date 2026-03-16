"""Maximal Marginal Relevance (MMR) for result diversification.

Selects a diverse subset from ranked candidates by balancing relevance
to the query against redundancy with already-selected results.

    MMR(d) = λ · rel(d) − (1−λ) · max_{s∈S} sim(d, s)

Uses term-frequency cosine similarity between chunk texts — no external
embedding calls required.  Operates as a post-rerank filter.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)

# Simple word tokenizer (same pattern as SparseEncoder)
_WORD_RE = re.compile(r"\b[\w-]+\b")


def mmr_select(
    results: list[RetrievalResult],
    top_k: int,
    lambda_: float = 0.7,
) -> list[RetrievalResult]:
    """Select top_k results using Maximal Marginal Relevance.

    Args:
        results: Score-sorted candidates (highest first).
        top_k: Number of results to select.
        lambda_: Balance between relevance (1.0) and diversity (0.0).

    Returns:
        Selected results in MMR order.
    """
    if not results or top_k <= 0:
        return []

    if len(results) <= top_k or lambda_ >= 1.0:
        return results[:top_k]

    # Precompute TF vectors and norms
    tf_vectors = [_term_freq(r.text) for r in results]
    norms = [_norm(v) for v in tf_vectors]

    # Normalize relevance scores to [0, 1]
    max_score = results[0].score
    min_score = results[-1].score
    score_range = max_score - min_score if max_score != min_score else 1.0

    selected_indices: list[int] = []
    remaining = set(range(len(results)))

    # First pick: always the highest-scoring result
    selected_indices.append(0)
    remaining.discard(0)

    # Iteratively select remaining
    while len(selected_indices) < top_k and remaining:
        best_idx = -1
        best_mmr = -math.inf

        for i in remaining:
            # Relevance term (normalized to [0, 1])
            rel = (results[i].score - min_score) / score_range

            # Redundancy term: max similarity to any selected result
            max_sim = max(
                _cosine_sim(tf_vectors[i], norms[i], tf_vectors[j], norms[j])
                for j in selected_indices
            )

            mmr_score = lambda_ * rel - (1.0 - lambda_) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [results[i] for i in selected_indices]


# ── text similarity helpers ─────────────────────────────────────────


def _term_freq(text: str) -> Counter[str]:
    """Build term frequency counter from text."""
    tokens = _WORD_RE.findall(text.lower())
    return Counter(tokens)


def _norm(tf: Counter[str]) -> float:
    """Compute L2 norm of a term frequency vector."""
    return math.sqrt(sum(v * v for v in tf.values())) or 1.0


def _cosine_sim(
    tf_a: Counter[str],
    norm_a: float,
    tf_b: Counter[str],
    norm_b: float,
) -> float:
    """Compute cosine similarity between two TF vectors."""
    # Dot product over shared terms
    if len(tf_a) < len(tf_b):
        smaller, larger = tf_a, tf_b
    else:
        smaller, larger = tf_b, tf_a

    dot = sum(count * larger.get(term, 0) for term, count in smaller.items())
    return dot / (norm_a * norm_b)
