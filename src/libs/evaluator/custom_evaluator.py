"""Custom evaluator for lightweight, deterministic metrics.

Computes hit_rate and MRR without any external dependencies or LLM calls.
Designed for fast regression checks and sanity validation.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator for lightweight metrics (hit_rate, mrr).

    Expects retrieved chunks to contain an identifier field.
    Supported id fields: ``id``, ``chunk_id``, ``document_id``, ``doc_id``.
    """

    SUPPORTED_METRICS = frozenset({"hit_rate", "mrr"})
    _ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")

    def __init__(
        self,
        metrics: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = kwargs

        normalized = [
            str(m).strip().lower() for m in (metrics or [])
        ]
        if not normalized:
            normalized = ["hit_rate", "mrr"]

        unsupported = [m for m in normalized if m not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported custom metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )

        self.metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[Any],
        generated_answer: str | None = None,
        ground_truth: Any = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute requested metrics for retrieval results.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks or records.
            generated_answer: Optional generated answer (unused).
            ground_truth: Ground truth — ids (str/list/dict) or None.
            trace: Optional TraceContext (unused).

        Returns:
            Dictionary of metric name to float value.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        retrieved_ids = self._extract_ids(retrieved_chunks, "retrieved_chunks")
        ground_truth_ids = self._extract_ground_truth_ids(ground_truth)

        results: dict[str, float] = {}
        if "hit_rate" in self.metrics:
            results["hit_rate"] = self._compute_hit_rate(
                retrieved_ids, ground_truth_ids,
            )
        if "mrr" in self.metrics:
            results["mrr"] = self._compute_mrr(
                retrieved_ids, ground_truth_ids,
            )
        return results

    # ------------------------------------------------------------------
    # Ground truth parsing
    # ------------------------------------------------------------------

    def _extract_ground_truth_ids(self, ground_truth: Any) -> list[str]:
        """Extract ground truth ids from various input shapes."""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, dict):
            if "ids" in ground_truth and isinstance(ground_truth["ids"], list):
                return self._extract_ids(ground_truth["ids"], "ground_truth.ids")
            return self._extract_ids([ground_truth], "ground_truth")
        if isinstance(ground_truth, list):
            return self._extract_ids(ground_truth, "ground_truth")
        raise ValueError(
            f"Unsupported ground_truth type: {type(ground_truth).__name__}. "
            "Expected str, dict, list, or None."
        )

    # ------------------------------------------------------------------
    # ID extraction
    # ------------------------------------------------------------------

    def _extract_ids(self, items: Iterable[Any], label: str) -> list[str]:
        """Extract ids from a list of items (str, dict, or object)."""
        ids: list[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                ids.append(item)
                continue
            if isinstance(item, dict):
                for field in self._ID_FIELDS:
                    if field in item:
                        ids.append(str(item[field]))
                        break
                else:
                    raise ValueError(
                        f"Missing id field in {label}[{index}]. "
                        f"Expected one of {', '.join(self._ID_FIELDS)}"
                    )
                continue
            if hasattr(item, "id"):
                ids.append(str(getattr(item, "id")))
                continue
            raise ValueError(
                f"Unable to extract id from {label}[{index}] "
                f"of type {type(item).__name__}"
            )
        return ids

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_hit_rate(
        self,
        retrieved_ids: Sequence[str],
        ground_truth_ids: Sequence[str],
    ) -> float:
        """Compute hit rate (binary: 1.0 if any relevant doc retrieved)."""
        if not ground_truth_ids:
            return 0.0
        gt_set = set(ground_truth_ids)
        return 1.0 if any(rid in gt_set for rid in retrieved_ids) else 0.0

    def _compute_mrr(
        self,
        retrieved_ids: Sequence[str],
        ground_truth_ids: Sequence[str],
    ) -> float:
        """Compute Mean Reciprocal Rank (1/rank of first relevant result)."""
        if not ground_truth_ids:
            return 0.0
        gt_set = set(ground_truth_ids)
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in gt_set:
                return 1.0 / rank
        return 0.0
