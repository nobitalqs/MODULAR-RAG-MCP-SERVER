"""Core layer Reranker orchestrating libs.reranker backends with fallback support.

This module implements the CoreReranker class that:
1. Integrates with libs.reranker (LLM, CrossEncoder, None) via DI
2. Provides graceful fallback when backend fails or times out
3. Converts RetrievalResult to/from reranker input/output format
4. Supports TraceContext for observability

Design Principles:
- Pluggable: Uses DI to accept any BaseReranker backend
- Config-Driven: Reads rerank settings from settings.yaml
- Graceful Fallback: Returns original order on backend failure
- Observable: TraceContext integration for debugging
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """Raised when reranking fails and fallback is disabled."""


@dataclass
class RerankConfig:
    """Configuration for CoreReranker.

    Attributes:
        enabled: Whether reranking is enabled.
        top_k: Number of results to return after reranking.
        timeout: Timeout for reranker backend (seconds).
        fallback_on_error: Whether to return original order on error.
    """

    enabled: bool = True
    top_k: int = 5
    timeout: float = 30.0
    fallback_on_error: bool = True


@dataclass
class RerankResult:
    """Result of a rerank operation.

    Attributes:
        results: Reranked list of RetrievalResults.
        used_fallback: Whether fallback was used due to backend failure.
        fallback_reason: Reason for fallback (if applicable).
        reranker_type: Type of reranker used ('llm', 'cross_encoder', 'none').
        original_order: Original results before reranking (for debugging).
    """

    results: list[RetrievalResult] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: str | None = None
    reranker_type: str = "none"
    original_order: list[RetrievalResult] | None = None


class CoreReranker:
    """Core layer Reranker with fallback support.

    This class wraps libs.reranker implementations and provides:
    1. Type conversion between RetrievalResult and reranker dict format
    2. Graceful fallback when backend fails
    3. Configuration-driven backend selection
    4. TraceContext integration

    Example:
        >>> reranker = CoreReranker(settings, reranker=fake_reranker)
        >>> result = reranker.rerank("query", retrieval_results)
        >>> print(result.results)
    """

    def __init__(
        self,
        settings: Settings,
        reranker: BaseReranker | None = None,
        config: RerankConfig | None = None,
    ) -> None:
        """Initialize CoreReranker.

        Args:
            settings: Application settings containing rerank configuration.
            reranker: Optional reranker backend. If None, uses NoneReranker.
            config: Optional RerankConfig. If None, extracts from settings.
        """
        self.settings = settings

        # Extract config from settings or use provided
        self.config = config if config is not None else self._extract_config(settings)

        # Initialize reranker backend
        if reranker is not None:
            self._reranker = reranker
        elif not self.config.enabled:
            self._reranker = NoneReranker()
        else:
            # No reranker injected but enabled — fall back to NoneReranker
            logger.warning(
                "Reranking enabled but no reranker backend provided, using NoneReranker",
            )
            self._reranker = NoneReranker()

        self._reranker_type = self._get_reranker_type()

    def _extract_config(self, settings: Settings) -> RerankConfig:
        """Extract RerankConfig from settings.

        Args:
            settings: Application settings.

        Returns:
            RerankConfig with values from settings.
        """
        try:
            rerank_settings = settings.rerank
            return RerankConfig(
                enabled=(bool(rerank_settings.enabled) if rerank_settings else False),
                top_k=(
                    int(rerank_settings.top_k)
                    if rerank_settings and hasattr(rerank_settings, "top_k")
                    else 5
                ),
                timeout=(
                    float(getattr(rerank_settings, "timeout", 30.0)) if rerank_settings else 30.0
                ),
                fallback_on_error=True,
            )
        except AttributeError:
            logger.warning(
                "Missing rerank configuration, using defaults (disabled)",
            )
            return RerankConfig(enabled=False)

    def _get_reranker_type(self) -> str:
        """Get the type name of the current reranker backend.

        Returns:
            String identifier for the reranker type.
        """
        class_name = self._reranker.__class__.__name__
        if "Cohere" in class_name:
            return "cohere"
        elif "LLM" in class_name:
            return "llm"
        elif "CrossEncoder" in class_name:
            return "cross_encoder"
        elif "None" in class_name:
            return "none"
        else:
            return class_name.lower()

    def _results_to_candidates(
        self,
        results: list[RetrievalResult],
    ) -> list[dict[str, Any]]:
        """Convert RetrievalResults to reranker candidate format.

        Args:
            results: List of RetrievalResult objects.

        Returns:
            List of dicts suitable for reranker input.
        """
        return [
            {
                "id": result.chunk_id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata.copy(),
            }
            for result in results
        ]

    def _candidates_to_results(
        self,
        candidates: list[dict[str, Any]],
        original_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Convert reranked candidates back to RetrievalResults.

        Args:
            candidates: Reranked candidates from reranker.
            original_results: Original results for reference.

        Returns:
            List of RetrievalResult in reranked order.
        """
        id_to_original = {r.chunk_id: r for r in original_results}

        results = []
        for candidate in candidates:
            chunk_id = candidate["id"]
            rerank_score = candidate.get(
                "rerank_score",
                candidate.get("score", 0.0),
            )

            if chunk_id in id_to_original:
                original = id_to_original[chunk_id]
                results.append(
                    RetrievalResult(
                        chunk_id=original.chunk_id,
                        score=rerank_score,
                        text=original.text,
                        metadata={
                            **original.metadata,
                            "original_score": original.score,
                            "rerank_score": rerank_score,
                            "reranked": True,
                        },
                    )
                )
            else:
                results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        score=rerank_score,
                        text=candidate.get("text", ""),
                        metadata=candidate.get("metadata", {}),
                    )
                )

        return results

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> RerankResult:
        """Rerank retrieval results using configured backend.

        Args:
            query: The user query string.
            results: List of RetrievalResult objects to rerank.
            top_k: Number of results to return. If None, uses config.top_k.
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters passed to reranker backend.

        Returns:
            RerankResult containing reranked results and metadata.
        """
        effective_top_k = top_k if top_k is not None else self.config.top_k

        # Early return for empty or single results
        if not results:
            return RerankResult(
                results=[],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )

        if len(results) == 1:
            return RerankResult(
                results=results[:],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )

        # If reranking disabled, return top_k results in original order
        if not self.config.enabled or isinstance(self._reranker, NoneReranker):
            return RerankResult(
                results=results[:effective_top_k],
                used_fallback=False,
                reranker_type="none",
                original_order=results[:],
            )

        # Convert to reranker input format
        candidates = self._results_to_candidates(results)

        # Attempt reranking
        try:
            logger.debug(
                "Reranking %d candidates with %s",
                len(candidates),
                self._reranker_type,
            )
            _t0 = time.monotonic()
            reranked_candidates = self._reranker.rerank(
                query=query,
                candidates=candidates,
                trace=trace,
                **kwargs,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0

            # Convert back to RetrievalResult
            reranked_results = self._candidates_to_results(
                reranked_candidates,
                results,
            )

            final_results = reranked_results[:effective_top_k]

            logger.info(
                "Reranking complete: %d results returned",
                len(final_results),
            )

            if trace is not None:
                trace.record_stage(
                    "rerank",
                    {
                        "method": self._reranker_type,
                        "input_count": len(candidates),
                        "output_count": len(final_results),
                        "chunks": [
                            {
                                "chunk_id": r.chunk_id,
                                "score": round(r.score, 4),
                                "text": r.text or "",
                                "source": r.metadata.get(
                                    "source_path",
                                    r.metadata.get("source", ""),
                                ),
                            }
                            for r in final_results
                        ],
                    },
                    elapsed_ms=_elapsed,
                )

            return RerankResult(
                results=final_results,
                used_fallback=False,
                reranker_type=self._reranker_type,
                original_order=results[:],
            )

        except Exception as e:
            logger.warning("Reranking failed, using fallback: %s", e)

            if self.config.fallback_on_error:
                fallback_results = [
                    RetrievalResult(
                        chunk_id=result.chunk_id,
                        score=result.score,
                        text=result.text,
                        metadata={
                            **result.metadata,
                            "reranked": False,
                            "rerank_fallback": True,
                        },
                    )
                    for result in results[:effective_top_k]
                ]

                return RerankResult(
                    results=fallback_results,
                    used_fallback=True,
                    fallback_reason=str(e),
                    reranker_type=self._reranker_type,
                    original_order=results[:],
                )
            else:
                raise RerankError(
                    f"Reranking failed and fallback disabled: {e}",
                ) from e

    @property
    def reranker_type(self) -> str:
        """Get the type of the current reranker backend."""
        return self._reranker_type

    @property
    def is_enabled(self) -> bool:
        """Check if reranking is enabled."""
        return self.config.enabled and not isinstance(self._reranker, NoneReranker)


def create_core_reranker(
    settings: Settings,
    reranker: BaseReranker | None = None,
) -> CoreReranker:
    """Factory function to create a CoreReranker instance.

    When no reranker is injected, uses RerankerFactory to create one
    from settings. Registers all known providers automatically.

    Args:
        settings: Application settings.
        reranker: Optional reranker backend override.

    Returns:
        Configured CoreReranker instance.
    """
    if reranker is None and settings.rerank.enabled:
        provider = settings.rerank.provider.lower()
        if provider not in ("none", ""):
            try:
                factory = RerankerFactory()
                # Lazy imports to avoid circular deps and optional deps
                from src.libs.reranker.cohere_reranker import CohereReranker
                from src.libs.reranker.cross_encoder_reranker import (
                    CrossEncoderReranker,
                )
                from src.libs.reranker.llm_reranker import LLMReranker

                factory.register_provider("cross_encoder", CrossEncoderReranker)
                factory.register_provider("llm", LLMReranker)
                factory.register_provider("cohere", CohereReranker)
                reranker = factory.create_from_settings(settings.rerank)
            except Exception:
                logger.warning(
                    "Failed to create reranker from factory, falling back to NoneReranker",
                    exc_info=True,
                )

    return CoreReranker(settings=settings, reranker=reranker)
