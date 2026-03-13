"""Ragas-based evaluator for RAG quality assessment.

This evaluator wraps the Ragas framework to compute LLM-as-Judge metrics:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision: Are the retrieved chunks relevant and well-ordered?

Design Principles:
- Pluggable: Implements BaseEvaluator interface, swappable via factory.
- Config-Driven: LLM/Embedding backend read from settings.yaml.
- Graceful Degradation: Clear ImportError if ragas not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metric name constants
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}


def _import_ragas() -> None:
    """Validate that ragas is importable, raising a clear error if not."""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics.

    Ragas does NOT require ground-truth labels.  It uses an LLM to judge
    the quality of the generated answer against the retrieved context.

    Supported metrics:
        - faithfulness: Measures factual consistency with context.
        - answer_relevancy: Measures how relevant the answer is to the query.
        - context_precision: Measures relevance/ordering of retrieved chunks.

    Example::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            settings: Application settings (used to configure LLM backend).
            metrics: Metric names to compute. Defaults to all supported.
            **kwargs: Additional parameters (reserved).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested.
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # -- public API ----------------------------------------------------

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate RAG quality using Ragas LLM-as-Judge metrics.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks (dicts with 'text' key or strings).
            generated_answer: The generated answer text. Required for Ragas.
            ground_truth: Ignored by Ragas (not needed for LLM-as-Judge).
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            Dictionary mapping metric names to float scores (0.0 - 1.0).

        Raises:
            ValueError: If query/chunks are invalid or generated_answer is missing.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)

        try:
            result = self._run_ragas(query, contexts, generated_answer)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # -- private helpers -----------------------------------------------

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """Execute Ragas collections metrics and return normalised scores.

        Ragas 0.4+ collections metrics use per-metric ``score()`` instead of
        the legacy ``evaluate()`` pipeline.  Each metric has its own signature:
        - Faithfulness / ContextPrecision: (user_input, response, retrieved_contexts)
        - AnswerRelevancy: (user_input, response)
        """
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
            Faithfulness,
        )

        llm, embeddings = self._build_wrappers()

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            if metric_name == FAITHFULNESS:
                m = Faithfulness(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            elif metric_name == ANSWER_RELEVANCY:
                m = AnswerRelevancy(llm=llm, embeddings=embeddings)
                result = m.score(user_input=query, response=answer)
            elif metric_name == CONTEXT_PRECISION:
                m = ContextPrecisionWithoutReference(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            else:
                continue

            scores[metric_name] = float(result.value) if result.value is not None else 0.0

        return scores

    def _build_wrappers(self) -> tuple:
        """Build Ragas LLM and Embedding wrappers from project settings.

        Priority: evaluation.llm/embedding (dedicated) > settings.llm/embedding (global).
        Allows separate LLM config for evaluation metrics if needed.

        Returns:
            Tuple of (llm_wrapper, embeddings_wrapper).

        Raises:
            ValueError: If settings are missing or provider is unsupported.
        """
        from openai import AsyncAzureOpenAI, AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory

        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        eval_cfg = self.settings.evaluation

        # -- LLM: prefer evaluation.llm, fallback to global llm --
        if eval_cfg.llm is not None:
            provider = eval_cfg.llm.provider.lower()
            model = eval_cfg.llm.model
            api_key = eval_cfg.llm.api_key
            azure_endpoint = eval_cfg.llm.azure_endpoint
            api_version = eval_cfg.llm.api_version
        else:
            provider = self.settings.llm.provider.lower()
            model = self.settings.llm.model
            api_key = self.settings.llm.api_key
            azure_endpoint = self.settings.llm.azure_endpoint
            api_version = self.settings.llm.api_version

        if provider == "azure":
            llm_client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version or "2024-02-15-preview",
            )
        elif provider == "openai":
            llm_client = AsyncOpenAI(api_key=api_key)
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: azure, openai. "
                "Tip: add evaluation.llm section in settings.yaml to use a "
                "dedicated provider for evaluation."
            )

        llm = llm_factory(model, client=llm_client, max_tokens=8192)

        # -- Embeddings: prefer evaluation.embedding, fallback to global --
        if eval_cfg.embedding is not None:
            emb_provider = eval_cfg.embedding.provider.lower()
            emb_model = eval_cfg.embedding.model
            emb_api_key = eval_cfg.embedding.api_key
            emb_azure_endpoint = eval_cfg.embedding.azure_endpoint
            emb_api_version = eval_cfg.embedding.api_version
        else:
            emb_provider = self.settings.embedding.provider.lower()
            emb_model = self.settings.embedding.model
            emb_api_key = self.settings.embedding.api_key
            emb_azure_endpoint = self.settings.embedding.azure_endpoint
            emb_api_version = self.settings.embedding.api_version

        if emb_provider == "azure":
            emb_client = AsyncAzureOpenAI(
                api_key=emb_api_key,
                azure_endpoint=emb_azure_endpoint,
                api_version=emb_api_version or "2024-02-15-preview",
            )
        elif emb_provider == "openai":
            emb_client = AsyncOpenAI(api_key=emb_api_key)
        else:
            raise ValueError(
                f"Unsupported embedding provider for Ragas: '{emb_provider}'. "
                "Supported: azure, openai. "
                "Tip: add evaluation.embedding section in settings.yaml."
            )

        embeddings = OpenAIEmbeddings(model=emb_model, client=emb_client)

        return llm, embeddings

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """Extract text strings from various chunk representations.

        Args:
            chunks: List of chunk dicts, strings, or objects with .text.

        Returns:
            List of text strings.
        """
        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # Filter to only ragas-supported metrics
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
