"""Evaluation Module - RAG quality assessment.

Components:
- RagasEvaluator: LLM-as-Judge metrics via Ragas framework
- CompositeEvaluator: Multi-backend parallel evaluation
- EvalRunner: Batch evaluation orchestration
"""

from src.observability.evaluation.eval_runner import (
    EvalReport,
    EvalRunner,
    GoldenTestCase,
    QueryResult,
    load_test_set,
)

__all__ = [
    "EvalReport",
    "EvalRunner",
    "GoldenTestCase",
    "QueryResult",
    "load_test_set",
]
