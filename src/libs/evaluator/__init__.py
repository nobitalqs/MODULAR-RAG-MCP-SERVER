"""
Evaluator - RAG evaluation abstraction.

Components:
- BaseEvaluator: Abstract base class
- NoneEvaluator: No-op fallback (disabled evaluation)
- CustomEvaluator: Lightweight metrics (hit_rate, mrr)
- EvaluatorFactory: Backend routing factory
"""

from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory

__all__ = ["BaseEvaluator", "NoneEvaluator", "CustomEvaluator", "EvaluatorFactory"]
