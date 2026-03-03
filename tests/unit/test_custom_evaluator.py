"""Tests for Evaluator abstraction: BaseEvaluator, NoneEvaluator, CustomEvaluator, EvaluatorFactory."""

from __future__ import annotations

from typing import Any

import pytest

from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator


# ===========================================================================
# BaseEvaluator.validate_query
# ===========================================================================

class TestValidateQuery:
    """Tests for BaseEvaluator.validate_query."""

    def setup_method(self) -> None:
        self.evaluator = NoneEvaluator()

    def test_valid_query(self) -> None:
        self.evaluator.validate_query("hello world")

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            self.evaluator.validate_query(123)  # type: ignore[arg-type]

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            self.evaluator.validate_query("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            self.evaluator.validate_query("   ")


# ===========================================================================
# BaseEvaluator.validate_retrieved_chunks
# ===========================================================================

class TestValidateRetrievedChunks:
    """Tests for BaseEvaluator.validate_retrieved_chunks."""

    def setup_method(self) -> None:
        self.evaluator = NoneEvaluator()

    def test_valid_chunks(self) -> None:
        self.evaluator.validate_retrieved_chunks([{"id": "1"}])

    def test_non_list_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            self.evaluator.validate_retrieved_chunks("not a list")  # type: ignore[arg-type]

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self.evaluator.validate_retrieved_chunks([])


# ===========================================================================
# NoneEvaluator
# ===========================================================================

class TestNoneEvaluator:
    """Tests for NoneEvaluator (returns empty metrics)."""

    def setup_method(self) -> None:
        self.evaluator = NoneEvaluator()

    def test_returns_empty_dict(self) -> None:
        result = self.evaluator.evaluate("query", [{"id": "1"}])
        assert result == {}

    def test_validates_query(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            self.evaluator.evaluate(123, [{"id": "1"}])  # type: ignore[arg-type]

    def test_validates_chunks(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self.evaluator.evaluate("query", [])

    def test_accepts_kwargs(self) -> None:
        result = self.evaluator.evaluate(
            "query", [{"id": "1"}],
            generated_answer="answer", ground_truth=["1"],
        )
        assert result == {}

    def test_accepts_config_kwargs(self) -> None:
        evaluator = NoneEvaluator(metrics=["hit_rate"], provider="none")
        assert evaluator.config == {"metrics": ["hit_rate"], "provider": "none"}


# ===========================================================================
# CustomEvaluator — construction
# ===========================================================================

class TestCustomEvaluatorInit:
    """Tests for CustomEvaluator construction and configuration."""

    def test_default_metrics(self) -> None:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator()
        assert set(evaluator.metrics) == {"hit_rate", "mrr"}

    def test_explicit_metrics(self) -> None:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator(metrics=["hit_rate"])
        assert evaluator.metrics == ["hit_rate"]

    def test_unsupported_metric_raises(self) -> None:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        with pytest.raises(ValueError, match="Unsupported custom metrics"):
            CustomEvaluator(metrics=["bleu"])

    def test_case_insensitive_metrics(self) -> None:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator(metrics=["HIT_RATE", "MRR"])
        assert evaluator.metrics == ["hit_rate", "mrr"]


# ===========================================================================
# CustomEvaluator — hit_rate
# ===========================================================================

class TestCustomEvaluatorHitRate:
    """Tests for CustomEvaluator hit_rate metric."""

    def setup_method(self) -> None:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        self.evaluator = CustomEvaluator(metrics=["hit_rate"])

    def test_hit_when_relevant_in_retrieved(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            ground_truth=["b"],
        )
        assert result["hit_rate"] == 1.0

    def test_miss_when_no_relevant_in_retrieved(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}, {"id": "b"}],
            ground_truth=["z"],
        )
        assert result["hit_rate"] == 0.0

    def test_no_ground_truth_returns_zero(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}],
            ground_truth=None,
        )
        assert result["hit_rate"] == 0.0

    def test_string_ground_truth(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}, {"id": "b"}],
            ground_truth="a",
        )
        assert result["hit_rate"] == 1.0

    def test_dict_ground_truth_with_ids_key(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}, {"id": "b"}],
            ground_truth={"ids": ["b", "c"]},
        )
        assert result["hit_rate"] == 1.0


# ===========================================================================
# CustomEvaluator — mrr
# ===========================================================================

class TestCustomEvaluatorMRR:
    """Tests for CustomEvaluator MRR metric."""

    def setup_method(self) -> None:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        self.evaluator = CustomEvaluator(metrics=["mrr"])

    def test_first_position(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            ground_truth=["a"],
        )
        assert result["mrr"] == 1.0

    def test_second_position(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            ground_truth=["b"],
        )
        assert result["mrr"] == pytest.approx(0.5)

    def test_third_position(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            ground_truth=["c"],
        )
        assert result["mrr"] == pytest.approx(1.0 / 3.0)

    def test_no_match_returns_zero(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}, {"id": "b"}],
            ground_truth=["z"],
        )
        assert result["mrr"] == 0.0

    def test_no_ground_truth_returns_zero(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}],
            ground_truth=None,
        )
        assert result["mrr"] == 0.0


# ===========================================================================
# CustomEvaluator — ID extraction
# ===========================================================================

class TestCustomEvaluatorIDExtraction:
    """Tests for CustomEvaluator id extraction from various shapes."""

    def setup_method(self) -> None:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        self.evaluator = CustomEvaluator(metrics=["hit_rate"])

    def test_string_ids_in_chunks(self) -> None:
        """Chunks can be plain strings (treated as IDs)."""
        result = self.evaluator.evaluate(
            "query",
            ["a", "b", "c"],
            ground_truth=["b"],
        )
        assert result["hit_rate"] == 1.0

    def test_chunk_id_field(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"chunk_id": "a"}, {"chunk_id": "b"}],
            ground_truth=["b"],
        )
        assert result["hit_rate"] == 1.0

    def test_missing_id_field_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing id field"):
            self.evaluator.evaluate(
                "query",
                [{"text": "no id here"}],
                ground_truth=["a"],
            )

    def test_unsupported_item_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unable to extract id"):
            self.evaluator.evaluate(
                "query",
                [12345],  # type: ignore[list-item]
                ground_truth=["a"],
            )

    def test_ground_truth_list_of_strings(self) -> None:
        result = self.evaluator.evaluate(
            "query",
            [{"id": "a"}],
            ground_truth=["a", "b"],
        )
        assert result["hit_rate"] == 1.0

    def test_unsupported_ground_truth_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported ground_truth type"):
            self.evaluator.evaluate(
                "query",
                [{"id": "a"}],
                ground_truth=12345,
            )


# ===========================================================================
# CustomEvaluator — combined metrics
# ===========================================================================

class TestCustomEvaluatorCombined:
    """Tests for CustomEvaluator with multiple metrics."""

    def test_both_metrics(self) -> None:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator(metrics=["hit_rate", "mrr"])
        result = evaluator.evaluate(
            "query",
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            ground_truth=["b"],
        )
        assert result["hit_rate"] == 1.0
        assert result["mrr"] == pytest.approx(0.5)

    def test_validates_query(self) -> None:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator()
        with pytest.raises(ValueError, match="empty or whitespace"):
            evaluator.evaluate("", [{"id": "a"}])

    def test_validates_chunks(self) -> None:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator()
        with pytest.raises(ValueError, match="cannot be empty"):
            evaluator.evaluate("query", [])


# ===========================================================================
# EvaluatorFactory
# ===========================================================================

class TestEvaluatorFactory:
    """Tests for the Evaluator factory routing logic."""

    def setup_method(self) -> None:
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        self.factory = EvaluatorFactory()

    def test_register_and_create(self) -> None:
        self.factory.register_provider("none_eval", NoneEvaluator)
        evaluator = self.factory.create_by_name("none_eval")
        assert isinstance(evaluator, NoneEvaluator)

    def test_case_insensitive_registration(self) -> None:
        self.factory.register_provider("NoNe", NoneEvaluator)
        evaluator = self.factory.create_by_name("none")
        assert isinstance(evaluator, NoneEvaluator)

    def test_case_insensitive_create(self) -> None:
        self.factory.register_provider("test", NoneEvaluator)
        evaluator = self.factory.create_by_name("TEST")
        assert isinstance(evaluator, NoneEvaluator)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Evaluator provider"):
            self.factory.create_by_name("nonexistent")

    def test_error_lists_available_providers(self) -> None:
        self.factory.register_provider("custom", NoneEvaluator)
        with pytest.raises(ValueError, match="custom"):
            self.factory.create_by_name("missing")

    def test_register_non_subclass_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a subclass"):
            self.factory.register_provider("bad", dict)  # type: ignore[arg-type]

    def test_list_providers_instance(self) -> None:
        """Instance-level list_providers only shows instance-registered ones."""
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory as EF

        factory = EF()
        factory.register_provider("beta", NoneEvaluator)
        factory.register_provider("alpha", NoneEvaluator)
        providers = sorted(factory._providers)
        assert providers == ["alpha", "beta"]

    def test_create_with_kwargs(self) -> None:
        self.factory.register_provider("test", NoneEvaluator)
        evaluator = self.factory.create_by_name("test", metrics=["hit_rate"])
        assert isinstance(evaluator, NoneEvaluator)
        assert evaluator.config["metrics"] == ["hit_rate"]

    def test_create_from_settings(self) -> None:
        from src.core.settings import EvaluationSettings
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        self.factory.register_provider("custom", CustomEvaluator)

        settings = EvaluationSettings(
            enabled=True,
            provider="custom",
            metrics=["hit_rate", "mrr"],
        )
        evaluator = self.factory.create_from_settings(settings)
        assert isinstance(evaluator, CustomEvaluator)

    def test_create_from_settings_forwards_metrics(self) -> None:
        from src.core.settings import EvaluationSettings
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        self.factory.register_provider("custom", CustomEvaluator)

        settings = EvaluationSettings(
            enabled=True,
            provider="custom",
            metrics=["hit_rate"],
        )
        evaluator = self.factory.create_from_settings(settings)
        assert isinstance(evaluator, CustomEvaluator)
        assert evaluator.metrics == ["hit_rate"]

    def test_create_from_settings_disabled_returns_none_evaluator(self) -> None:
        from src.core.settings import EvaluationSettings

        settings = EvaluationSettings(
            enabled=False,
            provider="custom",
            metrics=["hit_rate"],
        )
        evaluator = self.factory.create_from_settings(settings)
        assert isinstance(evaluator, NoneEvaluator)

    def test_create_from_settings_none_provider_returns_none_evaluator(self) -> None:
        from src.core.settings import EvaluationSettings

        settings = EvaluationSettings(
            enabled=True,
            provider="none",
            metrics=[],
        )
        evaluator = self.factory.create_from_settings(settings)
        assert isinstance(evaluator, NoneEvaluator)

    def test_empty_registry_lists_none(self) -> None:
        with pytest.raises(ValueError, match="\\(none\\)"):
            self.factory.create_by_name("anything")

    def test_class_list_providers_includes_lazy(self) -> None:
        """Class-level list_providers includes both eager and lazy providers."""
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory as EF

        providers = EF.list_providers()
        assert "custom" in providers
        assert "ragas" in providers
        assert "composite" in providers

    def test_duplicate_registration_overwrites(self) -> None:
        """Registering same name twice overwrites the previous provider."""
        self.factory.register_provider("test", NoneEvaluator)

        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        self.factory.register_provider("test", CustomEvaluator)
        evaluator = self.factory.create_by_name("test")
        assert isinstance(evaluator, CustomEvaluator)


# ===========================================================================
# Boundary tests — evaluator edge cases
# ===========================================================================


class TestCustomEvaluatorBoundary:
    """Boundary tests for CustomEvaluator edge cases."""

    def test_empty_ground_truth_list(self) -> None:
        """Empty ground_truth list returns 0 scores."""
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator()
        result = evaluator.evaluate(
            "query", [{"id": "a"}], ground_truth=[],
        )
        assert result["hit_rate"] == 0.0
        assert result["mrr"] == 0.0

    def test_generated_answer_ignored(self) -> None:
        """CustomEvaluator ignores generated_answer (not LLM-as-Judge)."""
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator()
        result = evaluator.evaluate(
            "query",
            [{"id": "a"}],
            ground_truth=["a"],
            generated_answer="this is ignored",
        )
        assert result["hit_rate"] == 1.0

    def test_large_candidate_list(self) -> None:
        """Evaluate with many candidates."""
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator()
        chunks = [{"id": f"c{i}"} for i in range(100)]
        result = evaluator.evaluate(
            "query", chunks, ground_truth=["c50"],
        )
        assert result["hit_rate"] == 1.0
        assert result["mrr"] == pytest.approx(1.0 / 51.0)

    def test_unicode_query_in_evaluator(self) -> None:
        """Chinese query is valid for evaluation."""
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator()
        result = evaluator.evaluate(
            "什么是混合检索？",
            [{"id": "a"}],
            ground_truth=["a"],
        )
        assert result["hit_rate"] == 1.0

    def test_dict_ground_truth_missing_ids_key(self) -> None:
        """Dict ground_truth without 'ids' key raises ValueError."""
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator()
        with pytest.raises(ValueError, match="Missing id field"):
            evaluator.evaluate(
                "query", [{"id": "a"}], ground_truth={"labels": ["a"]},
            )


class TestNoneEvaluatorBoundary:
    """Boundary tests for NoneEvaluator edge cases."""

    def test_accepts_all_optional_params(self) -> None:
        """NoneEvaluator accepts all optional params without error."""
        evaluator = NoneEvaluator()
        result = evaluator.evaluate(
            "query",
            [{"id": "a"}],
            generated_answer="answer",
            ground_truth=["a"],
            trace=None,
            extra_param="ignored",
        )
        assert result == {}

    def test_single_chunk(self) -> None:
        """NoneEvaluator works with single chunk."""
        evaluator = NoneEvaluator()
        result = evaluator.evaluate("query", [{"id": "a"}])
        assert result == {}
