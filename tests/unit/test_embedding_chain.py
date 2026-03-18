"""Unit tests for EmbeddingChain — multi-provider failover for embeddings."""

from unittest.mock import MagicMock

import pytest

from src.libs.circuit_breaker.circuit_breaker import CircuitBreaker, CircuitState
from src.libs.circuit_breaker.embedding_chain import (
    AllEmbeddingProvidersUnavailableError,
    EmbeddingChain,
)


def _make_embedding(name: str, fail: bool = False, dimension: int = 768) -> MagicMock:
    emb = MagicMock()
    emb.__class__.__name__ = name
    if fail:
        emb.embed.side_effect = RuntimeError(f"{name} down")
    else:
        emb.embed.return_value = [[0.1, 0.2, 0.3]]
    emb.get_dimension.return_value = dimension
    return emb


class TestEmbeddingChain:
    def test_uses_first_provider(self):
        emb1 = _make_embedding("primary")
        chain = EmbeddingChain([(emb1, CircuitBreaker())])
        result = chain.embed(["hello"])
        assert result == [[0.1, 0.2, 0.3]]
        emb1.embed.assert_called_once_with(["hello"])

    def test_failover_on_circuit_open(self):
        emb1 = _make_embedding("primary")
        cb1 = CircuitBreaker(failure_threshold=1)
        cb1.record_failure()  # opens circuit
        assert cb1.state == CircuitState.OPEN

        emb2 = _make_embedding("backup")
        chain = EmbeddingChain([(emb1, cb1), (emb2, CircuitBreaker())])
        result = chain.embed(["hello"])
        assert result == [[0.1, 0.2, 0.3]]
        emb1.embed.assert_not_called()
        emb2.embed.assert_called_once_with("hello" and ["hello"])

    def test_records_success(self):
        emb1 = _make_embedding("primary")
        cb1 = CircuitBreaker()
        chain = EmbeddingChain([(emb1, cb1)])
        chain.embed(["hello"])
        # After success, circuit should stay CLOSED
        assert cb1.state == CircuitState.CLOSED

    def test_records_failure(self):
        emb1 = _make_embedding("primary", fail=True)
        cb1 = CircuitBreaker(failure_threshold=1)
        emb2 = _make_embedding("backup")
        chain = EmbeddingChain(
            [
                (emb1, cb1),
                (emb2, CircuitBreaker()),
            ]
        )
        chain.embed(["hello"])
        # Primary failed — its circuit should be open after threshold=1
        assert cb1.state == CircuitState.OPEN

    def test_all_unavailable_raises(self):
        emb1 = _make_embedding("p1", fail=True)
        emb2 = _make_embedding("p2", fail=True)
        chain = EmbeddingChain(
            [
                (emb1, CircuitBreaker(failure_threshold=1)),
                (emb2, CircuitBreaker(failure_threshold=1)),
            ]
        )
        with pytest.raises(AllEmbeddingProvidersUnavailableError):
            chain.embed(["hello"])

    def test_get_dimension_from_first_available(self):
        emb1 = _make_embedding("primary", dimension=1536)
        chain = EmbeddingChain([(emb1, CircuitBreaker())])
        assert chain.get_dimension() == 1536

    def test_get_dimension_skips_circuit_open(self):
        emb1 = _make_embedding("primary", dimension=1536)
        cb1 = CircuitBreaker(failure_threshold=1)
        cb1.record_failure()  # opens circuit
        assert cb1.state == CircuitState.OPEN

        emb2 = _make_embedding("backup", dimension=768)
        chain = EmbeddingChain([(emb1, cb1), (emb2, CircuitBreaker())])
        assert chain.get_dimension() == 768

    def test_empty_providers_raises(self):
        with pytest.raises(ValueError, match="at least one provider"):
            EmbeddingChain([])

    def test_kwargs_forwarded(self):
        emb1 = _make_embedding("primary")
        chain = EmbeddingChain([(emb1, CircuitBreaker())])
        chain.embed(["hello"], batch_size=32)
        emb1.embed.assert_called_once_with(["hello"], batch_size=32)
