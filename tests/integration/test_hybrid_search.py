"""Integration tests for HybridSearch with TraceContext (F3).

Covers:
- Complete retrieval flow with trace recording
- Trace contains all expected stages: query_processing, dense_retrieval,
  sparse_retrieval, fusion, rerank
- Each stage records method and elapsed_ms
- trace_type == "query"
- Graceful degradation and configuration tests
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.core.query_engine.fusion import RRFFusion
from src.core.query_engine.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
    HybridSearchResult,
    create_hybrid_search,
)
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import CoreReranker, RerankConfig
from src.core.trace.trace_context import TraceContext
from src.core.types import ProcessedQuery, RetrievalResult
from src.libs.reranker.base_reranker import NoneReranker


# ── Mock Components ─────────────────────────────────────────────────


class MockDenseRetriever:
    """Mock Dense Retriever for testing."""

    def __init__(
        self,
        results: Optional[List[RetrievalResult]] = None,
        should_fail: bool = False,
    ):
        self.results = results or []
        self.should_fail = should_fail
        self.call_count = 0
        self.last_top_k: Optional[int] = None
        self.last_filters: Optional[Dict[str, Any]] = None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        self.call_count += 1
        self.last_top_k = top_k
        self.last_filters = filters
        if self.should_fail:
            raise RuntimeError("Dense retrieval failed")
        return self.results[:top_k]


class MockSparseRetriever:
    """Mock Sparse Retriever for testing."""

    def __init__(
        self,
        results: Optional[List[RetrievalResult]] = None,
        should_fail: bool = False,
    ):
        self.results = results or []
        self.should_fail = should_fail
        self.call_count = 0
        self.last_top_k: Optional[int] = None
        self.last_collection: Optional[str] = None

    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        self.call_count += 1
        self.last_top_k = top_k
        self.last_collection = collection
        if self.should_fail:
            raise RuntimeError("Sparse retrieval failed")
        return self.results[:top_k]


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def sample_dense_results() -> List[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="dense_1", score=0.95,
            text="Azure OpenAI 配置步骤详解",
            metadata={"source_path": "docs/azure.pdf", "collection": "api-docs"},
        ),
        RetrievalResult(
            chunk_id="common_chunk", score=0.85,
            text="通用配置说明",
            metadata={"source_path": "docs/common.pdf", "collection": "general"},
        ),
        RetrievalResult(
            chunk_id="dense_3", score=0.80,
            text="云服务配置概述",
            metadata={"source_path": "docs/cloud.pdf", "collection": "general"},
        ),
    ]


@pytest.fixture()
def sample_sparse_results() -> List[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="sparse_1", score=8.5,
            text="Azure 配置 Azure OpenAI 服务",
            metadata={"source_path": "docs/azure-setup.pdf", "collection": "tutorials"},
        ),
        RetrievalResult(
            chunk_id="common_chunk", score=7.2,
            text="通用配置说明",
            metadata={"source_path": "docs/common.pdf", "collection": "general"},
        ),
    ]


@pytest.fixture()
def query_processor() -> QueryProcessor:
    return QueryProcessor()


@pytest.fixture()
def rrf_fusion() -> RRFFusion:
    return RRFFusion(k=60)


# ── Trace Integration Tests (F3) ───────────────────────────────────


class TestHybridSearchTracing:
    """Verify that a query generates trace with all expected stages."""

    def test_trace_contains_all_query_stages(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        """A full hybrid search should record query_processing, dense, sparse, fusion."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        trace = TraceContext(trace_type="query")
        hybrid.search("如何配置 Azure OpenAI？", top_k=5, trace=trace)
        trace.finish()

        stage_names = [s["stage"] for s in trace.stages]
        assert "query_processing" in stage_names
        assert "dense_retrieval" in stage_names
        assert "sparse_retrieval" in stage_names
        assert "fusion" in stage_names

    def test_trace_type_is_query(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        trace = TraceContext(trace_type="query")
        hybrid.search("配置", top_k=5, trace=trace)
        trace.finish()

        d = trace.to_dict()
        assert d["trace_type"] == "query"

    def test_each_stage_has_method_and_elapsed(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        """Every stage must record 'method' in data and 'elapsed_ms'."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        trace = TraceContext(trace_type="query")
        hybrid.search("Azure 配置", top_k=5, trace=trace)
        trace.finish()

        for entry in trace.stages:
            assert "elapsed_ms" in entry, f"stage {entry['stage']} missing elapsed_ms"
            assert "method" in entry["data"], f"stage {entry['stage']} missing method"

    def test_trace_to_dict_is_json_serialisable(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        import json

        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        trace = TraceContext(trace_type="query")
        hybrid.search("Azure", top_k=5, trace=trace)
        trace.finish()

        text = json.dumps(trace.to_dict())
        parsed = json.loads(text)
        assert parsed["trace_type"] == "query"
        assert len(parsed["stages"]) >= 3

    def test_trace_total_elapsed_ms_positive(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        trace = TraceContext(trace_type="query")
        hybrid.search("Azure", top_k=5, trace=trace)
        trace.finish()

        assert trace.to_dict()["total_elapsed_ms"] > 0

    def test_rerank_stage_recorded(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        """CoreReranker should record a 'rerank' stage in the trace."""
        from unittest.mock import MagicMock
        from src.core.settings import Settings

        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        trace = TraceContext(trace_type="query")
        search_results = hybrid.search("Azure 配置", top_k=5, trace=trace)

        # Now rerank with a mock backend that actually reranks
        mock_backend = MagicMock()
        mock_backend.rerank.return_value = [
            {"id": r.chunk_id, "text": r.text, "score": r.score, "rerank_score": 1.0 - i * 0.1}
            for i, r in enumerate(search_results)
        ]

        mock_settings = MagicMock(spec=Settings)
        reranker = CoreReranker(
            settings=mock_settings,
            reranker=mock_backend,
            config=RerankConfig(enabled=True, top_k=3),
        )
        reranker.rerank("Azure 配置", search_results, trace=trace)
        trace.finish()

        stage_names = [s["stage"] for s in trace.stages]
        assert "rerank" in stage_names

        rerank_entry = next(s for s in trace.stages if s["stage"] == "rerank")
        assert "elapsed_ms" in rerank_entry
        assert "method" in rerank_entry["data"]

    def test_trace_without_trace_param_no_error(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        """Search without trace param should work without errors."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        # No trace — should not raise
        results = hybrid.search("Azure 配置", top_k=5)
        assert len(results) > 0


# ── Basic Functionality Tests ───────────────────────────────────────


class TestHybridSearchBasic:
    """Test basic HybridSearch functionality."""

    def test_search_returns_results(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        results = hybrid.search("如何配置 Azure OpenAI？", top_k=5)
        assert 0 < len(results) <= 5
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_search_calls_both_retrievers(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        hybrid.search("Azure 配置", top_k=5)
        assert dense.call_count == 1
        assert sparse.call_count == 1

    def test_common_chunks_deduplicated(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        results = hybrid.search("配置", top_k=10)
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))


# ── Graceful Degradation Tests ──────────────────────────────────────


class TestHybridSearchDegradation:
    """Test graceful degradation when components fail."""

    def test_dense_fails_uses_sparse_only(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_sparse_results: List[RetrievalResult],
    ) -> None:
        dense = MockDenseRetriever(should_fail=True)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        result = hybrid.search("Azure 配置", top_k=5, return_details=True)
        assert result.used_fallback is True
        assert result.dense_error is not None
        assert len(result.results) > 0

    def test_sparse_fails_uses_dense_only(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
        sample_dense_results: List[RetrievalResult],
    ) -> None:
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(should_fail=True)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        result = hybrid.search("Azure 配置", top_k=5, return_details=True)
        assert result.used_fallback is True
        assert result.sparse_error is not None
        assert len(result.results) > 0

    def test_both_fail_raises_error(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
    ) -> None:
        dense = MockDenseRetriever(should_fail=True)
        sparse = MockSparseRetriever(should_fail=True)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        with pytest.raises(RuntimeError, match="Both retrieval paths failed"):
            hybrid.search("Azure 配置", top_k=5)

    def test_empty_query_raises_error(
        self,
        query_processor: QueryProcessor,
        rrf_fusion: RRFFusion,
    ) -> None:
        hybrid = HybridSearch(
            query_processor=query_processor,
            fusion=rrf_fusion,
        )

        with pytest.raises(ValueError, match="(?i)empty"):
            hybrid.search("")


# ── Factory Tests ───────────────────────────────────────────────────


class TestCreateHybridSearch:
    """Test create_hybrid_search factory."""

    def test_creates_default_fusion(self) -> None:
        hybrid = create_hybrid_search()
        assert isinstance(hybrid.fusion, RRFFusion)
        assert hybrid.fusion.k == 60
