"""Tests for HybridSearch orchestration layer.

Test Strategy:
- Use mock/fake retrievers for deterministic behavior
- Test actual component integration (QueryProcessor, RRFFusion)
- Cover both success and failure scenarios
- Verify graceful degradation, filtering, and configuration
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from src.core.types import ProcessedQuery, RetrievalResult
from src.core.query_engine.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
    HybridSearchResult,
    create_hybrid_search,
)
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.fusion import RRFFusion


# =============================================================================
# Mock Components
# =============================================================================


class MockDenseRetriever:
    """Mock Dense Retriever for testing."""

    def __init__(
        self,
        results: Optional[List[RetrievalResult]] = None,
        should_fail: bool = False,
        error_message: str = "Dense retrieval failed",
    ):
        self.results = results or []
        self.should_fail = should_fail
        self.error_message = error_message
        self.call_count = 0
        self.last_query = None
        self.last_top_k = None
        self.last_filters = None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        self.call_count += 1
        self.last_query = query
        self.last_top_k = top_k
        self.last_filters = filters

        if self.should_fail:
            raise RuntimeError(self.error_message)

        return self.results[:top_k]


class MockSparseRetriever:
    """Mock Sparse Retriever for testing."""

    def __init__(
        self,
        results: Optional[List[RetrievalResult]] = None,
        should_fail: bool = False,
        error_message: str = "Sparse retrieval failed",
    ):
        self.results = results or []
        self.should_fail = should_fail
        self.error_message = error_message
        self.call_count = 0
        self.last_keywords = None
        self.last_top_k = None
        self.last_collection = None

    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        self.call_count += 1
        self.last_keywords = keywords
        self.last_top_k = top_k
        self.last_collection = collection

        if self.should_fail:
            raise RuntimeError(self.error_message)

        return self.results[:top_k]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dense_results() -> List[RetrievalResult]:
    """Sample results from dense retrieval."""
    return [
        RetrievalResult(
            chunk_id="dense_1",
            score=0.95,
            text="Azure OpenAI 配置步骤详解",
            metadata={"source_path": "docs/azure.pdf", "collection": "api-docs"},
        ),
        RetrievalResult(
            chunk_id="dense_2",
            score=0.88,
            text="OpenAI API 使用指南",
            metadata={"source_path": "docs/openai.pdf", "collection": "api-docs"},
        ),
        RetrievalResult(
            chunk_id="common_chunk",
            score=0.85,
            text="通用配置说明",
            metadata={"source_path": "docs/common.pdf", "collection": "general"},
        ),
        RetrievalResult(
            chunk_id="dense_4",
            score=0.80,
            text="云服务配置概述",
            metadata={"source_path": "docs/cloud.pdf", "collection": "general"},
        ),
    ]


@pytest.fixture
def sample_sparse_results() -> List[RetrievalResult]:
    """Sample results from sparse retrieval."""
    return [
        RetrievalResult(
            chunk_id="sparse_1",
            score=8.5,
            text="Azure 配置 Azure OpenAI 服务",
            metadata={
                "source_path": "docs/azure-setup.pdf",
                "collection": "tutorials",
            },
        ),
        RetrievalResult(
            chunk_id="common_chunk",
            score=7.2,
            text="通用配置说明",
            metadata={"source_path": "docs/common.pdf", "collection": "general"},
        ),
        RetrievalResult(
            chunk_id="sparse_3",
            score=6.8,
            text="配置文件 YAML 格式说明",
            metadata={
                "source_path": "docs/config.pdf",
                "collection": "tutorials",
            },
        ),
    ]


@pytest.fixture
def query_processor() -> QueryProcessor:
    """Real QueryProcessor instance."""
    return QueryProcessor()


@pytest.fixture
def rrf_fusion() -> RRFFusion:
    """Real RRFFusion instance with default k=60."""
    return RRFFusion(k=60)


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestHybridSearchBasic:
    """Test basic HybridSearch functionality."""

    def test_init_with_all_components(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test initialization with all components."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        assert hybrid.query_processor is query_processor
        assert hybrid.dense_retriever is dense
        assert hybrid.sparse_retriever is sparse
        assert hybrid.fusion is rrf_fusion

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = HybridSearchConfig(
            dense_top_k=30,
            sparse_top_k=30,
            fusion_top_k=15,
            parallel_retrieval=False,
        )

        hybrid = HybridSearch(config=config)

        assert hybrid.config.dense_top_k == 30
        assert hybrid.config.sparse_top_k == 30
        assert hybrid.config.fusion_top_k == 15
        assert hybrid.config.parallel_retrieval is False

    def test_search_returns_results(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that search returns fused results."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        results = hybrid.search("如何配置 Azure OpenAI？", top_k=5)

        assert len(results) > 0
        assert len(results) <= 5

        for r in results:
            assert isinstance(r, RetrievalResult)
            assert r.chunk_id
            assert isinstance(r.score, float)
            assert r.text

    def test_search_with_return_details(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test search with return_details=True."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        result = hybrid.search("Azure config", top_k=5, return_details=True)

        assert isinstance(result, HybridSearchResult)
        assert result.results is not None
        assert result.dense_results is not None
        assert result.sparse_results is not None
        assert result.dense_error is None
        assert result.sparse_error is None
        assert result.used_fallback is False
        assert result.processed_query is not None

    def test_search_calls_both_retrievers(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that both retrievers are called."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        hybrid.search("Azure OpenAI 配置", top_k=5)

        assert dense.call_count == 1
        assert sparse.call_count == 1

    def test_common_chunks_deduplicated(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that common chunks appear only once in results."""
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
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids found"
        assert chunk_ids.count("common_chunk") <= 1


# =============================================================================
# Graceful Degradation Tests
# =============================================================================


class TestHybridSearchDegradation:
    """Test graceful degradation when components fail."""

    def test_dense_fails_uses_sparse_only(
        self,
        query_processor,
        rrf_fusion,
        sample_sparse_results,
    ):
        """Test fallback to sparse when dense fails."""
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
        assert "Dense retrieval" in result.dense_error
        assert result.sparse_error is None
        assert len(result.results) > 0

    def test_sparse_fails_uses_dense_only(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
    ):
        """Test fallback to dense when sparse fails."""
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
        assert "Sparse retrieval" in result.sparse_error
        assert result.dense_error is None
        assert len(result.results) > 0

    def test_both_fail_raises_error(
        self,
        query_processor,
        rrf_fusion,
    ):
        """Test that RuntimeError is raised when both retrievers fail."""
        dense = MockDenseRetriever(should_fail=True)
        sparse = MockSparseRetriever(should_fail=True)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        with pytest.raises(RuntimeError) as exc_info:
            hybrid.search("Azure 配置", top_k=5)

        assert "Both retrieval paths failed" in str(exc_info.value)

    def test_no_retrievers_configured(
        self,
        query_processor,
        rrf_fusion,
    ):
        """Test behavior when no retrievers are configured."""
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=None,
            sparse_retriever=None,
            fusion=rrf_fusion,
        )

        with pytest.raises(RuntimeError) as exc_info:
            hybrid.search("Azure 配置", top_k=5)

        error_msg = str(exc_info.value)
        assert "No retriever" in error_msg or "Both" in error_msg

    def test_dense_only_mode(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
    ):
        """Test search with only dense retriever."""
        dense = MockDenseRetriever(results=sample_dense_results)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=None,
            fusion=rrf_fusion,
        )

        results = hybrid.search("Azure 配置", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3

    def test_sparse_only_mode(
        self,
        query_processor,
        rrf_fusion,
        sample_sparse_results,
    ):
        """Test search with only sparse retriever."""
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=None,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        results = hybrid.search("Azure 配置", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3


# =============================================================================
# Filter Tests
# =============================================================================


class TestHybridSearchFilters:
    """Test metadata filtering functionality."""

    def test_explicit_filters_passed_to_retrievers(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that explicit filters are passed to retrievers."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        hybrid.search("Azure", top_k=5, filters={"collection": "api-docs"})

        # "collection" is storage-level routing, stripped before reaching
        # dense retriever (ChromaDB has no such metadata field).
        assert dense.last_filters is None
        assert sparse.last_collection == "api-docs"

    def test_query_filter_syntax_extraction(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that filters in query syntax are extracted."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        result = hybrid.search(
            "collection:api-docs Azure 配置",
            top_k=5,
            return_details=True,
        )

        assert result.processed_query is not None
        assert "collection" in result.processed_query.filters

    def test_post_fusion_metadata_filter(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test post-fusion metadata filtering.

        "collection" is storage-level routing (ChromaDB table name), not
        chunk metadata — it must be stripped before post-fusion filtering.
        Only real metadata fields (doc_type, tags, etc.) should filter.
        """
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        config = HybridSearchConfig(metadata_filter_post=True)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
            config=config,
        )

        # collection-only filter should NOT remove any results
        results = hybrid.search(
            "Azure", top_k=10, filters={"collection": "api-docs"},
        )
        assert len(results) > 0  # nothing filtered out

        # real metadata filter (doc_type) should still work
        results_filtered = hybrid.search(
            "Azure", top_k=10,
            filters={"collection": "api-docs", "doc_type": "nonexistent"},
        )
        assert len(results_filtered) == 0  # all filtered out by doc_type


# =============================================================================
# Configuration Tests
# =============================================================================


class TestHybridSearchConfig:
    """Test configuration behavior."""

    def test_top_k_from_config(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that top_k values from config are used."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        config = HybridSearchConfig(
            dense_top_k=3,
            sparse_top_k=3,
            fusion_top_k=2,
        )
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
            config=config,
        )

        results = hybrid.search("Azure")  # No explicit top_k

        assert dense.last_top_k == 3
        assert sparse.last_top_k == 3
        assert len(results) <= 2

    def test_top_k_override(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that explicit top_k overrides config."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        results = hybrid.search("Azure", top_k=1)

        assert len(results) == 1

    def test_sequential_retrieval_mode(
        self,
        query_processor,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test sequential retrieval mode (non-parallel)."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        config = HybridSearchConfig(parallel_retrieval=False)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
            config=config,
        )

        results = hybrid.search("Azure", top_k=5)

        assert dense.call_count == 1
        assert sparse.call_count == 1
        assert len(results) > 0


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestHybridSearchEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query_raises_error(self, query_processor, rrf_fusion):
        """Test that empty query raises ValueError."""
        hybrid = HybridSearch(
            query_processor=query_processor,
            fusion=rrf_fusion,
        )

        with pytest.raises(ValueError) as exc_info:
            hybrid.search("")

        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_query_raises_error(self, query_processor, rrf_fusion):
        """Test that whitespace-only query raises ValueError."""
        hybrid = HybridSearch(
            query_processor=query_processor,
            fusion=rrf_fusion,
        )

        with pytest.raises(ValueError) as exc_info:
            hybrid.search("   \t\n  ")

        assert "empty" in str(exc_info.value).lower()

    def test_empty_results_from_both_retrievers(
        self,
        query_processor,
        rrf_fusion,
    ):
        """Test handling when both retrievers return empty results."""
        dense = MockDenseRetriever(results=[])
        sparse = MockSparseRetriever(results=[])

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        results = hybrid.search("obscure query with no matches", top_k=5)

        assert results == []

    def test_query_without_keywords_skips_sparse(
        self,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that sparse is skipped when no keywords extracted."""
        mock_processor = MagicMock()
        mock_processor.process.return_value = ProcessedQuery(
            original_query="的",
            keywords=[],
            filters={},
        )

        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        results = hybrid.search("的", top_k=5)

        assert dense.call_count == 1
        assert len(results) > 0

    def test_no_query_processor_fallback(
        self,
        rrf_fusion,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test fallback when no QueryProcessor is configured."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=None,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        results = hybrid.search("Azure OpenAI", top_k=5)

        assert len(results) > 0

    def test_no_fusion_interleave_fallback(
        self,
        query_processor,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test interleave fallback when no fusion is configured."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=None,
        )

        results = hybrid.search("Azure", top_k=5)

        assert len(results) > 0
        assert len(results) <= 5


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateHybridSearch:
    """Test the create_hybrid_search factory function."""

    def test_creates_default_fusion(self):
        """Test that default RRF fusion is created."""
        hybrid = create_hybrid_search()

        assert hybrid.fusion is not None
        assert isinstance(hybrid.fusion, RRFFusion)
        assert hybrid.fusion.k == 60

    def test_uses_provided_fusion(self):
        """Test that provided fusion is used."""
        custom_fusion = RRFFusion(k=30)

        hybrid = create_hybrid_search(fusion=custom_fusion)

        assert hybrid.fusion is custom_fusion
        assert hybrid.fusion.k == 30

    def test_passes_all_components(
        self,
        query_processor,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that all components are passed through."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)

        hybrid = create_hybrid_search(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
        )

        assert hybrid.query_processor is query_processor
        assert hybrid.dense_retriever is dense
        assert hybrid.sparse_retriever is sparse


# =============================================================================
# Weight Configuration Tests (K1)
# =============================================================================


class TestHybridSearchWeights:
    """Test hybrid search weight configuration (K1)."""

    def test_default_weights_are_uniform(self):
        """Test that default weights are 1.0 (backward compatible)."""
        config = HybridSearchConfig()
        assert config.dense_weight == 1.0
        assert config.sparse_weight == 1.0

    def test_weights_propagate_from_settings(self):
        """Test that weights are extracted from RetrievalSettings."""
        from src.core.settings import RetrievalSettings

        retrieval = RetrievalSettings(
            dense_top_k=20,
            sparse_top_k=20,
            fusion_top_k=10,
            rrf_k=60,
            dense_weight=1.5,
            sparse_weight=0.5,
        )

        # Create a mock settings object with the retrieval attribute
        mock_settings = MagicMock()
        mock_settings.retrieval = retrieval

        hybrid = HybridSearch(settings=mock_settings)

        assert hybrid.config.dense_weight == 1.5
        assert hybrid.config.sparse_weight == 0.5

    def test_asymmetric_weights_change_ranking(
        self,
        query_processor,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that asymmetric weights change result ordering."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        fusion = RRFFusion(k=60)

        # Dense-heavy: extreme weights so dense_1 (rank 1, dense only) beats
        # common_chunk (ranks in both lists but lower)
        config_dense_heavy = HybridSearchConfig(
            dense_weight=10.0, sparse_weight=0.1,
        )
        hybrid_dense = HybridSearch(
            query_processor=query_processor,
            dense_retriever=MockDenseRetriever(results=sample_dense_results),
            sparse_retriever=MockSparseRetriever(results=sample_sparse_results),
            fusion=RRFFusion(k=60),
            config=config_dense_heavy,
        )

        # Sparse-heavy: extreme weights so sparse_1 (rank 1, sparse only) wins
        config_sparse_heavy = HybridSearchConfig(
            dense_weight=0.1, sparse_weight=10.0,
        )
        hybrid_sparse = HybridSearch(
            query_processor=query_processor,
            dense_retriever=MockDenseRetriever(results=sample_dense_results),
            sparse_retriever=MockSparseRetriever(results=sample_sparse_results),
            fusion=RRFFusion(k=60),
            config=config_sparse_heavy,
        )

        results_dense = hybrid_dense.search("Azure 配置", top_k=5)
        results_sparse = hybrid_sparse.search("Azure 配置", top_k=5)

        # Top results should differ because weights shift ranking
        dense_top = results_dense[0].chunk_id if results_dense else None
        sparse_top = results_sparse[0].chunk_id if results_sparse else None
        # dense_1 ranks #1 in dense list; sparse_1 ranks #1 in sparse list
        # With extreme weighting, the dominant list's top item should win
        assert dense_top == "dense_1"
        assert sparse_top == "sparse_1"

    def test_zero_dense_weight_favors_sparse(
        self,
        query_processor,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that zero dense_weight effectively disables dense in fusion."""
        config = HybridSearchConfig(dense_weight=0.0, sparse_weight=1.0)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=MockDenseRetriever(results=sample_dense_results),
            sparse_retriever=MockSparseRetriever(results=sample_sparse_results),
            fusion=RRFFusion(k=60),
            config=config,
        )

        results = hybrid.search("Azure 配置", top_k=5)

        # With dense_weight=0, all dense contributions are 0.
        # sparse_1 should rank highest (rank 1 in sparse, weight 1.0)
        assert results[0].chunk_id == "sparse_1"

    def test_zero_sparse_weight_favors_dense(
        self,
        query_processor,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that zero sparse_weight effectively disables sparse in fusion."""
        config = HybridSearchConfig(dense_weight=1.0, sparse_weight=0.0)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=MockDenseRetriever(results=sample_dense_results),
            sparse_retriever=MockSparseRetriever(results=sample_sparse_results),
            fusion=RRFFusion(k=60),
            config=config,
        )

        results = hybrid.search("Azure 配置", top_k=5)

        # With sparse_weight=0, all sparse contributions are 0.
        # dense_1 should rank highest (rank 1 in dense, weight 1.0)
        assert results[0].chunk_id == "dense_1"

    def test_weights_in_config_yaml_parsing(self):
        """Test that weights are parsed from YAML dict via Settings.from_dict()."""
        from src.core.settings import Settings

        data = {
            "llm": {
                "provider": "ollama", "model": "m", "temperature": 0.0,
                "max_tokens": 100, "base_url": "http://localhost:11434",
            },
            "embedding": {
                "provider": "ollama", "model": "m", "dimensions": 768,
                "base_url": "http://localhost:11434",
            },
            "vector_store": {
                "provider": "chroma", "persist_directory": "./db",
                "collection_name": "c",
            },
            "retrieval": {
                "dense_top_k": 20, "sparse_top_k": 20,
                "fusion_top_k": 10, "rrf_k": 60,
                "dense_weight": 1.5, "sparse_weight": 0.8,
            },
            "rerank": {
                "enabled": False, "provider": "none", "model": "none",
                "top_k": 5,
            },
            "evaluation": {
                "enabled": False, "provider": "custom",
                "metrics": ["faithfulness"],
            },
            "observability": {
                "log_level": "INFO", "trace_enabled": False,
                "trace_file": "./t.jsonl", "structured_logging": False,
            },
        }

        settings = Settings.from_dict(data)
        assert settings.retrieval.dense_weight == 1.5
        assert settings.retrieval.sparse_weight == 0.8

    def test_settings_validation_rejects_negative_weight(self):
        """Test that negative weights are rejected by validate_settings()."""
        from src.core.settings import (
            RetrievalSettings, Settings, SettingsError, validate_settings,
        )

        # Build a minimal valid Settings with negative dense_weight
        settings = _make_minimal_settings(dense_weight=-0.5, sparse_weight=1.0)

        with pytest.raises(SettingsError, match="dense_weight"):
            validate_settings(settings)

    def test_settings_validation_rejects_both_zero(self):
        """Test that both weights = 0 is rejected."""
        from src.core.settings import SettingsError, validate_settings

        settings = _make_minimal_settings(dense_weight=0.0, sparse_weight=0.0)

        with pytest.raises(SettingsError, match="(?i)at least one"):
            validate_settings(settings)

    def test_settings_validation_accepts_one_zero(self):
        """Test that one weight = 0 is valid (disables that channel)."""
        from src.core.settings import validate_settings

        settings = _make_minimal_settings(dense_weight=0.0, sparse_weight=1.0)
        validate_settings(settings)  # Should not raise

        settings = _make_minimal_settings(dense_weight=1.0, sparse_weight=0.0)
        validate_settings(settings)  # Should not raise


def _make_minimal_settings(
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> "Settings":
    """Helper to build a minimal valid Settings for weight validation tests."""
    from src.core.settings import (
        EvaluationSettings,
        LLMSettings,
        EmbeddingSettings,
        VectorStoreSettings,
        RetrievalSettings,
        RerankSettings,
        ObservabilitySettings,
        Settings,
    )
    return Settings(
        llm=LLMSettings(provider="ollama", model="m", temperature=0.0, max_tokens=100),
        embedding=EmbeddingSettings(provider="ollama", model="m", dimensions=768),
        vector_store=VectorStoreSettings(
            provider="chroma", persist_directory="./db", collection_name="c",
        ),
        retrieval=RetrievalSettings(
            dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60,
            dense_weight=dense_weight, sparse_weight=sparse_weight,
        ),
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(
            enabled=False, provider="custom", metrics=["faithfulness"],
        ),
        observability=ObservabilitySettings(
            log_level="INFO", trace_enabled=False,
            trace_file="./t.jsonl", structured_logging=False,
        ),
    )


# =============================================================================
# RRF Fusion Integration Tests
# =============================================================================


class TestRRFFusionIntegration:
    """Test actual RRF fusion behavior in HybridSearch."""

    def test_common_chunks_boosted_by_rrf(
        self,
        query_processor,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that chunks appearing in both results get boosted."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        fusion = RRFFusion(k=60)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=fusion,
        )

        results = hybrid.search("配置", top_k=10)

        chunk_ids = [r.chunk_id for r in results]
        if "common_chunk" in chunk_ids:
            common_idx = chunk_ids.index("common_chunk")
            assert common_idx < 5, "common_chunk should be boosted by RRF"

    def test_rrf_scores_are_deterministic(
        self,
        query_processor,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that RRF fusion produces deterministic results."""
        dense = MockDenseRetriever(results=sample_dense_results)
        sparse = MockSparseRetriever(results=sample_sparse_results)
        fusion = RRFFusion(k=60)

        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=fusion,
        )

        results1 = hybrid.search("配置", top_k=5)
        results2 = hybrid.search("配置", top_k=5)

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.chunk_id == r2.chunk_id
            assert r1.score == r2.score


# =============================================================================
# Source Diversity (max_per_document)
# =============================================================================


class TestSourceDiversity:
    """Test max_per_document source diversity filtering."""

    def test_diversify_limits_chunks_per_source(self):
        """3 chunks from same PDF → only 2 kept with max_per_document=2."""
        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="chunk1",
                            metadata={"source_path": "a.pdf"}),
            RetrievalResult(chunk_id="c2", score=0.8, text="chunk2",
                            metadata={"source_path": "a.pdf"}),
            RetrievalResult(chunk_id="c3", score=0.7, text="chunk3",
                            metadata={"source_path": "a.pdf"}),
            RetrievalResult(chunk_id="c4", score=0.6, text="chunk4",
                            metadata={"source_path": "b.pdf"}),
        ]

        diversified = HybridSearch._diversify_results(results, max_per_document=2)

        assert len(diversified) == 3
        sources = [r.metadata["source_path"] for r in diversified]
        assert sources.count("a.pdf") == 2
        assert sources.count("b.pdf") == 1
        # Score order preserved
        assert diversified[0].chunk_id == "c1"
        assert diversified[1].chunk_id == "c2"
        assert diversified[2].chunk_id == "c4"

    def test_diversify_zero_means_no_limit(self):
        """max_per_document=0 returns all results unchanged."""
        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="a",
                            metadata={"source_path": "a.pdf"}),
            RetrievalResult(chunk_id="c2", score=0.8, text="b",
                            metadata={"source_path": "a.pdf"}),
            RetrievalResult(chunk_id="c3", score=0.7, text="c",
                            metadata={"source_path": "a.pdf"}),
        ]

        diversified = HybridSearch._diversify_results(results, max_per_document=0)

        assert len(diversified) == 3

    def test_diversify_with_max_1(self):
        """max_per_document=1 ensures unique sources."""
        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="a",
                            metadata={"source_path": "a.pdf"}),
            RetrievalResult(chunk_id="c2", score=0.8, text="b",
                            metadata={"source_path": "a.pdf"}),
            RetrievalResult(chunk_id="c3", score=0.7, text="c",
                            metadata={"source_path": "b.pdf"}),
            RetrievalResult(chunk_id="c4", score=0.6, text="d",
                            metadata={"source_path": "c.pdf"}),
        ]

        diversified = HybridSearch._diversify_results(results, max_per_document=1)

        assert len(diversified) == 3
        sources = [r.metadata["source_path"] for r in diversified]
        assert len(set(sources)) == 3  # all unique

    def test_diversify_preserves_score_order(self):
        """Diversified results maintain descending score order."""
        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="a",
                            metadata={"source_path": "x.pdf"}),
            RetrievalResult(chunk_id="c2", score=0.85, text="b",
                            metadata={"source_path": "x.pdf"}),
            RetrievalResult(chunk_id="c3", score=0.8, text="c",
                            metadata={"source_path": "y.pdf"}),
            RetrievalResult(chunk_id="c4", score=0.75, text="d",
                            metadata={"source_path": "x.pdf"}),
            RetrievalResult(chunk_id="c5", score=0.7, text="e",
                            metadata={"source_path": "y.pdf"}),
        ]

        diversified = HybridSearch._diversify_results(results, max_per_document=1)

        scores = [r.score for r in diversified]
        assert scores == sorted(scores, reverse=True)

    def test_diversify_empty_results(self):
        """Empty input returns empty output."""
        assert HybridSearch._diversify_results([], max_per_document=2) == []

    def test_config_reads_max_per_document(self):
        """HybridSearchConfig should include max_per_document."""
        config = HybridSearchConfig(max_per_document=2)
        assert config.max_per_document == 2

    def test_config_default_no_limit(self):
        """Default max_per_document is 0 (no limit)."""
        config = HybridSearchConfig()
        assert config.max_per_document == 0
