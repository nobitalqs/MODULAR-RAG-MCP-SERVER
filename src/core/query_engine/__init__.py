"""
Query Engine - Hybrid search, fusion, and reranking.

Components:
- QueryProcessor: Query preprocessing and keyword extraction
- HybridSearch: Dense + Sparse parallel retrieval
- DenseRetriever / SparseRetriever: Individual retrievers
- RRFFusion: Reciprocal Rank Fusion
- CoreReranker: Reranking with fallback
"""

from src.core.query_engine.query_processor import (
    QueryProcessor,
    QueryProcessorConfig,
    create_query_processor,
)
from src.core.query_engine.dense_retriever import (
    DenseRetriever,
    create_dense_retriever,
)
from src.core.query_engine.sparse_retriever import (
    SparseRetriever,
    create_sparse_retriever,
)
from src.core.query_engine.fusion import RRFFusion, rrf_score
from src.core.query_engine.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
    HybridSearchResult,
    create_hybrid_search,
)
from src.core.query_engine.reranker import (
    CoreReranker,
    RerankConfig,
    RerankResult,
    create_core_reranker,
)

__all__ = [
    "QueryProcessor",
    "QueryProcessorConfig",
    "create_query_processor",
    "DenseRetriever",
    "create_dense_retriever",
    "SparseRetriever",
    "create_sparse_retriever",
    "RRFFusion",
    "rrf_score",
    "HybridSearch",
    "HybridSearchConfig",
    "HybridSearchResult",
    "create_hybrid_search",
    "CoreReranker",
    "RerankConfig",
    "RerankResult",
    "create_core_reranker",
]
