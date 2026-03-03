"""MCP Tool: query_knowledge_hub

This tool provides knowledge retrieval capabilities through the MCP protocol.
It combines HybridSearch (Dense + Sparse + RRF Fusion) with optional Reranking
to find relevant documents and return formatted results with citations.

Usage via MCP:
    Tool name: query_knowledge_hub
    Input schema:
        - query (string, required): The search query
        - top_k (integer, optional): Number of results to return (default: 5)
        - collection (string, optional): Limit search to specific collection
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

from src.core.response.response_builder import MCPToolResponse, ResponseBuilder
from src.core.settings import Settings, load_settings, resolve_path
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import CoreReranker

logger = logging.getLogger(__name__)


# Tool metadata
TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = """Search the knowledge base for relevant documents.

This tool uses hybrid search (semantic + keyword) to find the most relevant
documents matching your query. Results include source citations for reference.

Parameters:
- query: Your search question or keywords
- top_k: Maximum number of results (default: 5)
- collection: Limit search to a specific document collection
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query or question to find relevant documents for.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        "collection": {
            "type": "string",
            "description": "Optional collection name to limit the search scope.",
        },
        "session_id": {
            "type": "string",
            "description": "Optional session ID for conversation memory.",
        },
    },
    "required": ["query"],
}


@dataclass
class QueryKnowledgeHubConfig:
    """Configuration for query_knowledge_hub tool.

    Attributes:
        default_top_k: Default number of results if not specified
        max_top_k: Maximum allowed top_k value
        default_collection: Default collection if not specified
        enable_rerank: Whether to apply reranking
    """

    default_top_k: int = 5
    max_top_k: int = 20
    default_collection: str = "default"
    enable_rerank: bool = True


class QueryKnowledgeHubTool:
    """MCP Tool for knowledge base queries.

    This class encapsulates the query_knowledge_hub tool logic,
    coordinating HybridSearch and Reranker to produce formatted results.

    Design Principles:
    - Lazy initialization: Components created on first use
    - Error resilience: Graceful handling of search/rerank failures
    - Configurable: All parameters from settings.yaml

    Example:
        >>> tool = QueryKnowledgeHubTool(settings)
        >>> result = await tool.execute(query="Azure \u914d\u7f6e", top_k=5)
        >>> print(result.content)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[QueryKnowledgeHubConfig] = None,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[CoreReranker] = None,
        response_builder: Optional[ResponseBuilder] = None,
    ) -> None:
        """Initialize QueryKnowledgeHubTool.

        Args:
            settings: Application settings. If None, loaded from default path.
            config: Tool configuration. If None, uses defaults.
            hybrid_search: Optional pre-configured HybridSearch instance.
            reranker: Optional pre-configured CoreReranker instance.
            response_builder: Optional pre-configured ResponseBuilder instance.
        """
        self._settings = settings
        self.config = config or QueryKnowledgeHubConfig()
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._embedding_client = None
        self._response_builder = response_builder or ResponseBuilder()

        # Track initialization state
        self._initialized = False
        self._current_collection: Optional[str] = None

        # Phase J: Advanced components (lazy-initialized)
        self._conversation_memory = None
        self._query_rewriter = None
        self._rate_limiter = None
        self._query_router = None
        self._advanced_initialized = False

    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary."""
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def _init_advanced_components(self) -> None:
        """Lazy-init Phase J components from settings (idempotent)."""
        if self._advanced_initialized:
            return
        self._advanced_initialized = True

        s = self.settings

        # Rate limiter
        if s.rate_limit is not None:
            from src.libs.rate_limiter.limiter_factory import RateLimiterFactory
            self._rate_limiter = RateLimiterFactory.create_from_settings(s.rate_limit)

        # Query rewriter
        if s.query_rewriting is not None:
            from src.libs.query_rewriter.rewriter_factory import QueryRewriterFactory
            llm = None
            if s.query_rewriting.enabled and s.query_rewriting.provider != "none":
                from src.libs.llm.llm_factory import LLMFactory
                llm = LLMFactory.create_llm(s)
            self._query_rewriter = QueryRewriterFactory.create_from_settings(
                s.query_rewriting, llm=llm,
            )

        # Conversation memory
        if s.memory is not None and s.memory.enabled:
            from src.libs.memory.memory_factory import MemoryFactory
            from src.libs.memory.conversation_memory import ConversationMemory
            redis_url = s.cache.redis_url if s.cache else None
            store = MemoryFactory.create_from_settings(s.memory, redis_url=redis_url)
            llm = None
            if s.memory.summarize_enabled:
                from src.libs.llm.llm_factory import LLMFactory
                llm = LLMFactory.create_llm(s)
            self._conversation_memory = ConversationMemory(
                store=store,
                max_turns=s.memory.max_turns,
                summarize_threshold=s.memory.summarize_threshold,
                summarize_enabled=s.memory.summarize_enabled,
                llm=llm,
            )

        # Query router
        if s.query_routing is not None:
            from src.libs.query_router.router_factory import QueryRouterFactory
            llm = None
            if s.query_routing.enabled and s.query_routing.provider != "none":
                from src.libs.llm.llm_factory import LLMFactory
                llm = LLMFactory.create_llm(s)
            self._query_router = QueryRouterFactory.create_from_settings(
                s.query_routing, llm=llm,
            )

    def _ensure_initialized(self, collection: str) -> None:
        """Ensure search components are initialized for the given collection.

        Caching strategy (balances speed vs freshness):
        - **Fully cached** (stateless, never go stale): embedding client,
          reranker, query processor, settings.
        - **Cached until collection changes**: vector store (ChromaDB
          PersistentClient reads from SQLite \u2014 sees data written by other
          processes), dense retriever, hybrid search.
        - **Auto-refreshes on every query**: BM25 sparse index \u2014 the
          ``SparseRetriever._ensure_index_loaded()`` always reloads from
          disk, so the cached SparseRetriever object is fine.

        Only when *collection* changes do we tear down and rebuild.

        Args:
            collection: Target collection name.
        """
        # Fast path: already initialised for the same collection
        if self._initialized and self._current_collection == collection:
            logger.debug(
                "Query components already initialised for collection: %s",
                collection,
            )
            return

        logger.info("Initializing query components for collection: %s", collection)

        # Import here to avoid circular imports and allow lazy loading
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.reranker import create_core_reranker
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        # === Fully cached components (stateless, never go stale) ===
        if self._embedding_client is None:
            from src.libs.embedding import AzureEmbedding, OllamaEmbedding, OpenAIEmbedding

            emb_factory = EmbeddingFactory()
            emb_factory.register_provider("openai", OpenAIEmbedding)
            emb_factory.register_provider("azure", AzureEmbedding)
            emb_factory.register_provider("ollama", OllamaEmbedding)
            self._embedding_client = emb_factory.create_from_settings(
                self.settings.embedding,
            )

        if self._reranker is None:
            self._reranker = create_core_reranker(settings=self.settings)

        # === Rebuild for new collection ===
        from src.libs.vector_store import ChromaStore

        vs_factory = VectorStoreFactory()
        vs_factory.register_provider("chroma", ChromaStore)
        # Use settings defaults but allow collection override
        vector_store = vs_factory.create(
            self.settings.vector_store.provider,
            persist_directory=self.settings.vector_store.persist_directory,
            collection_name=collection,
        )

        dense_retriever = create_dense_retriever(
            settings=self.settings,
            embedding_client=self._embedding_client,
            vector_store=vector_store,
        )

        bm25_indexer = BM25Indexer(
            index_dir=str(resolve_path(f"data/db/bm25/{collection}"))
        )
        sparse_retriever = create_sparse_retriever(
            settings=self.settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        query_processor = QueryProcessor()
        self._hybrid_search = create_hybrid_search(
            settings=self.settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

        self._current_collection = collection
        self._initialized = True
        logger.info("Query components initialized for collection: %s", collection)

    async def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> MCPToolResponse:
        """Execute the query_knowledge_hub tool.

        Args:
            query: Search query string.
            top_k: Maximum results to return.
            collection: Target collection name.
            session_id: Optional session ID for conversation memory.

        Returns:
            MCPToolResponse with formatted content and citations.

        Raises:
            ValueError: If query is empty or invalid.
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Apply defaults
        effective_top_k = min(
            top_k or self.config.default_top_k,
            self.config.max_top_k
        )
        effective_collection = collection or self.config.default_collection

        logger.info(
            "Executing query_knowledge_hub: query='%s...', "
            "top_k=%d, collection=%s, session_id=%s",
            query[:50], effective_top_k, effective_collection, session_id,
        )

        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = query[:200]
        trace.metadata["top_k"] = effective_top_k
        trace.metadata["collection"] = effective_collection
        trace.metadata["source"] = "mcp"
        if session_id:
            trace.metadata["session_id"] = session_id

        try:
            # Initialize components for collection
            # Run blocking I/O in a thread to avoid blocking the async event loop
            import time as _time
            _init_t0 = _time.monotonic()
            await asyncio.to_thread(self._ensure_initialized, effective_collection)
            self._init_advanced_components()
            _init_elapsed = (_time.monotonic() - _init_t0) * 1000.0
            trace.record_stage("initialization", {
                "collection": effective_collection,
                "cold_start": _init_elapsed > 500,
            }, elapsed_ms=_init_elapsed)

            # Phase J: Rate limiter acquire
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()

            # Phase J: Get conversation context
            search_query = query
            conversation_history = None
            if session_id and self._conversation_memory is not None:
                ctx = self._conversation_memory.get_context(session_id)
                trace.metadata["memory_turns"] = len(ctx.turns)
                conversation_history = self._conversation_memory.to_messages(session_id)

            # Phase J: Query rewriting \u2192 multi-query retrieval
            search_queries: list[str] = [search_query]
            if self._query_rewriter is not None:
                try:
                    rewrite_result = self._query_rewriter.rewrite(
                        query, conversation_history=conversation_history,
                    )
                    search_queries = list(rewrite_result.rewritten_queries)
                    if search_queries != [query]:
                        trace.metadata["rewritten_queries"] = [
                            q[:200] for q in search_queries
                        ]
                except Exception as exc:
                    logger.warning("Query rewriter failed, using original: %s", exc)
                    search_queries = [query]

            # Multi-query fan-out: parallel hybrid search per sub-query
            per_query_results: list[list[RetrievalResult]] = list(
                await asyncio.gather(
                    *(
                        asyncio.to_thread(
                            self._perform_search, q, effective_top_k, trace,
                        )
                        for q in search_queries
                    )
                )
            )

            # Cross-query RRF fusion when multiple sub-queries
            if len(per_query_results) > 1:
                from src.core.query_engine.fusion import RRFFusion

                fusion = RRFFusion(k=60)
                rerank_budget = effective_top_k * 2
                results = fusion.fuse(per_query_results, top_k=rerank_budget)
                trace.metadata["multi_query_counts"] = [
                    len(r) for r in per_query_results
                ]
            else:
                results = per_query_results[0] if per_query_results else []

            # Apply reranking if enabled (may call LLM API)
            # Use original query for reranking \u2014 rank by user intent, not sub-queries
            if self.config.enable_rerank and results:
                results = await asyncio.to_thread(
                    self._apply_rerank, query, results, effective_top_k, trace,
                )

            # Build response
            response = self._response_builder.build(
                results=results,
                query=query,
                collection=effective_collection,
            )

            # Store final results in trace for dashboard display
            trace.metadata["final_results"] = [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "text": r.text or "",
                    "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                    "title": r.metadata.get("title", ""),
                }
                for r in results
            ]

            # Phase J: Store conversation turn
            if session_id and self._conversation_memory is not None:
                try:
                    self._conversation_memory.add_turn(session_id, "user", query)
                    summary = response.content[:500] if response.content else ""
                    self._conversation_memory.add_turn(
                        session_id, "assistant", summary,
                    )
                except Exception as exc:
                    logger.warning("Memory add_turn failed: %s", exc)

            # Phase J: Rate limiter release
            if self._rate_limiter is not None:
                self._rate_limiter.release()

            logger.info(
                "query_knowledge_hub completed: %d results, is_empty=%s",
                len(results), response.is_empty,
            )

            # Collect trace (Phase F TraceCollector \u2014 optional)
            self._collect_trace(trace)
            return response

        except Exception as e:
            # Phase J: Rate limiter release on error
            if self._rate_limiter is not None:
                self._rate_limiter.release()
            logger.exception("query_knowledge_hub failed: %s", e)
            self._collect_trace(trace)
            return self._build_error_response(query, effective_collection, str(e))

    def _collect_trace(self, trace: TraceContext) -> None:
        """Persist trace if TraceCollector is available (Phase F).

        Falls back to trace.finish() if TraceCollector is not yet implemented.
        """
        try:
            from src.core.trace import TraceCollector
            TraceCollector().collect(trace)
        except (ImportError, AttributeError):
            trace.finish()

    def _perform_search(
        self,
        query: str,
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Perform hybrid search.

        Args:
            query: Search query.
            top_k: Maximum results.
            trace: Optional TraceContext for observability.

        Returns:
            List of RetrievalResult.
        """
        if self._hybrid_search is None:
            raise RuntimeError("HybridSearch not initialized")

        # Use a larger initial retrieval for reranking
        initial_top_k = top_k * 2 if self.config.enable_rerank else top_k

        try:
            results = self._hybrid_search.search(
                query=query,
                top_k=initial_top_k,
                filters=None,
                trace=trace,
                return_details=False,
            )
            return results if isinstance(results, list) else results.results
        except Exception as e:
            logger.warning("Hybrid search failed: %s", e)
            return []

    def _apply_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Apply reranking to search results.

        Args:
            query: Original query.
            results: Search results to rerank.
            top_k: Final number of results.
            trace: Optional TraceContext for observability.

        Returns:
            Reranked results (or original if reranking fails).
        """
        if self._reranker is None or not self._reranker.is_enabled:
            return results[:top_k]

        try:
            rerank_result = self._reranker.rerank(
                query=query,
                results=results,
                top_k=top_k,
                trace=trace,
            )

            if rerank_result.used_fallback:
                logger.warning(
                    "Reranker fallback: %s", rerank_result.fallback_reason,
                )

            return rerank_result.results
        except Exception as e:
            logger.warning("Reranking failed, using original order: %s", e)
            return results[:top_k]

    def _build_error_response(
        self,
        query: str,
        collection: str,
        error_message: str,
    ) -> MCPToolResponse:
        """Build error response.

        Args:
            query: Original query.
            collection: Target collection.
            error_message: Error description.

        Returns:
            MCPToolResponse indicating error.
        """
        content = "## \u67e5\u8be2\u5931\u8d25\n\n"
        content += f"\u67e5\u8be2: **{query}**\n"
        content += f"\u96c6\u5408: `{collection}`\n\n"
        content += f"**\u9519\u8bef\u4fe1\u606f:** {error_message}\n\n"
        content += "\u8bf7\u68c0\u67e5:\n"
        content += "- \u6570\u636e\u5e93\u8fde\u63a5\u662f\u5426\u6b63\u5e38\n"
        content += "- \u96c6\u5408\u662f\u5426\u5df2\u521b\u5efa\u5e76\u5305\u542b\u6570\u636e\n"
        content += "- \u914d\u7f6e\u6587\u4ef6\u662f\u5426\u6b63\u786e\n"

        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={
                "query": query,
                "collection": collection,
                "error": error_message,
            },
            is_empty=True,
        )


# Module-level tool instance (lazy-initialized)
_tool_instance: Optional[QueryKnowledgeHubTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> QueryKnowledgeHubTool:
    """Get or create the tool instance.

    Args:
        settings: Optional settings to use for initialization.

    Returns:
        QueryKnowledgeHubTool instance.
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = QueryKnowledgeHubTool(settings=settings)
    return _tool_instance


async def query_knowledge_hub_handler(
    query: str,
    top_k: int = 5,
    collection: Optional[str] = None,
    session_id: Optional[str] = None,
) -> types.CallToolResult:
    """Handler function for MCP tool registration.

    This function is registered with the ProtocolHandler and called
    when the MCP client invokes the query_knowledge_hub tool.

    Args:
        query: Search query string.
        top_k: Maximum number of results.
        collection: Optional collection name.
        session_id: Optional session ID for conversation memory.

    Returns:
        MCP CallToolResult with content blocks.
    """
    tool = get_tool_instance()

    try:
        response = await tool.execute(
            query=query,
            top_k=top_k,
            collection=collection,
            session_id=session_id,
        )

        # Use to_mcp_content() which handles multimodal (text + images)
        content_blocks = response.to_mcp_content()

        return types.CallToolResult(
            content=content_blocks,
            isError=response.is_empty and "error" in response.metadata,
        )

    except ValueError as e:
        # Invalid parameters
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"\u53c2\u6570\u9519\u8bef: {e}",
                )
            ],
            isError=True,
        )
    except Exception as e:
        # Internal error
        logger.exception("query_knowledge_hub handler error: %s", e)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text="\u5185\u90e8\u9519\u8bef: \u67e5\u8be2\u5904\u7406\u5931\u8d25",
                )
            ],
            isError=True,
        )


def register_tool(protocol_handler: Any) -> None:
    """Register query_knowledge_hub tool with the protocol handler.

    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=query_knowledge_hub_handler,
    )
    logger.info("Registered MCP tool: %s", TOOL_NAME)
