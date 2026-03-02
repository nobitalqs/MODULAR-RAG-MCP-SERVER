# Advanced Features Design

**Date**: 2026-03-02
**Status**: Approved
**Scope**: 6 new features inspired by Ragent (nageoffer/ragent) analysis

## Overview

Add 6 advanced capabilities to the Modular RAG MCP Server, organized in 3 layers:

1. **Infrastructure Layer**: Redis cache, rate limiter (foundation for other features)
2. **Query Enhancement Layer**: query rewriting, query router (improve retrieval quality)
3. **Reliability Layer**: session memory, circuit breaker + provider failover

All features follow the existing Registry + Factory pattern and are **configuration-driven** with sensible defaults. Redis dependency is optional — all features degrade gracefully to in-process implementations.

## Feature 1: Cache Infrastructure

### Files

```
src/libs/cache/
├── __init__.py
├── base_cache.py          # BaseCache(ABC): get/set/delete/exists
├── cache_factory.py       # CacheFactory: memory | redis
├── memory_cache.py        # InMemoryCache: OrderedDict LRU + TTL
├── redis_cache.py         # RedisCache: redis-py + pickle serialization
└── embedding_cache.py     # EmbeddingCache: decorator for BaseEmbedding
```

### Configuration

```yaml
cache:
  provider: "memory"             # memory | redis
  redis_url: "redis://localhost:6379/0"
  default_ttl: 3600
  max_memory_items: 10000
```

### Interface

```python
class BaseCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    @abstractmethod
    def delete(self, key: str) -> bool: ...
    @abstractmethod
    def exists(self, key: str) -> bool: ...
```

### EmbeddingCache

Wraps any `BaseEmbedding` implementation. Cache key = `emb:{model}:{sha256(text)[:16]}`.
On `embed(texts)`:
1. Split texts into cache hits and misses
2. Call underlying `embedding.embed()` only for misses
3. Write results back to cache
4. Reassemble in original order

## Feature 2: Rate Limiter

### Files

```
src/libs/rate_limiter/
├── __init__.py
├── base_limiter.py        # BaseLimiter(ABC): acquire/release
├── token_bucket.py        # TokenBucketLimiter: in-process
└── redis_limiter.py       # RedisLimiter: Redis ZSET sliding window
```

### Configuration

```yaml
rate_limit:
  enabled: true
  provider: "token_bucket"       # token_bucket | redis
  requests_per_minute: 60
  tokens_per_minute: 100000      # optional TPM limit
  max_concurrent: 10
```

### Token Bucket Algorithm

```python
class TokenBucketLimiter:
    capacity: int            # bucket capacity
    tokens: float            # current available tokens
    refill_rate: float       # tokens per second = rpm / 60
    last_refill: float       # monotonic timestamp

    def acquire(self, timeout: float = 30.0) -> bool:
        # Refill: tokens = min(tokens + elapsed * refill_rate, capacity)
        # If tokens >= 1: consume one, return True
        # Else: wait up to timeout or raise RateLimitExceeded
```

### Redis Sliding Window (for distributed mode)

Uses ZSET with score = timestamp, member = request_id.
Lua script atomically: count requests in window, add if under limit, clean expired.

### Integration

Applied as middleware/decorator on `BaseLLM.chat()` calls. All providers share the same limiter instance.

## Feature 3: Query Rewriting

### Files

```
src/libs/query_rewriter/
├── __init__.py
├── base_rewriter.py       # BaseQueryRewriter(ABC)
├── rewriter_factory.py    # QueryRewriterFactory
├── none_rewriter.py       # NoneRewriter: passthrough
├── llm_rewriter.py        # LLMRewriter: context completion + rewrite
└── hyde_rewriter.py       # HyDERewriter: hypothetical document embeddings
```

### Configuration

```yaml
query_rewriting:
  enabled: true
  provider: "llm"                # none | llm | hyde
  model: null                    # null = reuse llm.provider config
  max_rewrites: 3                # max sub-questions for decomposition
```

### Data Structures

```python
@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    rewritten_queries: list[str]    # 1 or more rewritten queries
    reasoning: str | None           # LLM reasoning (debug)
    strategy: str                   # "none" | "rewrite" | "decompose" | "hyde"

class BaseQueryRewriter(ABC):
    @abstractmethod
    def rewrite(
        self,
        query: str,
        conversation_history: list[Message] | None = None,
        trace: Any = None,
    ) -> RewriteResult: ...
```

### LLMRewriter Strategies

1. **Context completion**: When conversation_history provided, rewrite query to be self-contained
2. **Decomposition**: When query contains multiple sub-questions, split into 1-3 independent queries
3. **Simple rewrite**: Optimize query wording for semantic retrieval

### HyDERewriter

1. LLM generates a hypothetical ideal answer document
2. Returns strategy="hyde" in RewriteResult
3. HybridSearch uses the hypothetical document's embedding for retrieval

### Integration Point

Inserted between `QueryProcessor.process()` and `HybridSearch.search()` in the query flow.

## Feature 4: Query Router (Intent Classification)

### Files

```
src/libs/query_router/
├── __init__.py
├── base_router.py         # BaseQueryRouter(ABC)
├── router_factory.py      # QueryRouterFactory
├── none_router.py         # NoneRouter: always returns knowledge_search
└── llm_router.py          # LLMRouter: LLM-based classification
```

### Configuration

```yaml
query_routing:
  enabled: false                  # disabled by default (MCP client handles routing)
  provider: "llm"                # none | llm
  model: null
  routes:
    - name: "knowledge_search"
      description: "User wants to retrieve information from knowledge base"
    - name: "tool_call"
      description: "User wants to execute an operation or call a tool"
    - name: "chitchat"
      description: "Casual conversation unrelated to knowledge base"
```

### Data Structures

```python
@dataclass(frozen=True)
class RouteDecision:
    route: str                # "knowledge_search" | "tool_call" | "chitchat"
    confidence: float         # 0.0 - 1.0
    tool_name: str | None     # only when route == "tool_call"
    tool_params: dict | None
    reasoning: str | None

class BaseQueryRouter(ABC):
    @abstractmethod
    def route(
        self,
        query: str,
        available_tools: list[str] | None = None,
        conversation_history: list[Message] | None = None,
        trace: Any = None,
    ) -> RouteDecision: ...
```

### Note

Disabled by default because MCP protocol delegates routing to the LLM client (Claude/Copilot). Useful when exposing a non-MCP entry point (e.g., HTTP API, Streamlit chat) or for demonstration/interview purposes.

## Feature 5: Session Memory

### Files

```
src/libs/memory/
├── __init__.py
├── base_memory.py         # BaseMemoryStore(ABC)
├── memory_factory.py      # MemoryFactory
├── memory_store.py        # InMemoryStore: OrderedDict + TTL
├── redis_memory.py        # RedisMemoryStore: Redis Hash
└── conversation_memory.py # ConversationMemory: business logic
```

### Configuration

```yaml
memory:
  enabled: true
  provider: "memory"             # memory | redis (reuses cache.redis_url)
  max_turns: 10                  # sliding window size
  summarize_threshold: 20        # trigger LLM summarization above this
  summarize_enabled: true        # false = truncate instead of summarize
  session_ttl: 3600              # session expiry in seconds
```

### Data Structures

```python
@dataclass(frozen=True)
class ConversationTurn:
    user_message: Message
    assistant_message: Message
    timestamp: float
    metadata: dict[str, Any]

@dataclass(frozen=True)
class SessionContext:
    session_id: str
    turns: list[ConversationTurn]
    summary: str | None
    total_turns: int

    def to_messages(self) -> list[Message]:
        """Convert to flat message list for LLM/rewriter consumption."""
        messages = []
        if self.summary:
            messages.append(Message("system", f"Previous conversation summary: {self.summary}"))
        for turn in self.turns:
            messages.append(turn.user_message)
            messages.append(turn.assistant_message)
        return messages
```

### ConversationMemory Logic

- `get_context(session_id)`: Returns last N turns + summary
- `add_turn(session_id, turn)`: Appends turn; if total > threshold and LLM available, compress
- `_compress()`: LLM summarizes old turns, stores summary, keeps only recent N turns in store

### Integration

`query_knowledge_hub` input schema gains `session_id: str | None`. When provided, memory context is injected into QueryRewriter for context-aware rewriting.

## Feature 6: Circuit Breaker + Provider Failover

### Files

```
src/libs/circuit_breaker/
├── __init__.py
├── circuit_breaker.py     # CircuitBreaker class + decorator
└── provider_chain.py      # ProviderChain: multi-provider failover
```

### Configuration

```yaml
llm:
  provider: "azure"
  # ... existing config ...
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    cooldown_seconds: 60
    half_open_max_calls: 1
  fallback_providers:
    - provider: "openai"
      model: "gpt-4o-mini"
    - provider: "ollama"
      model: "qwen2.5"
```

### CircuitBreaker (Three-State Machine)

States: CLOSED -> OPEN -> HALF_OPEN

```python
class CircuitBreaker:
    # CLOSED: all requests pass through
    #   -> consecutive failures >= threshold -> OPEN
    # OPEN: all requests rejected immediately (fast-fail)
    #   -> cooldown elapsed -> HALF_OPEN
    # HALF_OPEN: allow 1 probe request
    #   -> success -> CLOSED
    #   -> failure -> OPEN

    def allow_request(self) -> bool: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...
```

Thread-safe with `threading.Lock`.

### ProviderChain

```python
class ProviderChain:
    def __init__(self, providers: list[tuple[BaseLLM, CircuitBreaker]]):
        ...

    def chat(self, messages, **kwargs) -> ChatResponse:
        for llm, breaker in self._providers:
            if not breaker.allow_request():
                continue  # skip circuit-open provider
            try:
                result = llm.chat(messages, **kwargs)
                breaker.record_success()
                return result
            except Exception:
                breaker.record_failure()
        raise AllProvidersUnavailableError(...)
```

### Integration

`LLMFactory.create_with_failover(settings)` returns `ProviderChain` if `fallback_providers` configured, else single provider with circuit breaker wrapper.

## End-to-End Query Flow (After All Features)

```
User query + session_id
  -> [QueryRouter.route()]           # optional, default disabled
  -> ConversationMemory.get_context()
  -> QueryRewriter.rewrite()         # with conversation history
  -> QueryProcessor.process()        # existing: stopwords, keywords, filters
  -> RateLimiter.acquire()           # throttle before LLM/API calls
  -> HybridSearch.search()           # EmbeddingCache wraps embedding calls
      ├── DenseRetriever (cached embeddings)
      └── SparseRetriever (BM25)
  -> RRFFusion.fuse()
  -> CoreReranker.rerank()
  -> [LLM answer generation]         # via ProviderChain (circuit-protected)
  -> ConversationMemory.add_turn()
  -> Response
```

## Implementation Order

Based on dependency analysis:

| Phase | Feature | Depends On | New Files | Estimated Complexity |
|-------|---------|-----------|-----------|---------------------|
| J1 | Cache Infrastructure | None | 6 | Medium |
| J2 | Rate Limiter | J1 (Redis shared) | 4 | Medium |
| J3 | Circuit Breaker + Failover | None | 3 | Low-Medium |
| J4 | Query Rewriter | None (uses existing LLM) | 6 | Medium |
| J5 | Session Memory | J1 (Redis shared) | 6 | Medium |
| J6 | Query Router | None (uses existing LLM) | 5 | Low |

Total: ~30 new files across 6 sub-phases.

## Settings Summary

New settings.yaml sections:

```yaml
cache:
  provider: "memory"
  redis_url: "redis://localhost:6379/0"
  default_ttl: 3600
  max_memory_items: 10000

rate_limit:
  enabled: true
  provider: "token_bucket"
  requests_per_minute: 60
  max_concurrent: 10

query_rewriting:
  enabled: true
  provider: "llm"
  max_rewrites: 3

query_routing:
  enabled: false
  provider: "llm"
  routes:
    - name: "knowledge_search"
      description: "Retrieve from knowledge base"
    - name: "tool_call"
      description: "Execute tool or operation"
    - name: "chitchat"
      description: "Casual conversation"

memory:
  enabled: true
  provider: "memory"
  max_turns: 10
  summarize_threshold: 20
  summarize_enabled: true
  session_ttl: 3600

# Within existing llm section:
llm:
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    cooldown_seconds: 60
  fallback_providers: []
```

## Testing Strategy

### Existing Test Baseline

```
Current test pyramid (1369 total):

          /\
         /  \     E2E: 32 tests (3 files)
        /    \    test_data_ingestion, test_query_script, test_recall
       /------\
      /        \  Integration: 36 tests (4 files)
     /          \ chroma_roundtrip, hybrid_search, pipeline, mcp_server
    /------------\
   /              \ Unit: 1301 tests (59 files)
  /                \ factory, provider, contract, smoke...
 /==================\
   Ratio: 95 : 3 : 2
```

### New Feature Test Pyramid

```
New tests (~178 total):

          /\
         /  \     E2E: ~8 tests (2 files)
        /    \
       /------\
      /        \  Integration: ~20 tests (6 files)
     /          \
    /------------\
   /              \ Unit + Contract: ~150 tests (~20 files)
  /                \
 /==================\

Combined total: 1369 + 178 ≈ 1547 tests
```

### Unit Tests (Bottom Layer — ~150 tests)

Each feature follows TDD: write test first (RED) → implement (GREEN) → refactor (IMPROVE).

#### J1: Cache Infrastructure

| File | Tests | Covers |
|------|-------|--------|
| `tests/unit/test_memory_cache.py` | ~20 | LRU eviction order, TTL expiry, get/set/delete/exists, max_items boundary, concurrent access, empty cache edge cases |
| `tests/unit/test_redis_cache.py` | ~15 | Same as above + pickle serialization roundtrip for numpy arrays, connection failure graceful handling |
| `tests/unit/test_cache_factory.py` | ~8 | memory/redis provider creation, invalid provider error, config defaults, missing redis_url error |
| `tests/unit/test_embedding_cache.py` | ~15 | Cache hit/miss splitting, original order reassembly, cache key generation (sha256), partial hit (some cached some not), all-hit fast path, all-miss passthrough |

#### J2: Rate Limiter

| File | Tests | Covers |
|------|-------|--------|
| `tests/unit/test_token_bucket.py` | ~18 | Token consumption, refill calculation (time-based), capacity ceiling, burst after idle, timeout rejection (RateLimitExceeded), zero-capacity edge, concurrent acquire, refill_rate=rpm/60 math |
| `tests/unit/test_redis_limiter.py` | ~10 | Sliding window count, Lua script atomicity (mock), window expiry cleanup, request_id uniqueness |
| `tests/unit/test_limiter_factory.py` | ~5 | Provider selection, enabled=false returns NullLimiter, config parsing |

#### J3: Circuit Breaker + Failover

| File | Tests | Covers |
|------|-------|--------|
| `tests/unit/test_circuit_breaker.py` | ~20 | State transitions: CLOSED→OPEN (consecutive failures), OPEN→HALF_OPEN (cooldown elapsed), HALF_OPEN→CLOSED (probe success), HALF_OPEN→OPEN (probe failure); failure count reset on success; thread safety under concurrent calls; cooldown timing precision; allow_request() in each state |
| `tests/unit/test_provider_chain.py` | ~12 | First provider success (no failover), first fails → second succeeds, all providers circuit-open → AllProvidersUnavailableError, breaker state isolation between providers, chat() kwargs forwarding, single provider chain (no failover) |

#### J4: Query Rewriter

| File | Tests | Covers |
|------|-------|--------|
| `tests/unit/test_none_rewriter.py` | ~5 | Passthrough: rewritten_queries == [original], strategy="none", conversation_history ignored |
| `tests/unit/test_llm_rewriter.py` | ~15 | Context completion mode (with history), decomposition mode (complex query → sub-queries), simple rewrite mode, max_rewrites cap, LLM returns invalid JSON → fallback to original, empty query edge, rewrite result immutability |
| `tests/unit/test_hyde_rewriter.py` | ~8 | Hypothetical document generation, strategy="hyde" flag, rewritten_queries contains generated doc, LLM failure → fallback to original |
| `tests/unit/test_rewriter_factory.py` | ~6 | none/llm/hyde provider creation, enabled=false returns NoneRewriter, model=null reuses main LLM |

#### J5: Session Memory

| File | Tests | Covers |
|------|-------|--------|
| `tests/unit/test_memory_store.py` | ~15 | add_turn ordering, get_turns returns chronological, TTL expiry (session_ttl), clear by session_id, non-existent session returns empty, max turns storage, ConversationTurn immutability |
| `tests/unit/test_redis_memory.py` | ~10 | Same behavioral contract + Redis Hash serialization roundtrip, JSON encoding of Message objects, TTL set via EXPIRE |
| `tests/unit/test_conversation_memory.py` | ~15 | Sliding window truncation (max_turns=10 keeps last 10), summarize trigger (turns > threshold), _compress() LLM call with correct prompt, summary stored and retrievable, to_messages() format (system summary + turn pairs), summarize_enabled=false → truncate only, LLM unavailable → truncate fallback |
| `tests/unit/test_memory_factory.py` | ~5 | Provider selection, enabled=false behavior, redis provider reuses cache.redis_url |

#### J6: Query Router

| File | Tests | Covers |
|------|-------|--------|
| `tests/unit/test_none_router.py` | ~4 | Always returns route="knowledge_search", confidence=1.0, tool_name=None |
| `tests/unit/test_llm_router.py` | ~10 | Knowledge search classification, tool call classification (with tool_name extraction), chitchat classification, low confidence handling, LLM returns invalid JSON → default to knowledge_search, available_tools passed to prompt |
| `tests/unit/test_router_factory.py` | ~5 | Provider selection, enabled=false returns NoneRouter, routes config parsing |

### Contract Tests (Within Unit Layer — ~30 tests)

Contract tests verify all implementations of an ABC satisfy the same behavioral interface. Uses `@pytest.mark.parametrize` to run identical assertions against every provider.

| File | Tests | Validates |
|------|-------|-----------|
| `tests/unit/test_cache_contract.py` | ~8 | InMemoryCache and RedisCache both satisfy BaseCache: get after set returns value, get missing returns None, delete returns True/False, TTL=0 expires immediately, set overwrites previous |
| `tests/unit/test_rewriter_contract.py` | ~6 | NoneRewriter, LLMRewriter, HyDERewriter all return RewriteResult with required fields, handle empty query, handle None conversation_history |
| `tests/unit/test_memory_store_contract.py` | ~8 | InMemoryStore and RedisMemoryStore both satisfy BaseMemoryStore: add_turn/get_turns roundtrip, clear removes all turns, get_summary/set_summary roundtrip, empty session returns [] |
| `tests/unit/test_router_contract.py` | ~5 | NoneRouter and LLMRouter both return RouteDecision with valid route string, confidence in [0,1], handle None available_tools |
| `tests/unit/test_limiter_contract.py` | ~3 | TokenBucketLimiter and RedisLimiter both satisfy BaseLimiter: acquire returns bool, release is idempotent |

### Integration Tests (Middle Layer — ~20 tests)

Test cross-module interactions with real (or realistic mock) dependencies.

| File | Tests | Covers |
|------|-------|-----------|
| `tests/integration/test_cache_with_embedding.py` | ~4 | EmbeddingCache wrapping a real/mock embedding client: first call is cache miss (embedding called), second call is cache hit (embedding NOT called), verify vector values match |
| `tests/integration/test_cache_redis_roundtrip.py` | ~3 | Redis get/set/delete roundtrip with numpy arrays and large strings. `@pytest.mark.skipif(not redis_available(), reason="Redis not running")` |
| `tests/integration/test_rewriter_with_hybrid_search.py` | ~3 | LLMRewriter → QueryProcessor → HybridSearch pipeline: verify rewritten query returns different (better) results than original |
| `tests/integration/test_memory_with_rewriter.py` | ~3 | Multi-turn: add 3 turns → get_context → pass to rewriter → verify rewritten query includes context from earlier turns |
| `tests/integration/test_circuit_breaker_with_llm.py` | ~4 | Mock a failing LLM: verify breaker opens after N failures, ProviderChain falls over to backup, backup response is correct |
| `tests/integration/test_rate_limiter_with_llm.py` | ~3 | Concurrent LLM calls via ThreadPoolExecutor: verify limiter blocks excess requests, allowed requests succeed |

### E2E Tests (Top Layer — ~8 tests)

Full system tests that exercise the complete query pipeline.

| File | Tests | Covers |
|------|-------|-----------|
| `tests/e2e/test_mcp_with_session.py` | ~4 | Launch MCP server subprocess → send query_knowledge_hub with session_id → send follow-up query (same session_id) → verify second response reflects conversation context from first turn |
| `tests/e2e/test_full_query_pipeline.py` | ~4 | Full new pipeline: router(disabled) → memory → rewrite → process → search → rerank → response. Verify: rewrite trace recorded, cache hit on repeated query, rate limiter allows request, response contains citations |

### Test Conventions

1. **Redis tests**: All Redis-dependent tests use `@pytest.mark.skipif(not redis_available())`. CI without Redis skips gracefully.
2. **LLM mock strategy**: Unit tests mock `BaseLLM.chat()` to return deterministic JSON. Integration tests may use real LLM with `@pytest.mark.slow` marker.
3. **Fixture format**: JSON fixtures use Object (keyed by scenario name), each with `input`, `expected_*`, `note` fields per project convention.
4. **Immutability assertions**: Every frozen dataclass test includes `pytest.raises(FrozenInstanceError)` for mutation attempts.
5. **TDD workflow**: Each phase starts by writing failing tests (RED), then implements until green (GREEN), then refactors (IMPROVE).
6. **Coverage target**: Maintain ≥ 80% coverage per new module. Run `pytest --cov=src/libs/cache --cov=src/libs/rate_limiter --cov=src/libs/circuit_breaker --cov=src/libs/query_rewriter --cov=src/libs/memory --cov=src/libs/query_router`.
