# Milvus Migration Design

> ChromaDB + BM25 pickle -> Milvus Standalone (dense + sparse unified)

## 1. Background & Motivation

- **Goal**: Decouple vector store from application (client-server architecture) for future distributed deployment
- **Current state**: ChromaDB embedded (single-process) + independent BM25 pickle file
- **Target state**: Milvus Standalone (Docker Compose) with built-in BM25 Function, single storage backend
- **Data scale**: ~417 chunks now, designed for million-level growth

## 2. Architecture

```
+-----------------------------+
|  Python App (MCP Server)    |
|  +------------------------+ |
|  | BaseVectorStore (ABC)  | |     pymilvus async (gRPC)
|  |  +- MilvusStore (new)  |-|---> Milvus Standalone -+
|  +------------------------+ |                        |
|  +------------------------+ |     +------------------+
|  | SparseRetriever        |-|---> | Milvus Collection |
|  |  (delegates to store)  | |     |  - dense field    |
|  +------------------------+ |     |  - sparse field   |
+-----------------------------+     |  (BM25 Function)  |
                                    +------------------+
Docker Compose:
  +---------+  +------+  +-------+
  | Milvus  |  | etcd |  | MinIO |
  | :19530  |  |:2379 |  | :9000 |
  +---------+  +------+  +-------+
```

### Deployment Modes (unified via `uri`)

| Mode | uri | Use Case |
|------|-----|----------|
| Lite | `./data/db/milvus.db` | Local dev, unit tests |
| Standalone | `http://localhost:19530` | Docker deployment, integration tests |
| Distributed | `http://milvus-proxy:19530` | Future K8s |

## 3. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Deployment | Standalone (Docker Compose) | Client-server separation, no K8s dependency |
| SDK | pymilvus async (`AsyncMilvusClient`) | MCP Server is async, avoid `asyncio.to_thread` |
| Dense index | HNSW | Suitable for current scale, can switch to IVF_FLAT later |
| Sparse index | Built-in BM25 Function + SPARSE_INVERTED_INDEX | Replaces BM25 pickle, equivalent scoring |
| Chinese tokenizer | jieba analyzer | Same quality as current jieba-based BM25 |
| Dense metric | COSINE | Consistent with current ChromaDB behavior |
| Sparse metric | IP (inner product) | Standard for BM25 sparse vectors |
| ChromaStore | Remove entirely | Milvus Lite replaces its local dev role |
| ABC interface | All methods async | One-step migration, no sync/async split |

## 4. Collection Schema

```python
fields = [
    FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=128),
    FieldSchema("text", DataType.VARCHAR, max_length=65535),
    FieldSchema("dense_vector", DataType.FLOAT_VECTOR, dim=768),
    FieldSchema("sparse_vector", DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema("source", DataType.VARCHAR, max_length=512),
    FieldSchema("doc_hash", DataType.VARCHAR, max_length=64),
    FieldSchema("chunk_index", DataType.INT64),
]

functions = [
    Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
        params={"analyzer_name": "jieba"},
    )
]

indexes = [
    Index("dense_vector",  IndexType.HNSW, metric_type="COSINE"),
    Index("sparse_vector", IndexType.SPARSE_INVERTED_INDEX, metric_type="IP"),
]
```

## 5. Async ABC Refactoring

### New BaseVectorStore Interface

```python
class BaseVectorStore(ABC):
    # -- Required (all providers must implement) --
    @abstractmethod
    async def upsert(self, records: list[dict], **kwargs) -> None: ...

    @abstractmethod
    async def query(self, vector: list[float], top_k: int = 10,
                    filters: dict | None = None, **kwargs) -> list[dict]: ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str], **kwargs) -> list[dict]: ...

    # -- Optional (default raise NotImplementedError) --
    async def delete(self, ids: list[str], **kwargs) -> None: ...
    async def clear(self, collection_name: str | None = None, **kwargs) -> None: ...
    async def count(self, collection_name: str | None = None) -> int: ...
    async def list_collections(self) -> list[str]: ...
    async def delete_by_metadata(self, filter_dict: dict, **kwargs) -> int: ...

    # -- New: sparse & hybrid search --
    async def sparse_query(self, text: str, top_k: int = 10,
                           filters: dict | None = None, **kwargs) -> list[dict]: ...
    async def hybrid_query(self, vector: list[float], text: str,
                           top_k: int = 10, filters: dict | None = None,
                           ranker: str = "rrf", **kwargs) -> list[dict]: ...
```

### Leaky Abstraction Fixes

| Leak Location | Current Code | Fix |
|---------------|-------------|-----|
| `document_manager.py:294` | `self.chroma.collection.get(where=...)` | Use `await self.store.delete_by_metadata()` or `get_by_ids` |
| `document_manager.py:304` | `self.chroma.collection.get(where=...)` | Same as above |
| `data_service.py:120` | `self._chroma.collection.get(...)` | Use ABC `get_by_ids` |
| `overview.py:33` | `store.list_collections()` + `hasattr` | Call ABC method directly |
| `overview.py:36` | `store.count()` + `hasattr` | Same |
| `document_manager.py:84` | param named `chroma_store` | Rename to `vector_store`, type `BaseVectorStore` |

## 6. VectorStoreSettings

```python
@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str              # "milvus"
    uri: str                   # "http://localhost:19530" or "./data/db/milvus.db"
    collection_name: str       # "knowledge_hub"
    dense_dim: int = 768       # embedding dimension
    sparse_analyzer: str = "jieba"  # BM25 tokenizer
```

### settings.yaml

```yaml
vector_store:
  provider: "milvus"
  uri: "http://localhost:19530"
  # uri: "./data/db/milvus.db"  # Lite mode for dev/test
  collection_name: "knowledge_hub"
  dense_dim: 768
  sparse_analyzer: "jieba"
```

## 7. Docker Compose

```yaml
services:
  milvus:
    image: milvusdb/milvus:v2.5-latest
    command: ["milvus", "run", "standalone"]
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - milvus_data:/var/lib/milvus
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000

  etcd:
    image: quay.io/coreos/etcd:v3.5.18
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
    volumes:
      - etcd_data:/etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 10s
      timeout: 5s
      retries: 5

  minio:
    image: minio/minio:latest
    command: minio server /data --console-address ":9001"
    ports:
      - "9001:9001"
    volumes:
      - minio_data:/data
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  milvus_data:
  etcd_data:
  minio_data:
```

## 8. SparseRetriever Refactoring

```
Before: query text -> jieba tokenize -> BM25 pickle search -> chunk_ids -> vector_store.get_by_ids()
After:  query text -> MilvusStore.sparse_query(text) -> results directly
```

The BM25Indexer module and pickle files are removed entirely.

## 9. Task Breakdown

### Phase 1: Infrastructure
- T1. Docker Compose + pymilvus dependency
- T2. VectorStoreSettings refactor (uri unification)
- T3. BaseVectorStore async ABC rewrite

### Phase 2: Core Implementation
- T4. MilvusStore implementation (upsert/query/get_by_ids/delete/clear)
- T5. MilvusStore management methods (count/list_collections/delete_by_metadata)
- T6. MilvusStore sparse_query + hybrid_query

### Phase 3: Upstream Refactoring
- T7. Factory registration + settings.yaml adaptation
- T8. SparseRetriever refactor (delegate to sparse_query)
- T9. DenseRetriever async adaptation
- T10. DocumentManager leak fix + async
- T11. Ingestion pipeline adaptation (vector_upserter + pipeline.py)

### Phase 4: Integration Layer
- T12. MCP tools refactor (query/delete/list_collections)
- T13. Dashboard refactor (data_service + overview)

### Phase 5: Cleanup
- T14. Remove ChromaStore + BM25 indexer + pickle-related code
- T15. Update requirements.txt (chromadb -> pymilvus)

### Phase 6: Verification
- T16. Full test suite fix + contract test adaptation
- T17. Re-ingest test documents + golden test set validation
- T18. Docker Compose end-to-end smoke test

## 10. Testing Strategy

| Level | Method | Target |
|-------|--------|--------|
| Unit | Milvus Lite (embedded, no Docker) | All MilvusStore methods, schema creation, BM25 Function |
| Contract | `@pytest.mark.parametrize` over MilvusStore | ABC contract (upsert -> query roundtrip) |
| Integration | Milvus Standalone (Docker) | End-to-end ingest -> query |
| Golden test | Re-ingest 3 docs -> run 16 golden queries | Recall@5 >= 67% (current baseline) |

## 11. Files Changed

| Action | File | Description |
|--------|------|-------------|
| Create | `src/libs/vector_store/milvus_store.py` | MilvusStore implementation |
| Create | `docker-compose.yml` | Milvus Standalone deployment |
| Rewrite | `src/libs/vector_store/base_vector_store.py` | async ABC + new methods |
| Rewrite | `src/libs/vector_store/__init__.py` | Export MilvusStore |
| Modify | `src/libs/vector_store/vector_store_factory.py` | Register milvus |
| Modify | `src/core/settings.py` | VectorStoreSettings refactor |
| Modify | `config/settings.yaml` | New config format |
| Refactor | `src/core/query_engine/sparse_retriever.py` | Delegate to sparse_query |
| Refactor | `src/core/query_engine/dense_retriever.py` | await store.query() |
| Refactor | `src/ingestion/document_manager.py` | Fix leaks + async |
| Refactor | `src/ingestion/pipeline.py` | Remove ChromaStore refs |
| Refactor | `src/ingestion/storage/vector_upserter.py` | async upsert |
| Refactor | `src/mcp_server/tools/query_knowledge_hub.py` | Use factory |
| Refactor | `src/mcp_server/tools/delete_document.py` | Use factory |
| Refactor | `src/mcp_server/tools/list_collections.py` | Use ABC method |
| Refactor | `src/observability/dashboard/services/data_service.py` | Fix leaks |
| Refactor | `src/observability/dashboard/pages/overview.py` | Remove hasattr |
| Delete | `src/libs/vector_store/chroma_store.py` | Replaced by MilvusStore |
| Delete | BM25 indexer related files | Milvus BM25 Function takes over |

## 12. Rollback Plan

- Original PDF/text files remain intact — re-ingest to ChromaDB anytime
- ChromaStore is deleted from code but recoverable from git history
- Phase 1-2 are pure additions, no existing functionality broken
- Phase 5 (deletion) happens last, only after full verification
