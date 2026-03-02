# Document Update Strategy Design

> Date: 2026-02-28
> Status: Approved
> Scope: Pipeline auto-cleanup on re-ingest + MCP delete tool with user confirmation

---

## 1. Problem Statement

When a user modifies an already-ingested document and re-runs ingestion:

1. **file_hash changes** (entire file SHA256) -> integrity checker sees a new file
2. **Pipeline runs all 6 stages** -> generates new chunk_ids (because content_hash differs)
3. **Old chunks remain** in ChromaDB + BM25 -> orphan data, never cleaned up
4. **Queries return duplicates** -> both old and new versions retrieved

Additionally, there is **no MCP tool or CLI command** for users to delete a document and its associated RAG data. The `DocumentManager.delete_document()` exists internally but has no user-facing interface.

---

## 2. Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Keep current chunk_id format** (`{source_hash}_{index:04d}_{content_hash}`) | Stability first. Project is progressing normally. No refactoring risk. |
| **Delete-then-reingest** (not incremental diff) | Low-frequency modifications. Full re-ingest cost is acceptable. Simple and reliable. |
| **Atomic deletion** (all stores or none) | Partial deletion (e.g., delete ChromaDB but keep BM25) causes data inconsistency across retrieval paths. Users should not need to know about internal storage layers. |
| **Embedding cache as future optimization** | Complementary to delete-then-reingest, not conflicting. Can be added later without modifying any current logic. |

---

## 3. Scope

### In Scope (Phase 1 - Current)

- **Scenario A**: MCP delete tool with two-phase confirmation
- **Scenario B**: Pipeline auto-detect and clean old version on re-ingest
- Supporting change: `FileIntegrity.lookup_by_path()` method

### Out of Scope (Future)

- chunk_id format refactoring (heading_path + relative_index)
- Differential embedding cache (`EmbeddingCache`)
- CLI delete script (can be added trivially once MCP tool works)

---

## 4. Detailed Design

### 4.1 FileIntegrity: New `lookup_by_path()` Method

**File**: `src/libs/loader/file_integrity.py`

**Purpose**: Given a source_path and optional collection, find the file_hash of the previously ingested version.

```python
def lookup_by_path(
    self, file_path: str, collection: str | None = None
) -> str | None:
    """Find file_hash of previously ingested version by source path.

    Args:
        file_path: Absolute path to the document.
        collection: Optional collection filter to avoid cross-collection interference.

    Returns:
        file_hash of the old version if found, None otherwise.
    """
    query = (
        "SELECT file_hash FROM ingestion_history "
        "WHERE file_path = ? AND status = 'success'"
    )
    params: list[str] = [file_path]
    if collection is not None:
        query += " AND collection = ?"
        params.append(collection)
    query += " ORDER BY updated_at DESC LIMIT 1"
    # execute and return first result or None
```

**Why collection filter matters**: The same file can theoretically be ingested into different collections. Modifying and re-ingesting into collection A must not accidentally delete data from collection B.

---

### 4.2 Pipeline Stage 1: Old Version Detection and Auto-Cleanup

**File**: `src/ingestion/pipeline.py`

**Change location**: After `should_skip()` returns False, before Stage 2.

#### Flow

```
Stage 1: Integrity Check
  │
  ├─ file_hash == old_hash  → should_skip() == True → return early (unchanged)
  │
  └─ file_hash != old_hash (or new file)
       │
       ├─ lookup_by_path(source_path, collection)
       │    │
       │    ├─ Returns old_hash (old_hash != file_hash)
       │    │    → Log: "Detected modified document, cleaning old version"
       │    │    → DocumentManager.delete_document(old_hash)
       │    │    → Record in stages["integrity"]["old_version_cleaned"]
       │    │
       │    └─ Returns None
       │         → First-time ingestion, no cleanup needed
       │
       └─ Proceed to Stage 2-6
```

#### Constructor Change

Pipeline `__init__` must create a `DocumentManager` instance:

```python
# In IngestionPipeline.__init__(), after creating storage components:
from src.ingestion.document_manager import DocumentManager

self.document_manager = DocumentManager(
    chroma_store=vector_store,
    bm25_indexer=self.bm25_indexer,
    image_storage=self.image_storage,
    file_integrity=self.integrity_checker,
)
```

#### Error Handling

- Cleanup failure **does not abort** the pipeline
- Logged as WARNING, recorded in trace
- Worst case: orphan data remains (same as current behavior, no regression)

#### PipelineResult Extension

Add optional field to report cleanup:

```python
@dataclass
class PipelineResult:
    # ... existing fields ...
    old_version_cleaned: bool = False
    old_chunks_deleted: int = 0
```

---

### 4.3 MCP Delete Tool: Two-Phase Confirmation

**New file**: `src/mcp_server/tools/delete_document.py`

**Tool name**: `delete_document`

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_path` | string | Yes | - | Document file path |
| `collection` | string | No | `"default"` | Collection name |
| `confirm_delete_data` | boolean | No | `false` | Execute deletion when true |

#### Behavior

**Phase 1 (confirm_delete_data = false)**: Preview mode

```
Input:  delete_document(source_path="/data/report.pdf", collection="contracts")
Output: {
    "status": "confirmation_required",
    "document": "report.pdf",
    "collection": "contracts",
    "associated_data": {
        "chunks": 42,
        "images": 3
    },
    "message": "Document 'report.pdf' has 42 chunks and 3 images in the RAG system. Call again with confirm_delete_data=true to delete all associated data.",
    "instructions": "To proceed, call delete_document with confirm_delete_data=true"
}
```

**Phase 2 (confirm_delete_data = true)**: Execute deletion

```
Input:  delete_document(source_path="/data/report.pdf", collection="contracts", confirm_delete_data=true)
Output: {
    "status": "deleted",
    "document": "report.pdf",
    "result": {
        "chunks_deleted": 42,
        "bm25_removed": true,
        "images_deleted": 3,
        "integrity_removed": true
    }
}
```

#### Data Collection for Preview

Use existing `DocumentManager.get_document_detail()` to gather chunk count and image count without performing deletion.

#### Registration

Register in `src/mcp_server/server.py` alongside existing tools.

---

## 5. Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `src/libs/loader/file_integrity.py` | **Modify** | Add `lookup_by_path()` method |
| `src/ingestion/pipeline.py` | **Modify** | Stage 1 old-version detection + DocumentManager init |
| `src/mcp_server/tools/delete_document.py` | **New** | MCP delete tool with two-phase confirmation |
| `src/mcp_server/server.py` | **Modify** | Register delete_document tool |
| `tests/unit/test_file_integrity_lookup.py` | **New** | Tests for lookup_by_path |
| `tests/unit/test_pipeline_old_version_cleanup.py` | **New** | Tests for auto-cleanup on re-ingest |
| `tests/unit/test_delete_document_tool.py` | **New** | Tests for MCP delete tool |

**Files NOT changed**: chunk_id generation, DocumentManager, BM25Indexer, VectorUpserter, ChromaStore, any storage layer logic.

---

## 6. Data Flow Diagrams

### Scenario A: User Deletes Document via Agent

```
User: "Delete report.pdf from the system"
  ↓
Agent calls: delete_document(source_path="report.pdf", confirm=false)
  ↓
MCP Tool: queries DocumentManager.get_document_detail()
  → Returns: "42 chunks, 3 images. Confirm deletion?"
  ↓
Agent shows user: "report.pdf has 42 chunks and 3 images. Delete all?"
  ↓
User confirms
  ↓
Agent calls: delete_document(source_path="report.pdf", confirm=true)
  ↓
MCP Tool: calls DocumentManager.delete_document()
  → ChromaDB: delete by doc_hash metadata    ✓
  → BM25: remove postings by doc_id prefix   ✓
  → ImageStorage: delete images by doc_hash   ✓
  → FileIntegrity: remove record              ✓
  ↓
Returns: DeleteResult summary
```

### Scenario B: User Modifies Document and Re-ingests

```
User modifies report.pdf, runs: ingest --path report.pdf --collection contracts
  ↓
Pipeline Stage 1:
  file_hash = SHA256(report.pdf)          → new_hash (changed)
  should_skip(new_hash)                   → False (no record with this hash)
  lookup_by_path("report.pdf", "contracts") → old_hash
  old_hash != new_hash                    → Document was modified!
  ↓
  DocumentManager.delete_document(old_hash)
    → Cascading delete across 4 stores    ✓ Clean slate
  ↓
Stage 2: Load PDF                         → Document object
Stage 3: Split                            → 100 Chunks
Stage 4: Transform                        → Refined/Enriched chunks
Stage 5: Embed                            → 100 embedding API calls
Stage 6: Upsert                           → ChromaDB + BM25 + Images
  ↓
FileIntegrity: record new_hash as success
  ↓
Result: 100 chunks ingested, 0 orphans
```

---

## 7. Future Optimization: Embedding Cache

**Not in current scope**, but the design is forward-compatible.

After Phase 1 is stable, an `EmbeddingCache` can be inserted between "detect old version" and "delete old version":

```
Stage 1 (enhanced):
  lookup_by_path() → old_hash exists
  ↓
  EmbeddingCache.warm(old_hash)           ← NEW: fetch old embeddings into memory
    → Reads from ChromaDB: {content_hash → embedding_vector}
  ↓
  DocumentManager.delete_document(old_hash) ← Same as Phase 1
  ↓
Stage 5 (enhanced):
  For each chunk:
    cache_hit = EmbeddingCache.get(content_hash)
    if cache_hit:  reuse embedding         ← Skip API call
    else:          call Embedding API       ← Only for changed chunks
```

**Interface sketch**:

```python
class EmbeddingCache:
    """Cache old embeddings before document deletion for reuse during re-ingest."""

    def warm(self, doc_hash: str, vector_store: BaseVectorStore) -> None:
        """Fetch old chunk embeddings into memory, keyed by content_hash."""

    def get(self, content_hash: str) -> list[float] | None:
        """Return cached embedding if content_hash matches, None otherwise."""
```

**Impact**: For a 100-chunk document with 8 changed chunks, reduces embedding API calls from 100 to 8 (92% saving).

**No changes required to**: chunk_id format, delete logic, upsert logic, or any storage layer.

---

## 8. Testing Strategy

### Unit Tests

| Test | Validates |
|------|-----------|
| `lookup_by_path` returns correct hash for existing path | FileIntegrity query correctness |
| `lookup_by_path` returns None for unknown path | No false positives |
| `lookup_by_path` respects collection filter | Cross-collection isolation |
| `lookup_by_path` returns latest record on multiple entries | ORDER BY updated_at DESC |
| Pipeline detects old version and calls delete | Auto-cleanup trigger |
| Pipeline proceeds normally when no old version exists | First-time ingestion unaffected |
| Pipeline continues on cleanup failure | Fault tolerance |
| MCP tool returns preview when confirm=false | Two-phase confirmation Phase 1 |
| MCP tool executes delete when confirm=true | Two-phase confirmation Phase 2 |
| MCP tool handles missing document gracefully | Error path |

### Integration Tests

| Test | Validates |
|------|-----------|
| Ingest file, modify, re-ingest → zero orphan chunks | End-to-end cleanup |
| Delete via MCP tool → all stores cleaned | Cascading delete through MCP layer |

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cleanup fails mid-delete (partial) | Low | Medium | DocumentManager already handles partial failures gracefully. Pipeline logs warning and continues. |
| lookup_by_path returns wrong hash | Very Low | High | Unit tests + collection filter. SQLite query is straightforward. |
| Pipeline performance regression | Low | Low | lookup_by_path is a single indexed SQLite query (~1ms). Negligible. |
| Concurrent ingestion of same file | Low | Medium | SQLite WAL mode handles concurrent reads. Concurrent writes to same file are already unsupported. |
