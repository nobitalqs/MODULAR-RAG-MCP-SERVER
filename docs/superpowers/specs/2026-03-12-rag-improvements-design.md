# RAG System Improvements Design

> Date: 2026-03-12
> Status: Approved
> Scope: Hybrid search weights + Table/formula extraction (implement now), Tiered memory (design only)

## Overview

Three independent improvements to the Modular RAG MCP Server, prioritized by complexity and impact:

| # | Improvement | Complexity | Status |
|---|-------------|-----------|--------|
| 1 | Hybrid search weight configuration | Low | Implement |
| 2 | Table & formula extraction | Medium | Implement |
| 3 | Tiered memory storage | High | Design only (Phase 2) |

---

## 1. Hybrid Search Weight Configuration

### Problem

RRF fusion uses uniform weights (1.0) for dense and sparse results. The `fuse_with_weights()` method exists in `src/core/query_engine/fusion.py` but is not exposed through settings. Users cannot tune the balance between semantic and keyword search.

### Design

**Changes: 4 files, ~30 lines of code.**

**Backward compatibility**: All new fields have defaults matching current behavior (`1.0`). Existing `settings.yaml` files continue to work without changes.

#### 1.1 Settings Layer

`src/core/settings.py` — add two fields to `RetrievalSettings`:

```python
@dataclass(frozen=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int
    dense_weight: float = 1.0   # NEW
    sparse_weight: float = 1.0  # NEW
    adaptive: AdaptiveRetrievalSettings | None = None
```

Validation in `validate_settings()`:
- `dense_weight >= 0.0` and `sparse_weight >= 0.0`
- At least one weight must be > 0.0 (both zero is rejected — all RRF scores would be 0.0)

#### 1.2 Config

`config/settings.yaml` and `config/settings.yaml.example`:

```yaml
retrieval:
  dense_weight: 1.0    # Semantic retrieval weight (>= 0.0)
  sparse_weight: 1.0   # BM25 keyword weight (>= 0.0)
```

#### 1.3 HybridSearch Wiring

The weight data flows through two layers:

**Step 1** — `HybridSearchConfig` (internal config dataclass in `hybrid_search.py`):

```python
@dataclass
class HybridSearchConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    enable_dense: bool = True
    enable_sparse: bool = True
    parallel_retrieval: bool = True
    metadata_filter_post: bool = True
    dense_weight: float = 1.0   # NEW
    sparse_weight: float = 1.0  # NEW
```

**Step 2** — `_extract_config()` propagates from `settings.retrieval`:

```python
def _extract_config(settings) -> HybridSearchConfig:
    r = settings.retrieval
    return HybridSearchConfig(
        ...,
        dense_weight=r.dense_weight,
        sparse_weight=r.sparse_weight,
    )
```

**Step 3** — `_fuse_results()` (line ~602) changes from `self.fusion.fuse()` to `self.fusion.fuse_with_weights()`:

```python
# Build weights list matching ranking_lists order: [dense, sparse]
weights = []
if dense_results:
    weights.append(self.config.dense_weight)
if sparse_results:
    weights.append(self.config.sparse_weight)

fused = self.fusion.fuse_with_weights(
    ranking_lists=ranking_lists,
    weights=weights,
    top_k=top_k,
    trace=trace,
)
```

Note: `self.fusion` (not `self.rrf_fusion`) is the correct attribute name per the existing codebase.

#### 1.4 Validation

- `dense_weight` and `sparse_weight` must be >= 0.0
- At least one weight must be > 0.0 (reject both-zero at settings validation)
- Weight of 0.0 effectively disables that retrieval channel in fusion scoring
- No upper bound enforced — practical range is 0.0 to 2.0; values beyond that are unusual but not harmful
- Unit test: weights correctly propagate through Config → `_fuse_results`; edge cases (0.0 weight, asymmetric weights)
- Benchmark: compare Recall@5 / MRR on golden test set with different weight combinations

#### 1.5 Files Changed

| File | Change |
|------|--------|
| `src/core/settings.py` | Add `dense_weight`, `sparse_weight` to `RetrievalSettings`; validation |
| `src/core/query_engine/hybrid_search.py` | Add to `HybridSearchConfig`; update `_extract_config()`; change `_fuse_results()` to use `fuse_with_weights()` |
| `config/settings.yaml` | Add weight config entries |
| `config/settings.yaml.example` | Add weight config entries with comments |
| `tests/unit/test_hybrid_search.py` | Test weight propagation and edge cases |

---

## 2. Table & Formula Extraction

### Problem

Current `PdfLoader` uses `page.get_text()` for plain text extraction. Tables lose structure (become garbled text) and formulas are lost entirely. This degrades retrieval quality for academic papers and technical documents.

### Design

Enhance `PdfLoader` within the existing pipeline architecture. No interface changes needed.

#### 2.1 Architecture

```
PdfLoader.load()
  |-- page.get_text()              # Existing: plain text
  |-- page.find_tables()           # NEW: tables -> Markdown
  |-- _extract_formulas()          # NEW: formula regions -> LaTeX
  +-- _extract_and_process_images() # Existing: images
```

#### 2.2 Table Extraction

**Tool**: PyMuPDF built-in `page.find_tables()` (zero new dependencies).

**Flow**:
1. Call `page.find_tables()` per page to get `TableFinder` object
2. Convert each table to Markdown via `.to_markdown()`
3. Locate table region in raw text by bbox coordinates, replace with Markdown table
4. If location fails, append to page end with `[TABLE_n]` marker

**Output example**:
```markdown
[TABLE: experiment_results]
| Model | BLEU | Params |
|-------|------|--------|
| Transformer | 28.4 | 65M |
| RNN | 25.2 | 80M |
```

#### 2.3 Formula Extraction

**Tool**: pix2tex (LaTeX-OCR), ~100MB model, CPU-compatible.

**Flow**:
1. **Detect**: Extract image regions from page, identify formulas via heuristics:
   - Aspect ratio > 3:1 (wide strip -> inline formula)
   - Height < 5% of page height (small image -> possible formula symbol)
   - Surrounding text contains math keywords ("equation", "where", etc.)
   - Low confidence -> mark as `[IMAGE]` instead of `[FORMULA]` to avoid false positives
2. **Extract**: Crop suspected formula regions, pass to pix2tex model
3. **Insert**: Embed OCR result as LaTeX: `$E = mc^2$` (inline) or `$$\sum_{i=1}^{n} x_i$$` (block)
4. **Degrade**: On pix2tex failure, preserve `[FORMULA: unrecognized]` placeholder; never block pipeline

#### 2.4 Table Extraction Strategy

PyMuPDF's `page.get_text()` returns plain text without bbox correlation, so mapping table bboxes back to text positions is impractical. The **primary strategy** is:

1. Extract page text via `page.get_text()` (existing)
2. Extract tables via `page.find_tables()` → Markdown
3. **Append** all detected tables at the end of each page's text block, tagged with `[TABLE_n]`
4. The downstream `ChunkRefiner` and `RecursiveSplitter` handle these naturally as part of the text

This avoids fragile text-position heuristics and keeps the implementation simple.

#### 2.5 Formula Detection Heuristics (Detail)

The `confidence_threshold` applies to the **pix2tex model's output confidence**, not detection confidence.

**Detection rules** (applied to each image region extracted from the page):

| Rule | Condition | Classification |
|------|-----------|----------------|
| Aspect ratio | width/height > 3 | Inline formula candidate |
| Small height | height < 5% page height AND width < 50% page width | Formula symbol candidate |
| Multi-line block | Located between paragraph breaks, height < 15% page height | Block formula candidate |
| Context keywords | Surrounding text ±200 chars contains: "equation", "formula", "where", "let", "given", "定义", "公式", "其中" | Boost confidence |
| None matched | No heuristic triggered | Classify as `[IMAGE]`, skip pix2tex |

Images that pass detection are sent to pix2tex. If pix2tex confidence < `confidence_threshold`, output `[FORMULA: unrecognized]` instead.

#### 2.6 Settings

`src/core/settings.py` — new frozen dataclasses:

```python
@dataclass(frozen=True)
class TableExtractionSettings:
    enabled: bool = True
    format: str = "markdown"  # "markdown" | "json"

@dataclass(frozen=True)
class FormulaExtractionSettings:
    enabled: bool = True
    model: str = "pix2tex"
    confidence_threshold: float = 0.5
```

**IngestionSettings update** — add new fields with defaults for backward compatibility:

```python
@dataclass(frozen=True)
class IngestionSettings:
    chunk_size: int
    chunk_overlap: int
    splitter: str = "recursive"
    batch_size: int = 100
    chunk_refiner: dict[str, Any] | None = None
    metadata_enricher: dict[str, Any] | None = None
    table_extraction: TableExtractionSettings | None = None   # NEW
    formula_extraction: FormulaExtractionSettings | None = None  # NEW
```

**Settings.from_dict() update** — parse nested dicts into dataclasses:

```python
# In Settings.from_dict() ingestion parsing section:
table_cfg = ingestion_data.get("table_extraction")
table_settings = TableExtractionSettings(**table_cfg) if table_cfg else None

formula_cfg = ingestion_data.get("formula_extraction")
formula_settings = FormulaExtractionSettings(**formula_cfg) if formula_cfg else None
```

**PdfLoader constructor wiring** — add optional parameters (backward compatible):

```python
class PdfLoader(BaseLoader):
    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
        table_extraction: TableExtractionSettings | None = None,   # NEW
        formula_extraction: FormulaExtractionSettings | None = None,  # NEW
    ):
```

The `IngestionPipeline` passes settings when constructing `PdfLoader`. If `None`, table/formula extraction is skipped (backward compatible).

`config/settings.yaml`:

```yaml
ingestion:
  table_extraction:
    enabled: true
    format: "markdown"
  formula_extraction:
    enabled: true
    model: "pix2tex"
    confidence_threshold: 0.5
```

#### 2.7 Dependencies

```
pix2tex ~= 0.1.2    # LaTeX OCR (~100MB model, auto-downloads on first run)
```

**Important**: pix2tex pulls in **PyTorch** as a transitive dependency (~700MB-2GB). To keep the base installation lightweight:
- pix2tex is an **optional dependency**: `pip install -r requirements-formula.txt` or `pip install modular-rag[formula]`
- Formula extraction uses **lazy import**: `import pix2tex` only when `formula_extraction.enabled: true`
- If `enabled: true` but pix2tex is not installed, logs a warning and falls back to `[FORMULA: unrecognized]` placeholders

PyMuPDF `find_tables()` requires no additional dependencies.

#### 2.8 Files Changed

| File | Change |
|------|--------|
| `src/libs/loader/pdf_loader.py` | Add table/formula extraction to `_extract_text()`; new constructor params |
| `src/core/settings.py` | Add `TableExtractionSettings`, `FormulaExtractionSettings`; update `IngestionSettings`; update `from_dict()` |
| `config/settings.yaml` | Add ingestion config sections |
| `config/settings.yaml.example` | Add ingestion config sections with comments |
| `requirements.txt` | (no change — pix2tex is optional) |
| `requirements-formula.txt` | NEW — optional `pix2tex` dependency |
| `tests/unit/test_loader_pdf_contract.py` | Add table/formula extraction tests |

#### 2.9 Graceful Degradation

Both extractors follow the project's Null Object pattern:
- Settings field is `None` or `enabled: false` → extraction skipped entirely, behavior identical to current
- pix2tex not installed → formula extraction logs warning once, all formulas become `[FORMULA: unrecognized]`
- pix2tex fails on a specific image → that formula becomes `[FORMULA: unrecognized]`, pipeline continues
- Table detection returns empty → no change to text output
- All degradation paths are logged at WARNING level

---

## 3. Tiered Memory Storage (Phase 2 — Design Only)

### Problem

Current memory implementation (InMemoryStore / RedisMemoryStore) permanently deletes session data on TTL expiry. Valuable conversation history is lost with no recovery path.

### Design

Three-tier storage with write-through and session-end archival:

```
Hot Layer (InMemoryStore / Redis)
  | write-through on every add_turn()
  v
Warm Layer (SQLite)
  | session end -> LLM summarize + embed
  v
Cold Layer (ChromaDB `memory_archive` collection)
```

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Demotion mechanism | Write-through (dual write) | Zero data loss, no background threads, SQLite write is microseconds |
| Recall modes | session_id exact lookup + semantic search | Warm: by ID; Cold: by embedding similarity |
| Archive trigger | Session end | Reuse existing LLM summarization, content is complete |
| Cold storage | ChromaDB separate collection | Zero new dependencies, reuse BaseVectorStore interface |

### Interface Preview

To avoid breaking existing implementations (`InMemoryStore`, `RedisMemoryStore`), archive methods are provided as **default no-ops** on `BaseMemoryStore`, not abstract methods. Only `TieredMemoryStore` overrides them with real logic.

```python
class BaseMemoryStore(ABC):
    # Existing abstract methods unchanged

    # New — default no-op implementations (non-breaking):
    def archive_session(self, session_id: str) -> None:
        """Archive a session to cold storage. No-op by default."""

    def search_archive(self, query: str, top_k: int = 5) -> tuple[SessionContext, ...]:
        """Search archived sessions by semantic similarity. Returns () by default."""
        return ()
```

### SQLite Warm Layer Schema (Preliminary)

```sql
CREATE TABLE sessions (
    session_id   TEXT PRIMARY KEY,
    turns        TEXT NOT NULL,      -- JSON-encoded list of ConversationTurn
    summary      TEXT,               -- LLM-generated summary (nullable)
    created_at   REAL NOT NULL,      -- epoch timestamp
    updated_at   REAL NOT NULL,      -- epoch timestamp
    archived_at  REAL                -- epoch timestamp, NULL if not archived to cold
);
CREATE INDEX idx_sessions_updated ON sessions(updated_at);
```

### New Files (Estimated)

| File | Purpose |
|------|---------|
| `src/libs/memory/sqlite_memory.py` | Warm layer: SQLite-backed session storage |
| `src/libs/memory/tiered_memory.py` | Orchestrator: hot + warm + cold coordination |
| `src/libs/memory/archive_service.py` | Session-end: summarize + embed + store to cold |

**This section is recorded for future implementation. Not included in the current implementation plan.**

---

## Implementation Priority

```
Phase 1 (Now):
  1. Hybrid search weights    (~1 day)
  2. Table & formula extraction (~3 days)

Phase 2 (Future):
  3. Tiered memory storage     (~5 days)
```
