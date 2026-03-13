# Design: Multi-Format Document Loader & Ingestion MCP Tool

**Date:** 2026-03-13
**Status:** Draft (v2 — revised scope)
**Scope:** Add `.md`, source code, `.ipynb` ingestion + `ingest_document` MCP tool

---

## 1. Motivation

The RAG knowledge base currently only supports PDF documents. Upcoming agent workflows
require ingesting Markdown notes (Obsidian vault + project docs), source code files
(`.C`, `.cpp`, `.py`), and Jupyter Notebooks. Additionally, external MCP clients need
the ability to trigger ingestion remotely via an MCP tool.

## 2. Requirements

### Functional

| ID | Requirement | Phase |
|----|-------------|-------|
| F1 | Ingest `.md` files with YAML frontmatter extraction as metadata | 1 |
| F2 | Ingest source code files (`.C`, `.cpp`, `.py`, `.h`, etc.) with language detection | 1 |
| F3 | LoaderFactory routes files by extension — pipeline is format-agnostic | 1 |
| F4 | `ingest_document` MCP tool for external clients to trigger ingestion | 1 |
| F5 | Ingest `.ipynb` files with cell-type-aware conversion to Markdown | 2 |
| F6 | Notebook text outputs preserved; image/chart outputs discarded | 2 |
| F7 | Structure-aware MarkdownSplitter: heading hierarchy as semantic boundary, code blocks and tables protected | 2 |
| F8 | Image extraction for Markdown (`![alt](path)` → `[IMAGE: {id}]`) | Future |

### Non-Functional

| ID | Requirement |
|----|-------------|
| N1 | No new runtime dependencies beyond stdlib + PyYAML (already present) |
| N2 | Graceful degradation: malformed frontmatter / corrupt notebook → warning + best-effort |
| N3 | ≥ 80% test coverage for all new modules |

## 3. Architecture

### 3.1 Component Overview

**Phase 1 (this iteration):**

```
                   ┌─────────────────┐
  file_path ──────►│  LoaderFactory   │
                   │  (ext routing)   │
                   └───┬───┬─────┬───┘
                       │   │     │
                ┌──────┘   │     └──────┐
                ▼          ▼            ▼
          ┌──────────┐ ┌────────────┐ ┌────────────────┐
          │PdfLoader │ │Markdown    │ │SourceCode      │
          │          │ │Loader      │ │Loader           │
          └────┬─────┘ └─────┬──────┘ └──────┬─────────┘
               │             │               │
               └─────────────┼───────────────┘
                             ▼
                   RecursiveSplitter (existing)
                             ▼
                   Transform → Encode → Store

  MCP Client ──► ingest_document tool ──► IngestionPipeline
```

**Phase 2 (later):**

```
  + NotebookLoader (.ipynb)
  + MarkdownSplitter (heading-aware, protected regions)
```

### 3.2 LoaderFactory

Registry-pattern factory that maps file extensions to loader classes. Thin wrapper
around a `dict[str, type[BaseLoader]]` registry.

```python
class LoaderFactory:
    _registry: dict[str, type[BaseLoader]]

    def register_provider(self, ext: str, cls: type[BaseLoader]) -> None: ...
    def create_for_file(self, file_path: str | Path, **kwargs) -> BaseLoader: ...
```

**Dispatch logic:** `Path(file_path).suffix.lower()` → registry lookup.

**Difference from other factories:** Other factories use `create_from_settings()` with
a provider name from config. LoaderFactory uses the input file's extension because the
choice of loader is determined by the file, not the configuration. This is an intentional
divergence. If a future loader needs per-format config (e.g., `.docx` password), a
`LoaderSettings` section can be added to `settings.yaml`.

**Kwargs passthrough:** `create_for_file(file_path, **kwargs)` passes all kwargs to the
loader constructor. Each loader's `__init__` accepts only the kwargs it needs and ignores
the rest via `**kwargs`. This keeps the pipeline format-agnostic — no extension-specific
branching in `run()`.

**Extension mapping (Phase 1):**

| Extension | Loader |
|-----------|--------|
| `.pdf` | `PdfLoader` |
| `.md`, `.markdown` | `MarkdownLoader` |
| `.c`, `.cpp`, `.cxx`, `.cc`, `.h`, `.hxx` | `SourceCodeLoader` |
| `.py` | `SourceCodeLoader` |

### 3.3 MarkdownLoader

**Input:** `.md` / `.markdown` file path
**Output:** `Document(id=doc_{hash[:16]}, text=body, metadata={...})`

**Processing steps:**

1. **Validate extension** — `.md` or `.markdown`, else raise `ValueError`
2. **Read file** — UTF-8 encoding
3. **Parse frontmatter** — Regex detect `---` YAML block at file start, parse with
   `yaml.safe_load`. Malformed YAML → log warning, treat as empty frontmatter.
4. **Build metadata:**
   - Required: `source_path`, `doc_type: "markdown"`, `doc_hash`
   - Frontmatter fields **flat-merged** into metadata via `metadata.update(frontmatter)`
   - **Guard**: reserved keys (`source_path`, `doc_type`, `doc_hash`) cannot be
     overwritten by frontmatter — skip any conflicting keys with a warning
   - `title`: frontmatter `title` → first `# heading` (scan first 20 lines) → filename
5. **Return** `Document(id=doc_{hash[:16]}, text=body.strip(), metadata=metadata)`

**Image handling:** Not processed in Phase 1. `![alt](path)` syntax preserved as-is
in the text — the alt text is still searchable. Image extraction deferred to Future.

**Obsidian-specific syntax:** `[[wiki links]]` kept as-is (no resolution).

### 3.4 SourceCodeLoader

**Input:** Source code file path (`.C`, `.cpp`, `.py`, `.h`, etc.)
**Output:** `Document(id=doc_{hash[:16]}, text=raw_source, metadata={...})`

**Processing steps:**

1. **Validate extension** — Check against `_LANGUAGE_MAP`, else raise `ValueError`
2. **Read file** — UTF-8 with `errors="replace"` (source code may have stray bytes)
3. **Build metadata:**
   - `source_path`, `doc_type: "source_code"`, `doc_hash`
   - `language`: mapped from extension (e.g., `.C` → `"C++"`, `.py` → `"Python"`)
   - `filename`: `path.name`
   - `line_count`: line count of the source file

**Language map:**

```python
_LANGUAGE_MAP = {
    ".c": "C++", ".cpp": "C++", ".cxx": "C++", ".cc": "C++",
    ".h": "C++", ".hxx": "C++",
    ".py": "Python",
}
```

**Design note:** The loader reads source code as-is. No AST parsing, no comment
extraction. The raw text is directly usable for embedding and retrieval. The existing
`RecursiveSplitter` handles splitting by `\n\n` / `\n` which works well for code.

### 3.5 ingest_document MCP Tool

**Purpose:** Allow external MCP clients to trigger document ingestion remotely.

**Tool schema:**

```json
{
  "name": "ingest_document",
  "description": "Ingest a document into the knowledge base.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "file_path": { "type": "string", "description": "Absolute path to document" },
      "collection": { "type": "string", "default": "default" }
    },
    "required": ["file_path"]
  }
}
```

**Handler flow:**

1. Validate file exists → return `isError=True` if not found
2. Run `IngestionPipeline` via `asyncio.to_thread()` (pipeline is sync)
3. Return success summary (doc_id, chunk_count, collection) or error message

**Implementation pattern:** Follow existing tools (`query_knowledge_hub.py`):
- Module-level `TOOL_NAME`, `TOOL_DESCRIPTION`, `TOOL_INPUT_SCHEMA`
- `ingest_document_handler()` async function
- `register_tool(protocol_handler)` function
- Register in `protocol_handler.py` `_register_default_tools()`

**Security considerations:**
- File path is user-provided — validate existence but do NOT validate path traversal
  (MCP clients are trusted within the system boundary)
- Pipeline runs with existing settings (no config override via MCP params)

### 3.6 Transform Adaptation

**ChunkRefiner** requires a minor change: PDF-specific cleaning rules (hyphenation
repair, header/footer removal) should only apply when `doc_type == "pdf"`.

The `doc_type` is accessed from chunk metadata: `chunk.metadata.get("doc_type", "pdf")`.
Default `"pdf"` ensures backward compatibility for existing chunks without this field.

```python
def _rule_based_refine(self, text: str, doc_type: str = "pdf") -> str:
    # PDF-only rules
    if doc_type == "pdf":
        text = self._fix_hyphenation(text)
        text = self._remove_headers(text)
    # Universal rules — apply to all doc types
    text = self._collapse_blank_lines(text)
    text = self._strip_trailing_whitespace(text)
    return text

# In transform() for-loop, replace the existing call:
#   rule_text = self._rule_based_refine(chunk.text)
# with:
doc_type = chunk.metadata.get("doc_type", "pdf")
text = self._rule_based_refine(chunk.text, doc_type=doc_type)
```

**MetadataEnricher** and **ImageCaptioner** require no changes.

### 3.7 Pipeline Integration

**`IngestionPipeline.__init__()` changes:**

```python
# Stage 2: Loader — factory replaces hardcoded PdfLoader
self.loader_factory = LoaderFactory()
self.loader_factory.register_provider(".pdf", PdfLoader)
self.loader_factory.register_provider(".md", MarkdownLoader)
self.loader_factory.register_provider(".markdown", MarkdownLoader)
self.loader_factory.register_provider(".c", SourceCodeLoader)
self.loader_factory.register_provider(".cpp", SourceCodeLoader)
self.loader_factory.register_provider(".cxx", SourceCodeLoader)
self.loader_factory.register_provider(".cc", SourceCodeLoader)
self.loader_factory.register_provider(".h", SourceCodeLoader)
self.loader_factory.register_provider(".hxx", SourceCodeLoader)
self.loader_factory.register_provider(".py", SourceCodeLoader)
```

**`IngestionPipeline.run()` changes:**

```python
# Before: doc = self.loader.load(file_path)
# After: pass ALL kwargs — each loader picks what it needs, ignores the rest
loader = self.loader_factory.create_for_file(
    file_path,
    extract_images=True,
    image_storage_dir="data/images",
    table_extraction=table_extraction,      # PdfLoader uses, others ignore
    formula_extraction=formula_extraction,  # PdfLoader uses, others ignore
)
doc = loader.load(file_path)
```

No `if ext == "pdf"` branching — the pipeline stays format-agnostic.

**`scripts/ingest.py` changes:**

The default parameter in `discover_files()` signature changes:

```python
# Before: extensions = extensions or [".pdf"]
# After:  extensions = extensions or [".pdf", ".md", ".markdown", ".c", ".cpp", ".cxx", ".cc", ".h", ".hxx", ".py"]
```

Call sites remain unchanged.

## 4. File Inventory

### Phase 1 (this iteration)

| Action | File | Est. Lines |
|--------|------|-----------|
| New | `src/libs/loader/loader_factory.py` | ~60 |
| New | `src/libs/loader/markdown_loader.py` | ~80 |
| New | `src/libs/loader/source_code_loader.py` | ~60 |
| New | `src/mcp_server/tools/ingest_document.py` | ~120 |
| Modify | `src/ingestion/pipeline.py` | ~30 lines changed |
| Modify | `src/ingestion/transform/chunk_refiner.py` | ~10 lines changed |
| Modify | `src/mcp_server/protocol_handler.py` | ~5 lines changed |
| Modify | `scripts/ingest.py` | ~3 lines changed |
| Modify | `src/libs/loader/__init__.py` | export new classes |

### Phase 2 (later)

| Action | File | Est. Lines |
|--------|------|-----------|
| New | `src/libs/loader/notebook_loader.py` | ~150 |
| New | `src/libs/splitter/markdown_splitter.py` | ~180 |
| Modify | `src/libs/splitter/__init__.py` | export new class |

**Phase 1 total new code:** ~320 lines across 4 new files
**Phase 1 total modifications:** ~50 lines across 5 existing files

## 5. Test Plan

### Phase 1

| Test File | Type | Coverage |
|-----------|------|----------|
| `tests/unit/test_loader_factory.py` | Unit | Extension routing, unknown ext error, register/override, kwargs passthrough |
| `tests/unit/test_markdown_loader.py` | Unit | Frontmatter parsing (valid/malformed/missing), title extraction fallback chain, extension validation, empty file, Document contract (`source_path` in metadata), reserved key guard |
| `tests/unit/test_source_code_loader.py` | Unit | C++/Python language detection, extension validation, line_count, filename metadata, encoding errors, Document contract |
| `tests/unit/test_ingest_document_tool.py` | Unit | Handler success (mock pipeline), file not found error, tool schema validation, register_tool |
| `tests/unit/test_pipeline_loader_selection.py` | Unit | Auto-select PDF/MD/source loader, unsupported extension error |

### Phase 2

| Test File | Type | Coverage |
|-----------|------|----------|
| `tests/unit/test_notebook_loader.py` | Unit | Cell type conversion, text output, image discard, kernel metadata, malformed JSON, Document contract |
| `tests/unit/test_markdown_splitter.py` | Unit | Heading splitting, code block protection, table protection, oversized chunk fallback |
| `tests/integration/test_md_notebook_ingestion.py` | Integration | End-to-end: md/ipynb → pipeline → ChromaDB → retrieval |

## 6. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Frontmatter key overwrites reserved metadata | Data corruption | Guard: skip `source_path`, `doc_type`, `doc_hash` with warning |
| Frontmatter YAML injection | Security | `yaml.safe_load` only (no `yaml.load`), no code execution |
| Large source files (10k+ lines) | Slow splitting, huge chunks | Log warning; RecursiveSplitter handles this reasonably |
| `ingest_document` MCP tool abuse (rapid calls) | Resource exhaustion | Existing rate limiter (Phase J) applies to all MCP tools |
| Source code encoding issues | Garbled text | `errors="replace"` in read_text prevents crashes |

## 7. Phase 2: NotebookLoader + MarkdownSplitter (Deferred)

Retained from v1 design, to be implemented after Phase 1 is stable:

- **NotebookLoader**: nbformat v4 → cell-aware Markdown conversion, `\n\n---\n\n`
  cell boundaries, text output preserved, image output discarded
- **MarkdownSplitter**: heading hierarchy separators (`# > ## > ### > \n\n > \n`),
  protected regions (fenced code blocks, tables), UUID-based placeholders,
  start-of-string `\n` prepend, `**kwargs` compatibility with SplitterFactory

Full details in v1 spec sections 3.4 and 3.5 (preserved in git history).

## 8. Future Extensions (Out of Scope)

- Markdown/Notebook image extraction (`![alt](path)` → `[IMAGE: {id}]` → ImageCaptioner)
- `_image_utils.py` shared image processing module
- `.docx` / `.html` loader — same LoaderFactory pattern, zero pipeline changes
- Obsidian wiki link resolution (`[[page]]` → target content inline)
- HTML table protection in MarkdownSplitter
- Semantic splitter (embedding-based boundary detection)
