# Design: Markdown & Notebook Document Loader

**Date:** 2026-03-13
**Status:** Draft
**Scope:** Add `.md` and `.ipynb` ingestion support with structure-aware splitting

---

## 1. Motivation

The RAG knowledge base currently only supports PDF documents. Upcoming agent workflows
require ingesting Markdown notes (Obsidian vault + project docs) and Jupyter Notebooks
(data analysis, tutorials, research). Both formats carry structural semantics (headings,
code blocks, cell boundaries) that a generic text splitter destroys.

## 2. Requirements

### Functional

| ID | Requirement |
|----|-------------|
| F1 | Ingest `.md` files with YAML frontmatter extraction as metadata |
| F2 | Ingest `.ipynb` files with cell-type-aware conversion to Markdown |
| F3 | Notebook text outputs preserved; image/chart outputs discarded |
| F4 | Local images in Markdown (`![alt](path)`) copied to `data/images/` and replaced with `[IMAGE: {id}]` placeholder (same as PDF) |
| F5 | Structure-aware splitting: heading hierarchy as semantic boundary, code blocks and tables protected from mid-split |
| F6 | LoaderFactory routes files by extension — pipeline is format-agnostic |
| F7 | Settings-driven splitter selection (`markdown` or `recursive`) |

### Non-Functional

| ID | Requirement |
|----|-------------|
| N1 | No new runtime dependencies beyond stdlib + PyYAML (already present) |
| N2 | Graceful degradation: malformed frontmatter / corrupt notebook → warning + best-effort |
| N3 | ≥ 80% test coverage for all new modules |

## 3. Architecture

### 3.1 Component Overview

```
                   ┌─────────────────┐
  file_path ──────►│  LoaderFactory   │
                   │  (ext routing)   │
                   └───┬─────┬─────┬─┘
                       │     │     │
                ┌──────┘     │     └──────┐
                ▼            ▼            ▼
          ┌──────────┐ ┌────────────┐ ┌──────────────┐
          │PdfLoader │ │MarkdownLoader│ │NotebookLoader│
          └────┬─────┘ └─────┬──────┘ └──────┬───────┘
               │             │               │
               ▼             ▼               ▼
            Document      Document        Document
            (text=md)     (text=md)       (text=md)
               │             │               │
               └─────────────┼───────────────┘
                             ▼
                    ┌─────────────────┐
                    │ SplitterFactory  │
                    │ (strategy cfg)   │
                    └───┬─────────┬───┘
                        │         │
                        ▼         ▼
                  Recursive   Markdown
                  Splitter    Splitter
                        │         │
                        └────┬────┘
                             ▼
                   Transform → Encode → Store
                   (existing pipeline, unchanged)
```

### 3.2 LoaderFactory

Registry-pattern factory that maps file extensions to loader classes.

```python
class LoaderFactory:
    _registry: dict[str, type[BaseLoader]]

    def register_provider(self, ext: str, cls: type[BaseLoader]) -> None: ...
    def create_for_file(self, file_path: Path, **kwargs) -> BaseLoader: ...
```

**Dispatch logic:** `file_path.suffix.lower().lstrip(".")` → registry lookup.

**Difference from other factories:** Other factories use `create_from_settings()` with
a provider name from config. LoaderFactory uses the input file's extension because the
choice of loader is determined by the file, not the configuration. This is an intentional
divergence: loader selection is inherently file-driven, not config-driven. If a future
loader needs per-format config (e.g., `.docx` password), a `LoaderSettings` section can
be added to `settings.yaml` and passed as kwargs — but that is out of scope for now.

**Kwargs passthrough:** `create_for_file(file_path, **kwargs)` passes all kwargs to the
loader constructor. Each loader's `__init__` accepts only the kwargs it needs and ignores
the rest via `**kwargs`. This keeps the pipeline format-agnostic — no `if ext == "pdf"`
branching in `run()`.

### 3.3 MarkdownLoader

**Input:** `.md` file path
**Output:** `Document(id=sha256, text=body, metadata={...})`

**Processing steps:**

1. **Read file** — UTF-8 encoding
2. **Parse frontmatter** — Detect `---` YAML block at file start, parse with
   `yaml.safe_load`. Malformed YAML → log warning, treat as empty frontmatter.
3. **Process images** — Regex scan for `![alt](path)`:
   - Local relative/absolute path → copy to `data/images/{doc_hash}/`, replace
     with `[IMAGE: {id}]` placeholder
   - Remote URL → keep original syntax unchanged (no download)
4. **Build metadata:**
   - `source_path`, `doc_hash`, `doc_type: "markdown"`
   - `title`: frontmatter `title` field → first `# heading` → filename
   - `frontmatter`: full parsed YAML dict (for downstream filtering)
   - `images`: list of extracted image metadata dicts

**Obsidian-specific syntax:** `[[wiki links]]` kept as-is (no resolution). Only
frontmatter is parsed per requirement scope.

### 3.4 NotebookLoader

**Input:** `.ipynb` file path (nbformat v4)
**Output:** `Document(id=sha256, text=cells_as_md, metadata={...})`

**Processing steps:**

1. **Parse JSON** — `json.loads()` with UTF-8. Invalid JSON → raise `ValueError`.
2. **Extract kernel metadata** — `metadata.kernelspec.{display_name, language}`
3. **Convert cells to Markdown sections:**

   | Cell type | Conversion |
   |-----------|------------|
   | `markdown` | Direct use (join source lines) |
   | `code` | Wrap in ` ```{lang} ... ``` `. Append text outputs as ` **Output:** ``` ... ``` ` |
   | `raw` | Direct use (join source lines) |

4. **Output handling:**
   - `stream` (stdout/stderr) → extract text, join
   - `execute_result` / `display_data` with `text/plain` → extract text
   - `image/png`, `image/svg+xml`, etc. → **discard** (per requirement)
   - `error` → discard (tracebacks are noise for RAG). **Trade-off:** tutorial
     notebooks sometimes use errors intentionally (e.g., demonstrating `TypeError`).
     This is accepted as a known limitation; errors can be enabled in a future
     config flag if needed.

5. **Join sections** — `"\n\n---\n\n"` separator between cells. The `---`
   (horizontal rule) serves as a cell boundary signal for MarkdownSplitter.
6. **Process images** — Same as MarkdownLoader for `![alt](path)` in markdown cells.
   Reuses shared `_image_utils` module.
7. **Build metadata:**
   - `source_path`, `doc_hash`, `doc_type: "notebook"`
   - `title`: first `# heading` in first markdown cell → filename
   - `kernel`, `language`, `cell_count`
   - `images`: list of extracted image metadata dicts

**Shared logic:** Image processing (`_process_images`, `_copy_local_image`) extracted
to `src/libs/loader/_image_utils.py` to avoid duplication with MarkdownLoader.

### 3.5 MarkdownSplitter

**Purpose:** Structure-aware splitting that respects Markdown semantics.

**Separator hierarchy (highest → lowest priority):**

```
\n---\n      → cell boundary / horizontal rule
\n# <space>  → H1
\n## <space> → H2
\n### <space>→ H3
\n#### <space>→ H4
\n\n          → paragraph
\n            → line
<space>       → word
```

**Start-of-string handling:** The `\n# ` separator won't match a heading at the
very beginning of the document (no leading newline). The splitter must prepend `\n`
to the input text before splitting and strip it from the first chunk afterward.
This avoids special-casing regex anchors while ensuring the first heading is a valid
split boundary.

**Protected regions:** Certain constructs must never be split mid-way:

| Region | Detection | Protection |
|--------|-----------|------------|
| Fenced code blocks | ` ``` ... ``` ` (with optional language tag) | Replace with `<<<PROTECTED_N>>>` placeholder before splitting, restore after |
| Tables | Consecutive lines matching `\|...\|` pattern | Same placeholder mechanism |

**Algorithm:**

1. Extract protected regions → replace with single-line placeholders
2. Recursive split using separator hierarchy (same algorithm as
   `RecursiveCharacterTextSplitter` from langchain)
3. Restore placeholders → original content
4. Post-pass: if any chunk exceeds `chunk_size` after restoration (a single
   protected region was very large), split that chunk by lines while preserving
   fenced code block wrappers

**Heading attribution:** When splitting at a heading boundary, the heading line
stays with the content below it (not the chunk above). This ensures each chunk
carries its own section title for retrieval context.

**Configuration:**

```yaml
ingestion:
  splitter:
    strategy: markdown    # or "recursive" for legacy behavior
    chunk_size: 1000
    chunk_overlap: 200
```

Registered in SplitterFactory: `factory.register_provider("markdown", MarkdownSplitter)`

**Constructor compatibility with SplitterFactory:** `SplitterFactory.create_from_settings()`
passes all `IngestionSettings` fields (including `batch_size`, `chunk_refiner`, etc.)
as kwargs. `MarkdownSplitter.__init__` must accept and ignore these via `**kwargs`:

```python
def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs: Any) -> None:
```

This matches the pattern used by `RecursiveSplitter`.

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

# Called from transform():
doc_type = chunk.metadata.get("doc_type", "pdf")
text = self._rule_based_refine(chunk.text, doc_type=doc_type)
```

**MetadataEnricher** and **ImageCaptioner** require no changes — they already
operate on generic `Chunk` objects with `[IMAGE: {id}]` placeholders.

### 3.7 Pipeline Integration

**`IngestionPipeline.__init__()` changes:**

```python
# Stage 2: Loader — factory replaces hardcoded PdfLoader
self.loader_factory = LoaderFactory()
self.loader_factory.register_provider("pdf", PdfLoader)
self.loader_factory.register_provider("md", MarkdownLoader)
self.loader_factory.register_provider("ipynb", NotebookLoader)

# Stage 3: Splitter — add markdown option
splitter_factory.register_provider("markdown", MarkdownSplitter)
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
# Before: def discover_files(path, extensions=None):  extensions = extensions or [".pdf"]
# After:  def discover_files(path, extensions=None):  extensions = extensions or [".pdf", ".md", ".ipynb"]
```

Call sites remain unchanged.

**`_image_utils.py` path resolution:** All image paths resolved via
`Path(image_storage_dir).resolve()` to avoid CWD-dependent behavior (same latent
issue exists in PdfLoader — not compounding it here).

## 4. File Inventory

| Action | File | Est. Lines |
|--------|------|-----------|
| New | `src/libs/loader/loader_factory.py` | ~60 |
| New | `src/libs/loader/markdown_loader.py` | ~120 |
| New | `src/libs/loader/notebook_loader.py` | ~150 |
| New | `src/libs/loader/_image_utils.py` | ~60 |
| New | `src/libs/splitter/markdown_splitter.py` | ~180 |
| Modify | `src/ingestion/pipeline.py` | ~30 lines changed |
| Modify | `src/ingestion/transform/chunk_refiner.py` | ~10 lines changed |
| Modify | `scripts/ingest.py` | ~3 lines changed |
| Modify | `src/libs/loader/__init__.py` | export new classes |
| Modify | `src/libs/splitter/__init__.py` | export new class |
| Modify | `config/settings.yaml.example` | add comments |

**Total new code:** ~570 lines across 5 new files
**Total modifications:** ~45 lines across 6 existing files

## 5. Test Plan

| Test File | Type | Coverage |
|-----------|------|----------|
| `tests/unit/test_loader_factory.py` | Unit | Extension routing, unknown ext error, register/override |
| `tests/unit/test_markdown_loader.py` | Unit | Frontmatter parsing (valid/malformed/missing), image extraction (local/remote/missing), title extraction fallback chain, empty file, non-UTF8, **Document contract assertion** (`source_path` in metadata) |
| `tests/unit/test_notebook_loader.py` | Unit | Cell type conversion (markdown/code/raw), text output preservation, image output discard, error output discard, kernel metadata, malformed notebook JSON, empty cells, **Document contract assertion** (`source_path` in metadata) |
| `tests/unit/test_markdown_splitter.py` | Unit | Heading-based splitting, code block protection, table protection, oversized protected region fallback, heading attribution, cell boundary `---` splitting, empty input |
| `tests/unit/test_image_utils.py` | Unit | Local image copy, remote URL passthrough, missing image graceful degradation, placeholder format |
| `tests/integration/test_md_notebook_ingestion.py` | Integration | End-to-end: md/ipynb → pipeline → ChromaDB → retrieval verification |

**Fixtures:**

- `tests/fixtures/sample_note.md` — Obsidian-style with frontmatter, headings, code block, image ref, table
- `tests/fixtures/sample_notebook.ipynb` — Mixed cells: markdown + code with text output + code with image output + raw
- `tests/fixtures/markdown_chunks.json` — Expected chunking results for MarkdownSplitter

## 6. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large notebook (100+ cells) produces huge Document.text | Memory spike, slow splitting | Log warning if cell_count > 50; consider streaming in future |
| Protected region placeholder collision with actual text | Corrupted chunks | Use UUID-based placeholder format: `<<<PROTECTED_{uuid}>>>` |
| Notebook nbformat v3 or older | Parse failure | Check `nbformat` (major version) field: `if notebook_json.get("nbformat", 0) != 4: raise ValueError(...)`. Do NOT check `nbformat_minor` — minor versions 0–3 within v4 are valid |
| Frontmatter YAML injection (malicious content) | Security | `yaml.safe_load` only (no `yaml.load`), no code execution |
| Image path traversal (`![](../../etc/passwd)`) | Security | Validate resolved path is under document's parent directory or `image_storage_dir` |

## 7. Future Extensions (Out of Scope)

- `.docx` / `.html` loader — same LoaderFactory pattern, zero pipeline changes
- Obsidian wiki link resolution (`[[page]]` → target content inline)
- Notebook image output captioning via VisionLLM
- Semantic splitter (embedding-based boundary detection)
