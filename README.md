<div align="center">

# Modular RAG MCP Server

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![MCP Protocol](https://img.shields.io/badge/MCP-Compatible-8A2BE2?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0wIDE4Yy00LjQxIDAtOC0zLjU5LTgtOHMzLjU5LTggOC04IDggMy41OSA4IDgtMy41OSA4LTggNHoiLz48L3N2Zz4=)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/Tests-1683%20passed-success?logo=pytest&logoColor=white)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-%E2%89%A580%25-success)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-ready, fully modular** RAG system with **MCP** integration | 一个**生产就绪、全模块化**的 RAG 系统，集成 **MCP 协议**

</div>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- ENGLISH                                                     -->
<!-- ═══════════════════════════════════════════════════════════ -->

<details open>
<summary><h2>📖 English</h2></summary>

### Overview

Modular RAG MCP Server is a **10-phase, 78-task** engineering project that implements a complete RAG pipeline — from document ingestion to hybrid search to LLM generation — all exposed via the [Model Context Protocol](https://modelcontextprotocol.io).

Zero-code component switching. Full-pipeline observability. Plug into Copilot, Claude, or any MCP host.

### Why This Project?

| Pain Point | Our Solution |
|:-----------|:-------------|
| Locked into one LLM vendor | **4 LLM providers** (Azure, OpenAI, Ollama, DeepSeek) — switch via config |
| Keyword OR semantic search | **Hybrid Search** — BM25 + Dense Embedding + RRF Fusion |
| RAG is a black box | **Full tracing** — every pipeline step recorded and visualized |
| Hard to evaluate quality | **Dual evaluation** — IR metrics + LLM-as-Judge (Ragas) |
| No standard AI tool interface | **MCP native** — works with Copilot, Claude Desktop, any MCP host |

### Core Highlights

- **Hybrid Retrieval** — BM25 sparse + dense embedding search, fused with Reciprocal Rank Fusion (RRF), then precision-refined with cross-encoder / LLM reranking
- **10 Pluggable Component Families** — LLM, Embedding, Splitter, VectorStore, Reranker, Evaluator, Cache, RateLimiter, QueryRewriter, Memory — all swappable via `settings.yaml`
- **MCP Server** — JSON-RPC over stdio, 4 tools (`query_knowledge_hub`, `list_collections`, `get_document_summary`, `delete_document`)
- **6-Page Streamlit Dashboard** — system overview, data browser, ingestion manager, query traces, ingestion traces, evaluation panel
- **Multimodal Ingestion** — PDF parsing with automatic image captioning (Vision LLM) embedded into chunk text
- **Advanced Features** — conversation memory, query rewriting (LLM / HyDE), circuit breaker, provider failover chain, embedding cache, rate limiting
- **1683 Tests** — unit, integration, E2E, contract tests, and golden test set recall regression

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server Layer                          │
│            JSON-RPC 2.0 · stdio Transport                   │
├─────────────────────────────────────────────────────────────┤
│                     Core Layer                               │
│     QueryEngine · HybridSearch · RRFFusion · ResponseBuilder│
├──────────────────────┬──────────────────────────────────────┤
│   Ingestion Pipeline │          Libs (Pluggable)            │
│  Load → Split →      │  LLM (4) · Embedding (3) · Reranker │
│  Transform → Embed → │  VectorStore · Splitter · Evaluator  │
│  Upsert              │  Cache · RateLimiter · Memory        │
│                      │  QueryRewriter · QueryRouter         │
├──────────────────────┴──────────────────────────────────────┤
│                  Observability Layer                          │
│       JSONL Tracing · Streamlit Dashboard · Ragas Eval      │
└─────────────────────────────────────────────────────────────┘
```

### Design Patterns

| Pattern | Where | Why |
|:--------|:------|:----|
| **Registry + Factory** | All 10 component families | Runtime provider registration, zero factory modification |
| **Null Object** | NoneReranker, NoneEvaluator, NoneRewriter | No-op fallbacks when features are disabled |
| **Config-Driven** | `settings.yaml` → frozen dataclass → Factory | Single YAML controls entire system behavior |
| **Dual Storage** | ChromaDB + BM25 pickle index | Dense semantic + sparse keyword retrieval |
| **Circuit Breaker** | External API calls | Three-state protection (CLOSED → OPEN → HALF_OPEN) |

### Evaluation

The project includes a **dual evaluation system** — deterministic IR metrics for fast regression checks, and LLM-as-Judge metrics for semantic quality assessment.

| Layer | Metrics | Method | Use Case |
|:------|:--------|:-------|:---------|
| **IR Metrics** (Custom) | Hit Rate, MRR | Deterministic, chunk-ID matching | Fast CI regression gating |
| **LLM-as-Judge** (Ragas) | Faithfulness, Answer Relevancy, Context Precision | LLM-based scoring | Semantic quality assessment |

**Golden Test Set** — 16 queries across 4 categories (exact fact, semantic understanding, cross-document, Chinese), evaluated against 3 indexed documents (417 chunks).

| Type | Count | Description |
|:-----|------:|:------------|
| Unit Tests | ~1500 | Factory, provider, contract, component tests |
| Integration Tests | ~100 | Cross-component, ChromaDB roundtrip, pipeline tests |
| E2E Tests | ~80 | MCP client, dashboard smoke, ingestion, query, recall |
| **Total** | **1683** | All passing |

### Quick Start

```bash
git clone https://github.com/nobitalqs/MODULAR-RAG-MCP-SERVER.git
cd MODULAR-RAG-MCP-SERVER
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp config/settings.yaml.example config/settings.yaml   # edit for your provider
```

```bash
python scripts/ingest.py --path /path/to/doc.pdf --collection my_kb   # ingest
python main.py                                                         # MCP server
python scripts/start_dashboard.py                                      # dashboard → http://localhost:8501
```

<details>
<summary><b>MCP Host Configuration</b></summary>

**GitHub Copilot / VS Code** — add to `.vscode/mcp.json`:
```json
{ "mcpServers": { "modular-rag": { "command": "python", "args": ["main.py"], "cwd": "/path/to/MODULAR-RAG-MCP-SERVER" } } }
```

**Claude Desktop** — add to Claude Desktop config:
```json
{ "mcpServers": { "modular-rag": { "command": "python", "args": ["main.py"], "cwd": "/path/to/MODULAR-RAG-MCP-SERVER" } } }
```

**Claude Code** — add to `.claude/settings.json` or use `/mcp`:
```json
{ "mcpServers": { "modular-rag": { "command": "python", "args": ["main.py"], "cwd": "/path/to/MODULAR-RAG-MCP-SERVER" } } }
```
</details>

### Dashboard

| Page | Capabilities |
|:-----|:-------------|
| **Overview** | Component health, collection stats, system configuration |
| **Data Browser** | Browse chunks, view metadata, search within collections |
| **Ingestion Manager** | Upload documents, trigger ingestion, delete documents |
| **Ingestion Traces** | Step-by-step trace: parse → split → transform → embed → store |
| **Query Traces** | Full pipeline trace: rewrite → retrieve → rerank → generate |
| **Evaluation Panel** | Run evaluations, view IR + LLM metrics, historical trends |

### Pluggable Components

| Component | Providers | Config Key |
|:----------|:----------|:-----------|
| LLM | `azure`, `openai`, `ollama`, `deepseek` | `llm.provider` |
| Embedding | `azure`, `openai`, `ollama` | `embedding.provider` |
| Vision LLM | `azure`, `openai`, `ollama` | `vision_llm.provider` |
| Reranker | `none`, `cross_encoder`, `llm`, `cohere` | `rerank.provider` |
| VectorStore | `chroma` | `vector_store.provider` |
| Splitter | `recursive`, `semantic`, `fixed` | `ingestion.splitter` |
| Evaluator | `custom`, `ragas` | `evaluation.provider` |
| Cache | `memory`, `redis` | `cache.provider` |
| Query Rewriter | `none`, `llm`, `hyde` | `query_rewriter.provider` |
| Memory | `memory`, `redis` | `memory.provider` |

> All components follow the **Base(ABC) → Factory(Registry) → Provider** pattern. Adding a new provider = inherit base + register.

### Repository Structure

```
src/
├── core/              # Settings, query engine (HybridSearch, RRFFusion)
├── libs/              # 10 pluggable component families
├── ingestion/         # Document parsing, chunking, transforms
├── mcp_server/        # MCP protocol handler + 4 tool implementations
└── observability/     # Tracing, evaluation (Ragas + Custom), dashboard

tests/
├── unit/              # ~85 test files (factory, provider, contract)
├── integration/       # Cross-component tests
├── e2e/               # MCP client, dashboard, ingestion, query, recall
└── fixtures/          # Golden test set + sample documents

scripts/               # CLI tools: ingest, query, evaluate, dashboard
config/                # settings.yaml — single config for everything
```

### FAQ

<details>
<summary><b>Can I use a free local model?</b></summary>

Yes. Install [Ollama](https://ollama.ai), pull models, and set `provider: "ollama"` in settings:
```bash
ollama pull llama3 && ollama pull nomic-embed-text
```
</details>

<details>
<summary><b>ChromaDB dimension mismatch error?</b></summary>

Happens when switching embedding providers (e.g., OpenAI 1536-dim → Ollama 768-dim). Delete the old collection and re-ingest.
</details>

<details>
<summary><b>How do I add a new LLM provider?</b></summary>

Inherit `BaseLLM`, implement `generate()` and `generate_stream()`, then register:
```python
LLMFactory.register_provider("my_provider", MyLLMClass)
```
</details>

<details>
<summary><b>Where are traces stored?</b></summary>

Default: `./logs/traces.jsonl`. Configure via `observability.trace_file` in settings.yaml.
</details>

### Branch Strategy

| Branch | Purpose |
|:-------|:--------|
| `main` | Latest stable code (single squashed commit) |
| `dev` | Full commit history — see how the project was built step by step |
| `clean-start` | Skeleton only (Skills + DEV_SPEC) — fork this to build it yourself |

</details>

---

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- 中文                                                        -->
<!-- ═══════════════════════════════════════════════════════════ -->

<details>
<summary><h2>📖 中文</h2></summary>

### 项目简介

Modular RAG MCP Server 是一个 **10 阶段、78 项任务** 的工程化项目，实现了完整的 RAG 流水线 — 从文档摄取到混合检索到 LLM 生成 — 全部通过 [Model Context Protocol](https://modelcontextprotocol.io) 对外暴露。

零代码切换组件，全链路可观测，即插即用对接 Copilot / Claude 等 AI 助手。

### 为什么做这个项目？

| 痛点 | 我们的方案 |
|:-----|:----------|
| 绑定单一 LLM 厂商 | **4 种 LLM 后端**（Azure、OpenAI、Ollama、DeepSeek）— 配置切换 |
| 关键词搜索 OR 语义搜索 | **混合检索** — BM25 + Dense Embedding + RRF 融合 |
| RAG 是黑盒 | **全链路追踪** — 每一步管线状态可视化 |
| 质量难以量化 | **双重评估** — IR 指标 + LLM-as-Judge (Ragas) |
| 无标准 AI 工具接口 | **MCP 原生** — 兼容 Copilot、Claude Desktop 等所有 MCP 宿主 |

### 核心亮点

- **混合检索** — BM25 稀疏 + 稠密向量检索，RRF 融合后经 Cross-encoder / LLM 精排
- **10 大可插拔组件族** — LLM、Embedding、Splitter、VectorStore、Reranker、Evaluator、Cache、RateLimiter、QueryRewriter、Memory — 全部通过 `settings.yaml` 切换
- **MCP 服务器** — JSON-RPC over stdio，提供 4 个工具（`query_knowledge_hub`、`list_collections`、`get_document_summary`、`delete_document`）
- **6 页 Streamlit 仪表盘** — 系统总览、数据浏览、摄取管理、查询追踪、摄取追踪、评估面板
- **多模态摄取** — PDF 解析 + Vision LLM 自动图片描述，嵌入 chunk 文本
- **高级功能** — 会话记忆、查询改写（LLM / HyDE）、熔断器、Provider 级联容错、Embedding 缓存、令牌桶限流
- **1683 个测试** — 单元、集成、E2E、契约测试 + Golden Test Set 召回回归

### 评估体系

项目内置 **双层评估系统** — 确定性 IR 指标用于快速回归检测，LLM-as-Judge 指标用于语义质量评估。

| 层级 | 指标 | 方法 | 场景 |
|:-----|:-----|:-----|:-----|
| **IR 指标**（Custom） | Hit Rate, MRR | 确定性 chunk-ID 匹配 | CI/CD 快速回归门控 |
| **LLM-as-Judge**（Ragas） | Faithfulness, Answer Relevancy, Context Precision | LLM 打分 | 语义质量评估 |

**Golden Test Set** — 精选 16 条查询，4 个类别（精确事实、语义理解、跨文档、中文），针对 3 份索引文档（共 417 chunks）进行评估。

| 类型 | 数量 | 说明 |
|:-----|-----:|:-----|
| 单元测试 | ~1500 | Factory、Provider、契约、组件测试 |
| 集成测试 | ~100 | 跨组件、ChromaDB 往返、管线测试 |
| E2E 测试 | ~80 | MCP 客户端、仪表盘冒烟、摄取、查询、召回 |
| **合计** | **1683** | 全部通过 |

### 快速开始

```bash
git clone https://github.com/nobitalqs/MODULAR-RAG-MCP-SERVER.git
cd MODULAR-RAG-MCP-SERVER
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp config/settings.yaml.example config/settings.yaml   # 按需修改
```

```bash
python scripts/ingest.py --path /path/to/doc.pdf --collection my_kb   # 摄取文档
python main.py                                                         # 启动 MCP 服务器
python scripts/start_dashboard.py                                      # 仪表盘 → http://localhost:8501
```

### 可插拔组件

| 组件 | 可选后端 | 配置项 |
|:-----|:---------|:-------|
| LLM | `azure`, `openai`, `ollama`, `deepseek` | `llm.provider` |
| Embedding | `azure`, `openai`, `ollama` | `embedding.provider` |
| Vision LLM | `azure`, `openai`, `ollama` | `vision_llm.provider` |
| Reranker | `none`, `cross_encoder`, `llm`, `cohere` | `rerank.provider` |
| VectorStore | `chroma` | `vector_store.provider` |
| Splitter | `recursive`, `semantic`, `fixed` | `ingestion.splitter` |
| Evaluator | `custom`, `ragas` | `evaluation.provider` |
| Cache | `memory`, `redis` | `cache.provider` |
| 查询改写 | `none`, `llm`, `hyde` | `query_rewriter.provider` |
| 会话记忆 | `memory`, `redis` | `memory.provider` |

> 所有组件遵循 **Base(ABC) → Factory(Registry) → Provider** 模式。新增后端 = 继承基类 + 注册。

### 分支策略

| 分支 | 用途 |
|:-----|:-----|
| `main` | 最新稳定代码（单次 squash commit） |
| `dev` | 完整提交历史 — 记录项目从零搭建的全过程 |
| `clean-start` | 仅骨架（Skills + DEV_SPEC）— fork 此分支从零开始自己实现 |

</details>

---

## 📄 License

MIT
