<div align="center">

# Modular RAG MCP Server

**[English](#-overview)** | **[中文](#-项目简介)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![MCP Protocol](https://img.shields.io/badge/MCP-Compatible-8A2BE2)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/Tests-1868%20passed-success?logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **fully modular** Retrieval-Augmented Generation system with **MCP protocol** integration.
Zero-code component switching. 4-layer API resilience. Dual-mode transport (Stdio + HTTP).

一个**全模块化** RAG 系统，集成 **MCP 协议**。
零代码组件切换，四层 API 容错，双模式传输（Stdio + HTTP）。

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Evaluation](#-evaluation) · [Roadmap](#-roadmap)

</div>

---

## 📖 Overview

A complete RAG pipeline — document ingestion, hybrid search, reranking, multimodal support — exposed as an [MCP](https://modelcontextprotocol.io) server. Connect it to GitHub Copilot, Claude Desktop, or any MCP-compatible AI assistant.

### Key Features

- **Hybrid Retrieval** — BM25 + Dense Embedding + RRF Fusion + Cross-Encoder / LLM Reranking
- **13 Pluggable Component Families** — LLM, Embedding, VectorStore, Reranker, Splitter, Evaluator, Cache, RateLimiter, QueryRewriter, QueryRouter, Memory, CircuitBreaker, Vision LLM — all swappable via `settings.yaml`
- **4-Layer Resilience** — Rate Limiter → Retry with Backoff → Circuit Breaker → Provider Failover (OpenAI → DeepSeek)
- **Dual Transport** — Stdio (local, zero-config) + Streamable HTTP (remote, Docker, multi-user)
- **5 MCP Tools** — `query_knowledge_hub`, `list_collections`, `get_document_summary`, `delete_document`, `ingest_document`
- **Multimodal** — PDF table extraction, formula OCR, Vision LLM image captioning, Base64 image return
- **Full Observability** — JSONL tracing across ingestion + query pipelines, 6-page Streamlit dashboard, Ragas evaluation
- **Multi-Representation Indexing** — LLM-generated natural language summaries for code chunks bridge the semantic gap between code and natural language queries

## 🏗 Architecture

```mermaid
graph TB
    subgraph Clients
        C1[GitHub Copilot]
        C2[Claude Desktop]
        C3[Other MCP Agents]
    end

    subgraph MCP["MCP Server Layer"]
        direction LR
        T1[Stdio Transport]
        T2[Streamable HTTP]
        PH[Protocol Handler]
        TOOLS["5 Tools"]
    end

    subgraph Core["Query Engine"]
        QR[Query Rewriter<br/>LLM / HyDE / None]
        QP[Query Processor]
        HS["Hybrid Search"]
        DR[Dense Retriever<br/>Embedding → ChromaDB]
        SR[Sparse Retriever<br/>BM25 Index]
        FU[RRF Fusion]
        RR[Reranker<br/>CrossEncoder / LLM / Cohere]
        RB[Response Builder<br/>+ Multimodal Assembler]
    end

    subgraph Ingestion["Ingestion Pipeline"]
        direction LR
        L[Loader<br/>PDF / MD / Code]
        S[Splitter]
        TR[Transform<br/>Refiner + Enricher<br/>+ Captioner + RTG]
        E[Embedding<br/>Dense + Sparse]
        U[Upsert<br/>ChromaDB + BM25]
    end

    subgraph Libs["Pluggable Libs Layer"]
        direction LR
        LLM["LLM ×4"]
        EMB["Embedding ×3"]
        VS["VectorStore"]
        CACHE["Cache"]
        MEM["Memory"]
        CB["Circuit Breaker<br/>+ Provider Chain"]
    end

    subgraph Observe["Observability"]
        TRACE[JSONL Tracing]
        DASH[Streamlit Dashboard<br/>6 Pages]
        EVAL[Ragas Evaluation]
    end

    C1 & C2 & C3 --> MCP
    T1 & T2 --> PH --> TOOLS
    TOOLS --> Core
    Core --> Libs
    Ingestion --> Libs
    Core & Ingestion --> Observe
    QR --> QP --> HS
    HS --> DR & SR
    DR & SR --> FU --> RR --> RB
    L --> S --> TR --> E --> U
```

### Design Patterns

| Pattern | Usage |
|:--------|:------|
| **Registry + Factory** | All component families — runtime provider registration |
| **Null Object** | NoneReranker, NoneEvaluator, NoneRewriter — no-op fallbacks |
| **Config-Driven** | `settings.yaml` → frozen dataclass → Factory.create_from_settings() |
| **Dual Storage** | ChromaDB (dense vectors) + independent BM25 index (sparse) |
| **4-Layer Resilience** | Rate Limiter → Retry → Circuit Breaker → Provider Failover |
| **Multi-Representation** | Code chunks: LLM summary for dense, raw code for BM25 |

### Resilience Stack

All LLM and Embedding API calls are protected by four composable layers:

```
ProviderChain (Layer 4 — Failover)
  └── CircuitBreaker (Layer 3 — Fast-fail after N failures)
        └── RateLimitedLLM (Layer 1 — Token bucket RPM control)
              └── @retry_with_backoff (Layer 2 — Exponential retry on 429/500/502/503)
                    └── HTTP API Call
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- LLM + Embedding provider: cloud API key (OpenAI / Azure / DeepSeek) or local [Ollama](https://ollama.ai)

### Setup

```bash
git clone https://github.com/nobitalqs/MODULAR-RAG-MCP-SERVER.git
cd MODULAR-RAG-MCP-SERVER
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp config/settings.yaml.example config/settings.yaml   # edit for your provider
```

### Run

```bash
# Ingest documents
python scripts/ingest.py --path /path/to/doc.pdf --collection my_kb

# MCP Server — Stdio (for Copilot / Claude Desktop)
python main.py

# MCP Server — HTTP (for remote / Docker deployment)
python main.py --transport http --host 0.0.0.0 --port 8000

# Dashboard
python scripts/start_dashboard.py    # → http://localhost:8501
```

### Connect to MCP Host

<details>
<summary><b>GitHub Copilot / VS Code</b></summary>

Add to `.vscode/mcp.json`:
```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "python",
      "args": ["main.py"],
      "cwd": "/path/to/MODULAR-RAG-MCP-SERVER"
    }
  }
}
```
</details>

<details>
<summary><b>Claude Desktop / Claude Code</b></summary>

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "python",
      "args": ["main.py"],
      "cwd": "/path/to/MODULAR-RAG-MCP-SERVER"
    }
  }
}
```
</details>

<details>
<summary><b>HTTP mode (remote)</b></summary>

Start server: `python main.py --transport http --host 0.0.0.0 --port 8000`

Connect via MCP Streamable HTTP client to `http://<host>:8000/mcp`
</details>

## 📊 Evaluation

### Dual Evaluation System

| Layer | Metrics | Method |
|:------|:--------|:-------|
| **IR Metrics** | Hit Rate, MRR | Deterministic chunk-ID matching |
| **LLM-as-Judge** (Ragas) | Faithfulness, Answer Relevancy, Context Precision | LLM-based scoring |

### Retrieval Quality Baseline

Evaluated against a golden test set of **16 queries across 4 categories** (3 documents, 417 chunks):

| Category | Queries | Recall@5 | MRR |
|:---------|:-------:|:--------:|:---:|
| Exact Fact | 5 | 50% | 1.0 |
| Semantic Understanding | 5 | 50% | 0.5 |
| Cross-Document | 3 | 100% | 0.5 |
| **Average** | **13** | **67%** | **0.67** |

### Test Suite

| Type | Count |
|:-----|------:|
| Unit | 1798 |
| Integration | 70 |
| **Total** | **1868** |

## 🔌 Pluggable Components

| Component | Providers | Config Key |
|:----------|:----------|:-----------|
| LLM | `openai`, `azure`, `ollama`, `deepseek` | `llm.provider` |
| Embedding | `openai`, `azure`, `ollama` | `embedding.provider` |
| Vision LLM | `openai`, `azure` | `vision_llm.provider` |
| Reranker | `cross_encoder`, `llm`, `cohere`, `none` | `rerank.provider` |
| VectorStore | `chroma` | `vector_store.provider` |
| Evaluator | `ragas`, `custom` | `evaluation.provider` |
| Cache | `memory`, `redis` | `cache.provider` |
| Rate Limiter | `token_bucket`, `null` | `rate_limit.provider` |
| Query Rewriter | `llm`, `hyde`, `none` | `query_rewriting.provider` |
| Memory | `memory`, `redis` | `memory.provider` |

> **Adding a new provider:** inherit the base class, implement the interface, register with the factory. Zero modification to existing code.

## 📺 Dashboard

6-page Streamlit management platform:

| Page | Capabilities |
|:-----|:-------------|
| **Overview** | Component config, collection stats, system health |
| **Data Browser** | Browse chunks, metadata, image preview |
| **Ingestion Manager** | Upload, ingest, delete documents with progress tracking |
| **Ingestion Traces** | Stage-by-stage waterfall: load → split → transform → embed → upsert |
| **Query Traces** | Pipeline trace: rewrite → dense/sparse → fusion → rerank |
| **Evaluation Panel** | Run evaluations, view metrics, historical trends |

## 🗺 Roadmap

### Multi-User & Distributed

| Priority | Item | Description |
|:--------:|:-----|:------------|
| 1 | Auth middleware | API Key / OAuth2 on HTTP transport |
| 2 | Multi-tenant isolation | Collection-level tenant separation |
| 3 | Redis state externalization | Cache, Memory, RateLimiter → Redis (code ready, config switch) |
| 4 | BM25 shared storage | Pickle → Redis / Elasticsearch |
| 5 | Distributed vector store | ChromaDB → Milvus / Qdrant |
| 6 | Docker containerization | Dockerfile + docker-compose |

### Advanced RAG

| Item | Description |
|:-----|:------------|
| Agentic RAG | Atomic tools (list_directory, verify_fact) for multi-step agent reasoning |
| Hierarchical Retrieval | Document-level summary → chunk-level search for large corpora |
| Multi-representation expansion | Extend LLM summaries from code to tables and formulas |
| Domain-specific embedding | Different embedding models per doc_type |

## 📂 Project Structure

```
src/
├── core/              # Settings, types, query engine, response builder, tracing
├── libs/              # 13 pluggable component families (Factory + Base + Providers)
│   ├── llm/           # 4 LLM + 2 Vision LLM providers
│   ├── embedding/     # 3 providers + EmbeddingChain failover
│   ├── reranker/      # CrossEncoder / LLM / Cohere / None
│   ├── resilience/    # RetryWithBackoff, RateLimitedLLM
│   ├── circuit_breaker/  # CircuitBreaker, ProviderChain, EmbeddingChain
│   └── ...            # cache, memory, query_rewriter, query_router, etc.
├── ingestion/         # 6-stage pipeline: load → split → transform → embed → upsert
├── mcp_server/        # Protocol handler + 5 MCP tools
└── observability/     # Logger, tracing, dashboard (6 pages), evaluation (Ragas)
```

## ❓ FAQ

<details>
<summary><b>Can I run fully offline?</b></summary>

Yes. Install [Ollama](https://ollama.ai), set all providers to `ollama`, and run with zero API costs:
```bash
ollama pull qwen2.5:3b && ollama pull nomic-embed-text
```
</details>

<details>
<summary><b>ChromaDB dimension mismatch?</b></summary>

Happens when switching embedding providers (e.g., OpenAI 1536-dim → Ollama 768-dim). Delete the old collection and re-ingest.
</details>

<details>
<summary><b>How to add a new LLM provider?</b></summary>

1. Inherit `BaseLLM`, implement `chat()`
2. Register: `factory.register_provider("my_provider", MyLLM)`
3. Set `llm.provider: "my_provider"` in settings.yaml
</details>

## 📄 License

MIT

---

## 📖 项目简介

一个完整的 RAG 流水线系统 — 文档摄取、混合检索、精排重排、多模态支持 — 通过 [MCP 协议](https://modelcontextprotocol.io) 对外暴露，可直接对接 GitHub Copilot、Claude Desktop 等 AI 助手。

### 核心特性

- **混合检索** — BM25 稀疏 + 稠密向量检索，RRF 融合，Cross-Encoder / LLM 精排
- **13 大可插拔组件族** — 全部通过 `settings.yaml` 零代码切换
- **四层 API 容错** — 限流 → 指数退避重试 → 熔断器 → Provider 自动降级（OpenAI → DeepSeek）
- **双模式传输** — Stdio（本地零配置）+ Streamable HTTP（远程部署/Docker/多用户）
- **5 个 MCP 工具** — 知识检索、集合列表、文档摘要、文档删除、远程摄入
- **多模态** — PDF 表格提取、公式 OCR、Vision LLM 图片描述、Base64 图片返回
- **全链路可观测** — JSONL 追踪 + 6 页 Streamlit 仪表盘 + Ragas 评估
- **多表示索引** — 代码块自动生成自然语言摘要用于语义检索，原始代码用于关键词检索

### 检索质量基线

| 类别 | 查询数 | Recall@5 | MRR |
|:-----|:------:|:--------:|:---:|
| 精确事实 | 5 | 50% | 1.0 |
| 语义理解 | 5 | 50% | 0.5 |
| 跨文档 | 3 | 100% | 0.5 |
| **平均** | **13** | **67%** | **0.67** |

### 快速开始

```bash
git clone https://github.com/nobitalqs/MODULAR-RAG-MCP-SERVER.git
cd MODULAR-RAG-MCP-SERVER
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp config/settings.yaml.example config/settings.yaml

# 摄取文档
python scripts/ingest.py --path /path/to/doc.pdf --collection my_kb

# 启动 MCP 服务器
python main.py                                    # Stdio（默认）
python main.py --transport http --port 8000       # HTTP 模式

# 启动仪表盘
python scripts/start_dashboard.py
```

### 可插拔组件

| 组件 | 可选后端 | 配置项 |
|:-----|:---------|:-------|
| LLM | `openai`, `azure`, `ollama`, `deepseek` | `llm.provider` |
| Embedding | `openai`, `azure`, `ollama` | `embedding.provider` |
| Vision LLM | `openai`, `azure` | `vision_llm.provider` |
| Reranker | `cross_encoder`, `llm`, `cohere`, `none` | `rerank.provider` |
| 评估器 | `ragas`, `custom` | `evaluation.provider` |
| 缓存 | `memory`, `redis` | `cache.provider` |
| 查询改写 | `llm`, `hyde`, `none` | `query_rewriting.provider` |
| 会话记忆 | `memory`, `redis` | `memory.provider` |

> 新增后端 = 继承基类 + 注册到 Factory，不改现有代码。

### 扩展方向

**多用户与分布式**：认证中间件 → 多租户隔离 → Redis 状态外置 → 分布式向量库 → Docker 容器化

**高级 RAG**：Agentic RAG（原子化工具）、分层检索、多表示扩展（表格/公式）、领域专用 Embedding

## 📄 License

MIT
