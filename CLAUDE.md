# Project: Modular RAG MCP Server

## 项目简介

模块化 RAG 系统，基于多阶段检索增强生成（RAG）与模型上下文协议（MCP）设计。
支持混合检索（BM25 + Dense Embedding）、Rerank 精排、MCP 协议对接，具备全链路可观测性与可视化管理能力。

## 核心文档

| 文档 | 用途 |
|------|------|
| `DEV_SPEC.md` | **唯一权威规范** — 架构、技术选型、模块设计、进度跟踪 |
| `findings.md` | 技术决策与架构洞察记录 |
| `config/settings.yaml` | 运行时配置（零代码切换所有组件） |

**修改代码前必须先读 `DEV_SPEC.md` 对应章节。**

## 环境设置

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp config/settings.yaml.example config/settings.yaml  # 按需修改
```

## 开发命令

| 操作 | 命令 |
|------|------|
| 虚拟环境 | `source .venv/bin/activate` |
| 运行测试 | `pytest tests/` |
| 同步规范 | `python .claude/skills/spec-sync/sync_spec.py --force` |
| 启动 MCP Server | `python main.py` |
| 启动 Dashboard | `python scripts/start_dashboard.py` |
| 数据摄取 | `python scripts/ingest.py --path <file> --collection <name>` |
| 查询测试 | `python scripts/query.py --query "问题" --verbose` |

## 开发工作流（Skills Pipeline）

使用 `.claude/skills/` 下的 6 个 skill 驱动开发，每次迭代完成一个子任务：

```
spec-sync → progress-tracker → implement → testing-stage → checkpoint
```

| Skill | 触发方式 | 职责 |
|-------|---------|------|
| `dev-workflow` | "下一阶段" / "next task" | 编排整个流水线 |
| `spec-sync` | "同步规范" / "sync spec" | 拆分 DEV_SPEC.md 为 chapter 文件 |
| `progress-tracker` | "检查进度" / "status" | 定位当前任务，验证进度一致性 |
| `implement` | "实现" / "implement" | 读 spec → 提取设计原则 → 写代码 |
| `testing-stage` | "运行测试" / "run tests" | 按任务性质选择测试类型并执行 |
| `checkpoint` | "保存进度" / "checkpoint" | 更新 DEV_SPEC.md 进度 + git commit |

**进度跟踪唯一源**：`DEV_SPEC.md` 进度跟踪表（通过 checkpoint skill 更新）。

## 项目结构

```
src/
├── core/           # 配置加载、Settings dataclass
├── libs/           # Phase B: 可插拔组件层（6 大模块）
│   ├── llm/        # BaseLLM + LLMFactory + 4 providers + Vision
│   ├── embedding/  # BaseEmbedding + EmbeddingFactory + 3 providers
│   ├── splitter/   # BaseSplitter + SplitterFactory + RecursiveSplitter
│   ├── vector_store/ # BaseVectorStore + VectorStoreFactory + ChromaStore
│   ├── reranker/   # BaseReranker + RerankerFactory + NoneReranker/LLM/CrossEncoder
│   └── evaluator/  # BaseEvaluator + EvaluatorFactory + CustomEvaluator
├── pipeline/       # Phase C+: 摄取与检索管线
├── mcp/            # MCP 协议层
└── observability/  # 可观测性（Ragas 评估、链路追踪）
tests/
├── unit/           # 单元测试（factory、provider smoke、contract）
└── integration/    # 集成测试（ChromaStore roundtrip 等）
```

## 关键架构决策

- **Registry Pattern**: Base(ABC) + Factory(Registry) + 多实现，运行时注册 provider，新增无需改 Factory
- **双存储**: ChromaDB（dense vector + content + metadata）+ 独立 BM25 倒排索引
- **Transform 三步**: ChunkRefiner / MetadataEnricher / ImageCaptioner 独立模块，各有开关和降级
- **配置驱动**: `settings.yaml` → Settings dataclass → Factory.create_from_settings() → 实例
- **Null Object Pattern**: NoneReranker / NoneEvaluator 作为禁用时的无操作回退
- **参考实现**: `/home/nobi/Downloads/MODULAR-RAG-MCP-SERVER-main`，每完成一个阶段对比

## 文档输出

当用户要求总结文档到 Obsidian 时，使用 `md-doc-writer` subagent（Task tool, subagent_type="md-doc-writer"）生成高质量 Markdown。
输出路径：`/home/nobi/notes/note/MODULAR-RAG-MCP-SERVER/`。

## 当前进度

- **Phase A** (Core Settings): ✅ 完成 — 4/4 tasks
- **Phase B** (Libs Pluggable Layer): ✅ 完成 — 16/16 tasks
- **Phase C** (Ingestion Pipeline): ✅ 完成 — 15/15 tasks
- **Phase D** (Retrieval MVP): ✅ 完成 — 7/7 tasks
- **Phase E** (MCP Server Layer): ✅ 完成 — 6/6 tasks
- **Phase F** (Trace Infrastructure): ✅ 完成 — 5/5 tasks
- **Phase G** (Dashboard): ✅ 完成 — 6/6 tasks
- **Phase H** (Evaluation System): ✅ 完成 — 5/5 tasks
- **Phase I** (E2E & Documentation): ⬚ 未开始 — 0/5 tasks
- **总进度**: 64/69 tasks (93%)
- **总测试数**: 1337 passed

## 测试文件规范

1. **优先使用参考实现的测试文件**：参考实现 `/home/nobi/Downloads/MODULAR-RAG-MCP-SERVER-main/tests/` 中的测试经过成熟验证，应作为首选。如果参考实现有对应测试，直接采用或基于其适配，不要从头编写。
2. **测试 fixture 必须包含期望输出**：如 `noisy_chunks.json` 需有 `expected_clean` 字段，测试用精确断言（`assert result == expected`），避免模糊的 `in` 检查。
3. **新建测试文件前先与用户讨论**：当参考实现没有对应测试时，先向用户说明测试设计方案（覆盖哪些场景、fixture 设计、断言策略），经确认后再编写。
4. **测试 fixture 格式**：JSON fixture 使用 Object（按名称索引），非 Array。每个场景包含 `input`、`expected_*`、`note` 字段。

## 实现注意事项

1. Chroma 原生不支持 sparse vector 字段，BM25 使用独立倒排索引（pickle）
2. `libs/evaluator/` 放契约接口，`observability/evaluation/` 放业务逻辑（Ragas 等），不重复
3. Phase I2 的 Dashboard 冒烟测试（Streamlit AppTest）不可跳过
4. TDD: 每个模块先写测试再实现，覆盖率 ≥ 80%
5. 新增 provider 只需: 继承 Base → 实现方法 → `factory.register_provider("name", Class)`
