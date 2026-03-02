# Findings — Modular RAG MCP Server

> 规划和实现过程中积累的技术发现、设计决策与参考笔记。
> 更新日期：2026-02-14

---

## 1. 项目背景

### 1.1 本次实现性质
- 这是一个**经验证设计的重新落地**，新 repo 从零开始
- DEV_SPEC.md 是成熟文档（3200+ 行），任务拆解经过实战验证
- 上次实现完成了 Phase A-H（63/68 个任务），仅剩 Phase I
- 当前 repo：Phase A 已完成，49 个测试通过

### 1.2 项目定位
- **核心定位**：自学与教学同步的 RAG 实战平台，面向面试准备与简历丰富
- **目标受众**：学习 RAG 的开发者，准备 AI/ML 面试的求职者
- **交付物**：代码 + 技术文档 + 视频讲解（三位一体）

### 1.3 参考实现
- 上次实现代码位于 `/home/nobi/Downloads/MODULAR-RAG-MCP-SERVER-main`
- 每个 Phase 完成后可与之对比验证（见 CLAUDE.md 指令）

---

## 2. 架构设计决策（来自 DEV_SPEC）

### 2.1 五层架构
```
MCP Server 层    →  接口层（Stdio Transport，JSON-RPC 2.0）
Core 层          →  核心业务逻辑（QueryEngine、Response、Trace）
Ingestion 层     →  离线流水线（Load → Split → Transform → Embed → Upsert）
Libs 层          →  可插拔抽象层（Base + Factory + Provider 实现）
Observability 层 →  结构化日志 + Streamlit Dashboard + 评估模块
```

### 2.2 存储架构（多后端）
| 存储 | 用途 | 技术方案 |
|------|------|----------|
| 稠密向量 + 内容 + 元数据 | 语义检索 | ChromaDB（嵌入式，本地） |
| 稀疏索引（倒排 + IDF） | BM25 关键词检索 | 自建实现，pickle 持久化 |
| 文件完整性记录 | 增量摄取 | SQLite（data/db/ingestion_history.db） |
| 图片索引映射 | 图片检索与展示 | SQLite（data/db/image_index.db） |
| 追踪日志 | 可观测性 | JSON Lines（logs/traces.jsonl） |
| 原始图片 | 多模态返回 | 本地文件系统（data/images/） |

### 2.3 核心设计模式：Base + Factory
每个可插拔组件遵循统一模式：
```python
# 1. 抽象基类
class BaseXxx(ABC):
    @abstractmethod
    def method(self, ...) -> ...: ...

# 2. 工厂 + 配置路由
class XxxFactory:
    @staticmethod
    def create(settings: Settings) -> BaseXxx:
        provider = settings.xxx.provider
        if provider == "a": return AImpl(settings)
        elif provider == "b": return BImpl(settings)
        raise ValueError(f"Unknown provider: {provider}")

# 3. 多个具体实现
class AImpl(BaseXxx): ...
class BImpl(BaseXxx): ...
```

---

## 3. 技术选型与理由

### 3.1 PDF 解析：MarkItDown + PyMuPDF
- **MarkItDown**：PDF → Markdown 转换（文本提取）
- **PyMuPDF (fitz)**：PDF 图片提取
- 选型理由：MarkItDown 输出干净的 Markdown，与 RecursiveCharacterTextSplitter 配合良好

### 3.2 切分：LangChain RecursiveCharacterTextSplitter
- 天然适配 Markdown 结构（标题、段落、代码块）
- 可配置 separators 和 chunk_size/overlap
- 由 DocumentChunker 适配器封装（添加 chunk_id、元数据继承、溯源链接）

### 3.3 混合检索：BM25 + Dense + RRF
- **Dense**：OpenAI text-embedding-3 → ChromaDB cosine similarity
- **Sparse**：自建 BM25 倒排索引 + IDF
- **融合**：RRF（Reciprocal Rank Fusion）— 基于排名，不依赖绝对分数
- **精排**：可选 Cross-Encoder 或 LLM rerank，失败回退到 RRF 排名

### 3.4 多模态：Image-to-Text 策略
- Vision LLM（Azure GPT-4o）生成图片文本描述
- 描述注入 chunk 正文 → 复用纯文本检索链路
- 无需 CLIP 或独立图像向量库
- 原始图片本地存储，检索命中后 base64 返回

### 3.5 MCP：Python 官方 SDK + Stdio Transport
- JSON-RPC 2.0，通过 stdin/stdout 通信
- Client 零配置（Copilot、Claude Desktop）
- stdout 仅输出 MCP 消息；日志仅走 stderr

### 3.6 Dashboard：Streamlit 六页面应用
- 系统总览 | 数据浏览器 | Ingestion 管理 | Ingestion 追踪 | Query 追踪 | 评估面板
- 追踪页面读取 traces.jsonl；数据页面直接访问存储层
- 基于 trace 中 method/provider 字段动态渲染，更换组件后自动适配

---

## 4. 关键实现注意事项

### 4.1 Chroma + BM25 双存储
Chroma 不原生支持 sparse vector。解决方案：
- **Dense 路径**：Chunk 内容 + 稠密向量 + 元数据 → ChromaDB upsert
- **Sparse 路径**：词频 + IDF → 自建 BM25 倒排索引（pickle 文件）
- **查询时**：并行 Dense（Chroma）+ Sparse（BM25）→ RRF 融合
- **SparseRetriever** 需要 `VectorStore.get_by_ids()` 在 BM25 返回 chunk_id 后补全文本

### 4.2 libs/evaluator/ 与 observability/evaluation/ 的区分
- `src/libs/evaluator/`：抽象契约 — BaseEvaluator、EvaluatorFactory、CustomEvaluator
- `src/observability/evaluation/`：业务逻辑 — RagasEvaluator、CompositeEvaluator、EvalRunner
- 原则：libs/ 定义接口；observability/ 实现具体评估器

### 4.3 Transform 降级链
三个 Transform（ChunkRefiner、MetadataEnricher、ImageCaptioner）共享模式：
- 规则默认 → 可选 LLM 增强 → LLM 失败时优雅降级
- metadata 标记：`refined_by: "rule"|"llm"`，降级时记录原因
- 单个 chunk 处理失败不阻塞其他 chunk

### 4.4 Chunk ID 生成
确定性算法：`hash(source_path + chunk_index + content_hash[:8])`
- 相同内容 → 相同 ID（幂等 upsert）
- 内容变更 → 新 ID
- DocumentChunker 中的格式：`{doc_id}_{index:04d}_{hash_8chars}`

### 4.5 SQLite 统一约定
所有 SQLite 模块遵循：
- WAL 模式保证并发安全
- 首次访问时自动建表
- 统一的错误处理模式
- 存储位置：`data/db/` 目录

---

## 5. 环境与依赖

### 5.1 可用 API 服务
- **Azure OpenAI**：LLM（GPT-4o）+ Embedding + Vision — 已配置
- **OpenAI API**：LLM + Embedding — 已配置
- Ollama：架构支持，但未确认本地环境是否就绪

### 5.2 已安装依赖
```
pyyaml    # 配置加载（Phase A3）
pytest    # 测试框架（Phase A2）
```

### 5.3 待安装依赖（Phase B 及以后）
```
openai                    # LLM Provider SDK
langchain-text-splitters  # RecursiveCharacterTextSplitter
chromadb                  # 向量存储
mcp                       # 官方 MCP SDK
markitdown                # PDF → Markdown
PyMuPDF                   # PDF 图片提取
streamlit                 # Dashboard
ragas                     # RAG 评估框架
sentence-transformers     # Cross-Encoder reranker（可选）
python-dotenv             # .env 加载
pytest-asyncio / pytest-mock / pytest-cov  # 测试增强
```

### 5.4 Python 版本
- 要求：3.10+
- 虚拟环境：`.venv` 已存在

---

## 6. Phase A 实现经验

### 6.1 Settings 设计
- `frozen=True` 保证不可变性 — 全局配置只读，任何修改需重新加载
- 可选节（ingestion/vision_llm/dashboard）用 `Optional[XxxSettings] = None` — 最小配置即可启动
- 参考实现中 ingestion 要求 `splitter` 和 `batch_size` 字段但 YAML 中未定义 — 我们用默认值修复了这个不一致

### 6.2 测试模式
- 冒烟测试用 `from src.xxx import yyy` 验证包结构完整性
- 配置测试用 `tmp_path` fixture 创建临时 YAML — 避免污染真实配置
- YAML section 删除测试需要整块移除（key + indented children），简单 replace 会破坏 YAML 语法

### 6.3 路径解析
- `REPO_ROOT = Path(__file__).resolve().parents[2]` — 锚定到 settings.py 文件位置
- `DEFAULT_SETTINGS_PATH` 使用绝对路径 — CWD 无关
- `resolve_path()` 辅助函数处理相对路径 → 绝对路径转换

---

## 7. 待确认问题

- [ ] MarkItDown 具体版本及其与 PyMuPDF 图片提取的协调方式
- [ ] Cross-Encoder 模型在本地环境的可用性（sentence-transformers 下载）
- [ ] Ragas 版本与当前 OpenAI SDK 的兼容性
- [ ] Streamlit AppTest API（Phase I2 冒烟测试使用）
