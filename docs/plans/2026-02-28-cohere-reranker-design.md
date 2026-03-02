# Cohere Reranker Feature Design

## Context

项目需要支持 Azure AI Foundry 部署的 Cohere-rerank-v4.0-fast 模型作为 rerank provider。
同时修复 `create_core_reranker()` 未使用 `RerankerFactory` 的生产环境缺口。

## Architecture Decision

**方案**: httpx 直接调用 Cohere v1/rerank API（零新依赖）

| 替代方案 | 不选原因 |
|----------|---------|
| cohere SDK | 新增 ~50MB 依赖，仅用一个 API |
| azure-ai-inference SDK | 新依赖，项目无先例 |

## API Integration

### Endpoint

```
POST {AZURE_OPENAI_ENDPOINT}/openai/deployments/{deployment_name}/v1/rerank?api-version={api_version}
Headers: api-key: {AZURE_OPENAI_API_KEY}
```

### Request

```json
{
  "query": "user question",
  "documents": ["chunk text 1", "chunk text 2", ...],
  "top_n": 5
}
```

### Response

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.95},
    {"index": 2, "relevance_score": 0.87}
  ]
}
```

### Authentication

- 复用 `AZURE_OPENAI_API_KEY` 和 `AZURE_OPENAI_ENDPOINT`（所有模型共用同一 Azure AI Services 资源）
- 无需新增环境变量

## Components

### 1. `src/libs/reranker/cohere_reranker.py` (NEW)

```python
class CohereReranker(BaseReranker):
    """Azure Cohere Rerank API provider using httpx."""

    def __init__(self, model, top_k=5, api_key=None,
                 azure_endpoint=None, deployment_name=None,
                 api_version="2024-05-01-preview", **kwargs):
        ...

    def rerank(self, query, candidates, trace=None, **kwargs):
        # 1. Extract text from candidates
        # 2. POST to Azure Cohere endpoint
        # 3. Map relevance_score back to candidates
        # 4. Sort desc, return top_k
```

### 2. `src/core/settings.py` — RerankSettings 扩展

新增可选字段（env var fallback）：
- `deployment_name: str`
- `api_version: str`
- `api_key: str` (env: AZURE_OPENAI_API_KEY)
- `azure_endpoint: str` (env: AZURE_OPENAI_ENDPOINT)

### 3. `src/core/query_engine/reranker.py` — 接通 Factory

修改 `create_core_reranker()` 使用 `RerankerFactory`：

```python
def create_core_reranker(settings, reranker=None):
    if reranker is None:
        factory = RerankerFactory()
        factory.register_provider("cross_encoder", CrossEncoderReranker)
        factory.register_provider("llm", LLMReranker)
        factory.register_provider("cohere", CohereReranker)
        reranker = factory.create_from_settings(settings.rerank)
    return CoreReranker(settings=settings, reranker=reranker)
```

### 4. `CoreReranker._get_reranker_type()` 更新

```python
if "Cohere" in class_name:
    return "cohere"
```

### 5. Settings YAML

```yaml
rerank:
  enabled: true
  provider: "cohere"
  model: "Cohere-rerank-v4.0-fast"
  deployment_name: "Cohere-rerank-v4.0-fast"
  api_version: "2024-05-01-preview"
  top_k: 5
```

### 6. `__init__.py` 更新

导出 `CohereReranker`。

## Testing

- Mock httpx 响应，验证请求构造和响应解析
- Factory 集成测试
- CoreReranker 集成测试
- 错误处理（网络超时、API 错误、空文档列表）

## Risk: API Path Uncertainty

Azure cognitiveservices endpoint 的 Cohere Rerank API path 可能不是 `/openai/deployments/{deployment}/v1/rerank`。
如果不对，需要调整为：
- `/v1/rerank`（body 中指定 model）
- 或其他 Azure 特定路径

实现时会让 URL 构造可配置，方便调整。
