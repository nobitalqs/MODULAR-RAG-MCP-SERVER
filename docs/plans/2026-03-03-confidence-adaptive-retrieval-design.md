# Confidence-Based Adaptive Retrieval Design

> **动机**: 当前管线 Recall@5 平均 67%。部分 case 相关内容存在但排在 top_k 窗口之外。通过 reranker 分数判断结果置信度，低于阈值时自动扩大检索窗口重试，零 LLM 开销。

## 核心思路

```
Query → Search(top_k) → Rerank → 检查 top-1 score
  ├─ score >= threshold → 返回
  └─ score < threshold → Search(top_k * expand_factor) → Rerank → 返回
```

最多重试 1 次，避免延迟爆炸。

## 设计决策

### 置信度指标
- 使用 **reranker top-1 score** 作为置信度信号
- CrossEncoder 输出 logit 分数：正值 = 高相关，负值 = 低相关
- 阈值建议：`0.0`（基于 golden test 观察：好结果 score > 5.0，差结果 score < -1.0）

### 参数（全部 settings.yaml 可配）

```yaml
retrieval:
  adaptive:
    enabled: true               # 开关
    score_threshold: 0.0        # reranker top-1 score 低于此值触发重试
    expand_factor: 2            # 重试时 top_k 乘以此因子
    max_retries: 1              # 最多重试次数（建议 1，避免延迟）
```

### 适用条件
- `rerank.enabled = true`（无 reranker 时无分数可用，跳过自适应）
- 首次检索结果非空（空结果说明数据库没内容，重试无意义）
- 仅在 multi-query RRF fusion + rerank 之后触发

### 不适用时的降级
- rerank 关闭 → 直接返回（Null 行为）
- adaptive.enabled = false → 直接返回
- 首次结果为空 → 直接返回

## 实现位置

修改 `src/mcp_server/tools/query_knowledge_hub.py` 的 `execute()` 方法。

**插入点**：在 rerank 之后、`results[:effective_top_k]` 之前：

```python
# 现有代码
if self.config.enable_rerank and results:
    results = await asyncio.to_thread(
        self._apply_rerank, query, results, effective_top_k, trace,
    )

# === 新增：Confidence-Based Adaptive Retrieval ===
# if top-1 score < threshold and retry budget > 0:
#     expand top_k, re-search, re-fuse, re-rank
# ================================================

# Enforce top_k contract
results = results[:effective_top_k]
```

### 新增组件

1. **AdaptiveRetrievalConfig** — frozen dataclass，加到 Settings
2. **`_should_retry(results, threshold)`** — 判断是否需要重试
3. **`_adaptive_retry(query, search_queries, ...)`** — 执行扩大检索

不新建文件，全部在 `query_knowledge_hub.py` 内完成（<50 行新代码）。

## Trace 集成

重试时记录到 trace：
```python
trace.metadata["adaptive_retry"] = True
trace.metadata["adaptive_reason"] = f"top1_score={top1_score:.2f} < threshold={threshold}"
trace.metadata["adaptive_expanded_top_k"] = expanded_top_k
```

Dashboard 可直接展示这些字段。

## 测试计划

1. **test_adaptive_triggers_on_low_score** — top-1 < 0.0 → 重试，search 被调用 2 次
2. **test_adaptive_skips_on_high_score** — top-1 > 0.0 → 不重试，search 调用 1 次
3. **test_adaptive_skips_when_disabled** — enabled=false → 不重试
4. **test_adaptive_skips_when_rerank_off** — rerank 关闭 → 不重试
5. **test_adaptive_max_retries_respected** — 只重试 max_retries 次
6. **test_adaptive_trace_metadata** — trace 包含 adaptive_retry 字段

## 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 阈值选不好导致误触发 | 中 | 延迟翻倍 | settings.yaml 可调 + trace 可观测 |
| 重试仍然得不到更好结果 | 低 | 浪费一轮检索 | max_retries=1 限制最大代价 |
| CrossEncoder 分数分布因模型不同而变化 | 中 | 阈值需要重新校准 | 文档说明 + 建议用 golden test 校准 |
