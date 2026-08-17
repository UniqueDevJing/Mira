# QA 超时预算分析

**目的**: 验证分阶段超时配置下，最坏情况总延迟是否满足业务 SLA。

> ⚠️ **2026-08-16 警示**: 全局预算已放宽至 12s / LLM 生成 8s（原 5s/2s，见 `performance-benchmark.md` 08-10 修复）。实测 20 并发 **P99 7.7s**，远超文档原承诺的 "P99 < 3s"，SLA 需重定标。LLM 单请求 6.2s 占总延迟 95%，是唯一瓶颈。

当前阈值/超时均通过环境变量可调（`RAG_*`，见 `api/config.py`）。

## 阶段超时配置

| 阶段 | 预算 | 实际典型耗时 | 触发降级 |
|------|------|-------------|---------|
| Router 规则 | 50ms | ~0ms | — |
| Router LLM 分类 | 1.5s | 200-500ms | → fallback tech |
| Embedding (缓存+重试×2) | 2s | 20ms/命中, ~20ms/未命中(CPU) | 跳过向量 |
| 向量 + BM25 并行 | 0.8s | 12-180ms | 向量超时 → 仅 BM25 (L2) |
| 跨库兜底 (并行) | 0.6s | 0-600ms | 超时放弃 |
| Rerank | 0.5s | 25-50ms | 跳过 (L1) |
| LLM 生成 | 8s | 500ms-6.2s | 检索摘要 (L3) |

全局预算: **12s**（原 5s，08-10 放宽）。每阶段实际超时 = min(阶段值, 全局剩余)。

## 最坏情况延迟链路（按阶段预算上限累加）

```
Router 规则 0.05s (或 LLM 分类 1.5s)
+ Embedding 2s
+ 向量+BM25 0.8s
+ 图谱检索 0.5s
+ 跨库兜底 0.6s
+ Rerank 0.5s
+ LLM 生成 8s
────────────────────────
最坏上限 ≈ 12.4s (规则直通) / 13.9s (LLM 路由) — 均超 12s, 靠预算削减钉死
```

**结论**: 各阶段预算上限简单相加 > 全局 12s。因此必须依赖「每阶段取 min(阶段值, 全局剩余)」的预算削减机制，而非简单相加。最坏情况下全局 12s 到点后，后续阶段直接降级返回，总延迟被钉死在 12s 内（实测 P99 7.7s，见 `performance-benchmark.md`）。

## 预算削减机制验证（应满足）

| 场景 | 期望总延迟 | 削减行为 |
|------|-----------|---------|
| 全阶段慢 | ≤ 12s | 到点即降级，不无限等待 |
| 仅 LLM 慢 | ≤ 10s | Rerank 后剩余预算 > 8s 才给 LLM 全预算 |
| 仅向量慢 | ≤ 3s | 0.8s 超时 → BM25 兜底 (L2) |
| Router LLM 挂 | ≤ 2.5s | 1.5s 超时 → fallback，余预算给检索+生成 |

## 压测验证（✅ 已做 2026-08-10）

`scripts/load_test.py` 两轮压测结果见 `docs/performance-benchmark.md`。结论:
- 20 并发 QPS 3.0, P99 7.7s, 0 失败 0 降级（embedding 快路径后）
- 40 并发 QPS 降至 2.8 且 28% L3 降级 — TokenHub 上游并发受限, 线上建议 ≤20 并发
- 优化方向: LLM 流式化首 token 快返（待办）、多 worker

## 跨库阈值校准（✅ 已做 2026-08-15）

`scripts/calibrate_threshold.py`: 18 条标注扫 t∈[0.3,0.8], F1 打平, 0.50 精度更优（1.0 vs 0.4）→ 默认 0.55→0.50。标注集偏小, 扩充后重跑。阈值经 `RAG_CROSS_KB_THRESHOLD` env 可调。

---

## 相关可观测指标（/metrics）

| 指标 | 用途 |
|------|------|
| `rag_qa_latency_seconds` | 端到端 P50/P95/P99 |
| `rag_retrieval_latency_seconds` | 检索阶段 |
| `rag_rerank_latency_seconds` | Rerank 阶段 |
| `rag_llm_latency_seconds` | LLM 生成阶段 |
| `rag_degradation_levels_total` | L1/L2/L3 触发次数 |
| `rag_routing_sources_total` | 路由来源分布 (rule/llm/fallback/manual) |
| `rag_cross_kb_fallback_total` | 跨库兜底次数 (from_kb/to_kb) |
| `rag_embed_cache_hits_total` / `_misses_total` | Embedding 缓存命中率 |
| `rag_llm_tokens_total` | Token 成本 |
