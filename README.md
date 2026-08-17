# RAG 2.0 — 企业级检索增强生成系统

> 混合检索（向量 + BM25/RRF 融合）+ GraphRAG 关系推理 + 多技能路由 + 重排序 + 流式生成 + 4 级降级，带真实测试套件与可观测性。

![Tests](https://img.shields.io/badge/tests-166%20passed%20%2F%20175%20collected-10b981)
![Python](https://img.shields.io/badge/python-3.12%2B-4f6ef7)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 📌 诚实定位（请先读）

本项目**核心算法与工程质量优秀**，但**生产化基础设施（Milvus / Neo4j / Celery / Redis / K8s）为设计中 / 规划项（Phase 4–6），代码中尚未实现**。

- ✅ 适合：内部 MVP、技术演示、作品集、RAG 全链路学习
- ⚠️ 当前形态：单 API 容器 + 本地文件存储（LanceDB），图谱 / 限流 / 缓存为内存态（重启即丢），不支持水平扩展
- ⚠️ 实测 SLA：P99 ≈ 7.7s @ 20 并发（受 deepseek-v4-flash 强制 reasoning 影响），单机 QPS ≈ 3

**请按"生产级 MVP / 内部可部署"对外表述**，勿将规划中的微服务架构描述为"已实现"。

---

## ✨ 核心特性

- **混合检索**：稠密向量（BGE-small-zh）+ BM25 稀疏，RRF 融合；跨知识库兜底
- **GraphRAG**：关系规则抽取 + 内存图谱推理，补充向量检索的结构化上下文
- **多技能路由**：自动选择 客服库 / 技术库 / 直接回答，附降级与跨库徽章
- **重排序**：交叉编码器重排，提升 top-k 相关性
- **流式生成**：SSE 逐字输出，首 token 快、体验好
- **可靠性工程**：分阶段超时预算、L0–L3 四级降级、LLM 熔断、Embedding/QA 缓存、忠实度护栏（防幻觉）、`reset_stale_processing`
- **安全**：API Key 常量时间比对（`secrets.compare_digest`）、缺失 fail-closed、错误脱敏、CORS 收紧
- **可观测性**：`/metrics`（Prometheus）、Grafana dashboard、结构化日志、Token 用量落库
- **测试**：175 collected / 166 passed（解析/分块/嵌入/检索/重排/图谱/路由/API/流式/缓存）

---

## 🏗️ 架构

### 当前（已实现，单容器）

```
API(FastAPI) → Orchestrator → Engines(检索/重排/图谱/路由/生成)
                        │
        LanceDB(文件) · BM25(内存) · 图谱(内存) · 限流/缓存(内存)
```

`engines/` 零 FastAPI 依赖，可独立复用；`engines/interfaces.py` 有 ABC 抽象便于替换实现。

### 规划（Phase 4–6，设计完成 / 未实现）

| 组件 | 用途 | 状态 |
|------|------|------|
| Milvus | 分布式向量库 | ⏳ 设计 |
| Neo4j | 图谱持久化 | ⏳ 设计 |
| Celery + Redis | 异步文档处理 / 集中缓存限流 | ⏳ 设计 |
| PostgreSQL | 关系库 + 多租户 | ⏳ 设计 |
| Kubernetes | 编排 + 多副本 | ⏳ 设计 |

---

## 🚀 快速开始

```bash
# 本地
pip install -e .
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:RAG_LLM_API_KEY="your-tokenhub-key"
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Docker（单容器）
docker-compose up -d --build
# 生产 profile（非 root + 资源限制 + 健康检查）
docker-compose -f docker-compose.prod.yml up -d --build
```

打开 `http://localhost:8000` → 上传 PDF → 提问。

> 生产环境务必在 API 前加反向代理做 TLS 终止（见 `docs/deployment.md` §6）。

---

## 📊 测试

```bash
pytest -m "not slow and not integration" -q
# 175 collected / 166 passed / 9 deselected
```

CI：push / PR 自动跑 `ruff` + `pytest`（见 `.github/workflows/ci.yml`）。

---

## 📁 文档

- `docs/prd.md` — 产品需求
- `docs/design-decisions.md` — 设计决策与替代方案对比
- `docs/performance-benchmark.md` — 真实性能基准
- `docs/deployment.md` — **诚实部署文档（当前 vs 规划）**
- `docs/quality-check-report.md` — 自检报告（注意：其中"全微服务 A- 92 分"与代码不符，以本文档为准）

---

## 🗺️ 路线图（Phase 4–6）

1. 向量库切换 Milvus（或 LanceDB 单机压测）
2. 图谱持久化 Neo4j
3. Celery + Redis 异步文档处理
4. 多租户 + RBAC
5. K8s 编排

---

> 本项目用于技术展示与学习。生产化需完成上述路线图，并补充 HTTPS / CI-CD / 多副本。
