# CLAUDE.md — RAG 2.0 项目规范

## 架构概览

```
用户请求 → FastAPI (main.py)
         → 限流 (Redis + slowapi)
         → API Key 认证中间件
         → 路由层 (documents / qa / evaluate)
         → 引擎层 ─┬─ parsing (PDF + OCR)
                   ├─ chunking (语义分块)
                   ├─ embedding (BGE-small-zh, 512d)
                   ├─ retrieval (向量 + 图谱 + 重排 + Self-RAG)
                   └─ graph_rag (实体抽取 + Neo4j)
         → LLM 生成 (OpenAI 兼容 API, 重试 + 连接池)
         → 错误脱敏中间件
         → 响应 (answer + sources + latency_ms)
```

## 启动命令

```bash
# 本地开发
cd rag-2.0
.\venv\Scripts\Activate.ps1
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:TRANSFORMERS_OFFLINE="1"
$env:HF_HUB_OFFLINE="1"
$env:RAG_LLM_API_KEY="your-key"
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Docker 全栈 (Milvus + Neo4j + Redis + Postgres + MinIO + Celery)
docker-compose up -d
```

## 测试

```bash
pytest                                  # 全部测试
pytest tests/test_graph.py              # 单个文件
pytest -m "not slow and not integration"  # 跳过慢测试和集成测试
pytest --cov=api --cov=engines --cov-report=term-missing
```

## 设计决策速查

详细对比见 `docs/design-decisions.md`。核心决策：

| 决策点 | 选择 | 排除的替代方案 |
|--------|------|---------------|
| Embedding | BGE-small-zh (512d) | text2vec-large-chinese (1024d, 2x内存, 精度提升<3%) |
| 分块策略 | 语义相似度 + 标题树动态切分 | RecursiveCharacterTextSplitter (无语义感知) |
| 向量存储 | LanceDB (开发) / Milvus (生产) | Chroma (无原生中文索引优化), Qdrant (运维复杂) |
| 重排序 | Cross-Encoder (bge-reranker-base) | LLM重排 (延迟10x+, 成本高) |
| 图谱抽取 | 规则(TECH_PATTERNS) + LLM 混合 | 纯LLM (成本高, 常见技术实体不需要LLM) |
| LLM客户端 | 自定义 httpx (重试+连接池) | LangChain LLMChain (抽象层过厚, 调试困难) |

## 项目特有约束

- Python >= 3.12, FastAPI >= 0.115, Pydantic Settings v2
- Embedding 模型下载走 HuggingFace 国内镜像 `HF_ENDPOINT=https://hf-mirror.com`
- LLM 默认走 TokenHub API (`https://tokenhub.itcast.cn/v1`)，Key 通过环境变量注入
- 文档处理走 Celery 异步任务，不在请求线程内执行
- LanceDB 数据目录在 `./lancedb_data`，不纳入 git
- OCR 默认 CPU 模式，GPU 需额外配置 `RAG_OCR_USE_GPU=true`

## 开发约定

- 配置: 所有配置通过 `api/config.py` 的 Pydantic Settings 管理，环境变量前缀 `RAG_`
- 单例: Embedding 模型用模块级懒加载单例 (`engines/embedding/embedder.py`)
- 降级: 所有外部依赖必须有降级策略 (参考 `docs/design-decisions.md` 降级链章节)
- 日志: 使用结构化日志，不要用 `print`
- LLM 调用: 通过统一 LLM 客户端，不要直接调 httpx
- 状态管理: 全局单例通过 `api/state.py` 获取，在 `lifespan` 中初始化

## 三层评估自查清单

每次功能变更后自检：

**设计层:**
- [ ] 技术选型有 >=2 条排除替代方案的理由
- [ ] 参数值有测试依据或文献引用（不能是"感觉这个值合适"）
- [ ] 更新了 `docs/design-decisions.md`

**工程层:**
- [ ] 新外部调用有降级策略
- [ ] 新 API 端点有对应的测试
- [ ] 变更有延迟影响评估（会不会让 QA 接口从 200ms → 2s？）
- [ ] Token 消耗可追踪（至少日志记录）

## 待补工程化事项

优先级排序（详见 `docs/design-decisions.md` 工程化章节）:

1. **P0** — 性能基准测试 (QPS / P50 / P99)
2. **P0** — Token 成本追踪 (每次 QA 消耗统计)
3. **P1** — 压测脚本 (locust/k6)
4. **P1** — LLM 熔断机制 (连续失败 N 次后快速失败)
5. **P1** — QA 结果缓存 (Redis, 相同问题返回缓存)
6. **P2** — Prometheus + Grafana 监控
7. **P2** — GraphRAG 批量抽取 (减少 LLM 调用次数)
8. **P2** — 补充缺失测试 (test_qa.py, test_reranker.py, test_self_retrieval.py)
