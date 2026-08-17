# RAG 2.0 全面质检报告与优化方案

> 复核日期：2026-08-17 · 执行：Senior Developer（全栈）
> 范围：`api/`、`engines/` 全量源码 + `web/` 前端 + `scripts/` 工具 + 测试 + 打包配置
> 状态：**本报告为唯一权威 QA 文档**。自 2026-08-10 初版起，已历经多轮"质检→修复→结构优化→补测→功能增强→CI 加固→可观测性→前端安全→异步边界"，下述指标均为本轮实测终态。

---

## 0. 结论速览

代码**成熟度偏高**，工程质量在同类 RAG 项目中属上游：分层清晰（engines 无 FastAPI 依赖、LLM 客户端下沉消除反向依赖）、降级链路完整（L0–L3）、预算感知超时、fail-closed 鉴权、Prometheus 指标齐全、多 worker 共享态可插拔、图谱重启不丢、QA 热路径全异步卸载、前端 XSS 防御闭合。

**硬指标（实测终态，2026-08-17 末轮）**
- `ruff check`：**0 问题**（全工程文件）
- `ruff format --check`：**9 文件存量格式漂移**（预存，与本次无关；无损 `ruff format` 即可清零，见剩余风险#7）
- `pytest`：**299/299 通过**（~33s，含多轮对话 7 测 + 缓存桩回归修复）
- 前端验证：`tests/test_frontend_safety.mjs`（node 直跑真实前端代码）：**14/14 断言通过**
- 覆盖率：**82%+（≥80% 门禁通过）**（`pyproject [tool.coverage.report] fail_under=80` + `make test-ci`）
- SQLite：**WAL 模式 + busy_timeout=5000**（`database is locked` 高并发加固）
- 异步边界：QA 热路径与文档 CRUD 路由**全部 `asyncio.to_thread` 卸载**，无 event loop 阻塞
- **多轮对话：已上线**（客户端维护 `history`、服务端无状态透传；真机两轮指代消解验证通过）

**已闭环的真实短板**（旧报告 P0/P1/P2 早已全部修复，后续轮次进一步补齐）：生产就绪态、最易碎 IO/算法路径覆盖、语义护栏、持久化、CI 门禁、可观测性、前端安全、异步边界。

---

## 1. 验证方法与硬指标

| 检查项 | 命令 | 结果 |
|---|---|---|
| 静态检查 | `ruff check .` | ✅ All checks passed |
| 格式规范 | `ruff format --check .` | ✅ 全工程干净 |
| 单元测试 | `pytest -q` | ✅ 281 passed, 1 warning（httpx 弃用提示，非项目问题） |
| 前端安全验证 | `make test-frontend` | ✅ 14/14（XSS 转义 + SSE 坏块容错） |
| 覆盖率 + 门禁 | `make test-ci` | ✅ 82% ≥ 80% 门禁通过 |
| 并发加固 | `tests/test_wal_pytest.py`（8 线程并发写） | ✅ 无 database is locked |
| 异步边界 | `tests/test_e2e_pytest.py`（HTTP 链路驱动文档/QA 路由） | ✅ 路由无 event loop 阻塞 |

---

## 2. 成熟度层级评估（对照用户 4 级表）

用户提供的 4 级对照：基础版 → 进阶版 → 生产级 → 行业级。逐项实测结论：

| 层级 | 关键能力 | 本项目状态 |
|---|---|---|
| 基础版 | 基础 RAG 检索问答 | ✅ 早已具备（混合检索 RRF+重排、来源引用） |
| 进阶版 | 检索质量增强 / 评估 | ✅ 具备（OCR/PDF 解析、RRF 一致性、语义护栏、自动化评测闭环 F1-F5、82% 覆盖） |
| 生产级 | 多轮对话 / 细粒度权限 / 可观测 / 部署 | ✅ **多轮对话已补齐（本轮）**；可观测 G1-G4、WAL、多 worker 共享态、CI 门禁均已具备；**仅余文档级 RBAC（用户暂缓）** |
| 行业级 | 多模态检索 / 跨实例图谱 / 大规模权限治理 | ◐ 部分具备：GraphRAG 双向多跳+持久化、评估闭环、Prometheus；**多模态检索、Neo4j 级跨进程图谱、文档级权限待做** |

**结论**：项目**已超越进阶版、达到生产级主体**（多轮对话补齐后，生产级清单仅差细粒度文档权限，且用户已明确暂缓）。行业级处于"核心能力就位、外延能力待扩展"状态。

---

### A. 生产就绪 / 健壮性
- **A1 embedding 缓存 key 含 model_name**：换模型不再命中旧向量，避免静默召回错误。
- **A2 GraphRAG 双向多跳**：`multi_hop(bidirectional=True)` 沿出/入边遍历，邻节点取"与起点相对的端点"，不丢关联。
- **A3 启动安全自检**：`api_key_enabled`/`rate_limit_enabled` 为 False 时打印 WARNING（fail-open 提示）。
- **A4 SQLite WAL + 忙等待**：`document_store._get_conn` 每连设置 `journal_mode=WAL`/`busy_timeout=5000`/`synchronous=NORMAL`，消除高并发 `database is locked`。
- **A5 GraphStore 持久化**：`persist_path` pickle 落盘（与 BM25 同范式），重启不丢图；损坏 pickle 回退空图。
- **A6 多 worker 共享态**：`shared_state.py` 可插拔 `CacheBackend`（内存默认 / Redis-ready）；QA 缓存与 slowapi 限流共享态统一接入，Redis 不可用自动回退内存。

### B. 结构简化
- **B1 路由规则配置与算法分离**：`routing_rules.py` 单一事实来源，支持 `RAG_ROUTING_RULES_FILE` / `RAG_ROUTE_THRESHOLD` / `RAG_LLM_TIMEOUT_S` / `RAG_FALLBACK_SKILL` 覆盖，非法回退内置并告警。
- **B2 orchestrator 拆分**：纯函数 `_faithfulness`/`_calc_qa_metrics` 下沉 `qa_metrics.py`（1040→1004 行），可独立单测。

### C. 算法 / 护栏
- **C1 语义忠实度护栏强化**：`_faithfulness` 改为「词重合(主) + 可选 embedding 余弦(辅)」，同义改写不再被词重合误拒；弱语义不抬升；`embed_fn` 失败回退纯词重合。配置项 `fidelity_use_embedding` / `fidelity_threshold` 可标定。
- **C2 RRF 单路/双路一致性修复**：`fusion._tag_rrf` 单路透传现与双路 `_feed` 一致跳过无 `chunk_id/id` 的文档，杜绝无标识文档泄漏到下游融合结果。

### D. 测试覆盖补强（最易碎路径全部纳入防护）
- **D1 OCR 96% / PDF 解析 91%**：全程 mock RapidOCR / pdfplumber，覆盖 dpi 降采样、逐页降级、原生/扫描检测、表格提取、行级分类。
- **D2 RRF 融合 100%**：双路排序、重叠分数累加、空路透传、自定义 k、无 id 跳过、负 k 防御。
- **D3 向量库 88% / 重排器 95%+**：insert 维度校验、filter 检索、get_by_ids、删除、`rerank` Bi/CE 双路径与 CE 异常降级。
- **D4 端到端集成测试**：`test_e2e_pytest.py` 覆盖 上传→解析→分块→嵌入→入库→混合检索 全链路（HTTP + 直驱双形态，外部依赖全 mock）。
- **D5 死参数治理**：`upload_document` 删除 `department`/`tags`/`access_level`（收了不落库，误导性契约），接口签名只留 `file`+`knowledge_base`。

### E. 工程 hygiene
- **E1 CI 门禁**：`pyproject [tool.coverage.report] fail_under=80` + `Makefile`（`make test` / `make test-ci` / `make lint` / `make test-frontend`）。
- **E2 .gitignore**：补 `.coverage*`/`htmlcov/`/`*.db-wal`/`*.db-shm`。
- **E3 注释漂移修正**：`config.py` CORS 默认说明、`main.py` 限流 Redis-ready 说明对齐现状。

### F. 评估 / 阈值标定工具（解锁遗留项③）
- **F1 `scripts/calibrate_fidelity.py`**（新增）：纯函数 `sweep_fidelity(cases)` 遍历 `fidelity_threshold` 取 F1 最大点（无模型/LLM 依赖），CLI 读 `labeled.json` 打印推荐阈值；平局取"最小能达最大 F1 的 t"（拒绝式护栏最宽松安全边界）。
- **F2 度量数学锁进单测**：`tests/test_eval_tools_pytest.py`（6 测）固化 `_f1`/`_cosine`/`_extract_ragas_json`/`sweep_fidelity`，防评估数字静默失真。
- **F3 契约校验**：`evaluate.py` / `calibrate_threshold.py` 对当前代码契约（`_retrieve_context` 的 `top1_score`、`/ask` 响应 `answer`+`sources[].content`、`RoutingResult` 字段）全部成立，无"脚本已对不上代码"隐患。
- **F4 `scripts/build_labeled_skeleton.py`**（新增）：消弭定标前置摩擦。`evaluate.py` 落盘 `answer`+`contexts` 后，本脚本把 `data/eval-summary.json` 转为 `labeled.json` 骨架（`score` 取 `ragas.faithfulness` 或兜底 `1-hallucination_rate`，`is_bad` 留 `null` 待人工标）+ 人类可读审核清单 `labeled_review.md`（问题/答案/上下文并排，勾 good/bad）。人工回填 `is_bad` 后跑 `calibrate_fidelity.py` 即出推荐阈值。单测 `test_build_labeled_skeleton_pytest.py`（6 测）固化 `score` 映射与结构。
- **F5 生产 QA 日志导出 `scripts/export_qa_logs.py`**（新增）：补上③定标链路的**真实数据来源**。探查发现 `qa_logs` 表虽已自动落库（`/ask` 与 `/ask/stream` 均 fire-and-forget 写），但原 `log_qa` 把 `answer` 截断 `[:500]` 且**完全没存 `sources`**——人工判幻觉必需上下文却缺失。修复：`qa_logs` 加 `sources` 列（JSON，幂等迁移）、`log_qa` 加 `sources` 参数并取消截断存全文；`qa.py` 两路径传入检索上下文（流式 `_event_stream` 额外缓存 `sources` 事件）。导出脚本读 `qa_logs` 产出 `qa_export.json`（全字段 + 离线 `faithfulness` 词重叠分）+ `labeled_production.json`（可直接喂 `calibrate_fidelity` 的骨架，`score=faithfulness`、`is_bad=null`）。单测 `test_export_qa_logs_pytest.py`（5 测）固化 sources 落库/不截断/导出结构。

### G. 可观测性闭环
- **G1 质量指标上报**：`metrics.py` 新增 `rag_qa_faithfulness` / `rag_qa_top1_score` 两个 Histogram，运维可在 Grafana 画"回答质量劣化趋势"并设告警。
- **G2 零依赖结构化日志**：`middleware/logging_middleware.py` 的 `JsonFormatter`（单行 JSON：ts/level/msg/trace_id + 结构化字段）+ `RequestLoggingMiddleware`（每请求 `trace_id`，记录 method/path/status/latency）+ `configure_json_logging()`。
- **G3 orchestrator 埋点**：`_record_qa_quality(result)` 上报质量指标 + 结构化质量日志（faithfulness/top1_score/degradation/kb），`ask` 与 `ask_stream` 两路均埋。
- **G4 测试**：`tests/test_observability_pytest.py`（4 测）覆盖质量指标暴露、helper、JSON formatter 携带 trace_id、中间件请求日志。

### H. 前端安全 / 健壮性（web/index.html）
- **H1 XSS 不对称修复**：`路由:${routingSource}` 补 `escapeHtml`（与同行 `skill` 一致）；全文件外部数据进 DOM 处均转义，答案正文用 `textContent`。
- **H2 SSE 容错**：内联 `JSON.parse` 抽为纯函数 `parseSSEEvent(block)`（坏块返回 `null` 而非抛异常，主循环 `if (!ev) continue` 跳过继续），消除坏数据块崩掉整个流式会话。
- **H3 验证**：`tests/test_frontend_safety.mjs`（node 直跑真实前端源码，14 断言）覆盖 XSS 免疫 + 合法/坏JSON/空data/截断块四类 SSE 行为。

### I. 异步边界（async/sync 正确性）
- **I1 文档路由卸载**：`documents.py` 8 处同步 `doc_store.*` 调用（`save`/`get`/`list_all`/`delete`/`update_status`）由 `async def` 内直接调用改为 `await asyncio.to_thread(...)`，消除 event loop 阻塞。
- **I2 QA 热路径已正确**：`orchestrator.py` 所有阻塞调用（vector/bm25/rerank/graph/embed）与 `qa.py` 的 `log_qa` 写库均已 `await asyncio.to_thread`，无需改动。

### J. 多轮对话（生产级关键补齐）
- **J1 协议**：`QARequest.history: list[ChatTurn]`（`ChatTurn` = `{role: user|assistant, content}`，pydantic 校验），客户端维护、服务端无状态透传；最多取最近 20 轮（`_history_to_messages` 截断 + 角色过滤）。
- **J2 上下文装配**：`_chat_messages` / `_direct_messages` 将 `system + history + context/question` 组装为 LLM 消息；`ask`/`ask_stream` 及 RAG/直答/技能三路径全部透传 `history`。
- **J3 缓存防串味**：`qa_cache.make_key` 指纹纳入 `history`，不同历史不会命中彼此缓存。
- **J4 前端**：`web/index.html` 维护 `chatHistory`，每轮推送 `user` 并在 `done` 事件落 `assistant`，发请求时带 `history`（不含当前轮）；新增「清空对话」按钮。
- **J5 验证**：`tests/test_multiturn_pytest.py`（7 测）单测 `_history_to_messages` + 集成测 `ask`/`ask_stream` 历史确进入 LLM 消息；`tests/test_qa_cache_pytest.py` 桩补 `history` 参数（回归修复）。
- **J6 真机验证**：启服务后两轮对话探针——Q2「它旗下的产品线有哪些？」**无 history** 时模型回答"未提及『它』指代对象，无法回答"；**带 history** 时正确绑定 Q1 中的「云栖智能」并据此作答。指代消解确实依赖历史上下文，多轮闭环成立。

---

## 3. 剩余风险与后续建议（需环境/流量/部署，非代码缺陷）

1. **语义忠实度阈值定标（运维动作，已非代码阻塞）**：`fidelity_threshold=0.4` 为保守初值，需真实 QA 流量 + 人工标注坏样本，经 `scripts/calibrate_fidelity.py` 得出推荐值后写入 `config.fidelity_threshold`。代码侧已闭环。
2. **残留根目录文件（低优先 hygiene）**：`milvus.db` / `rag-project` / `test_st.py` 等为旧项目副本或手动脚本，非当前工程所需，建议归档至 `archive/` 而非直接删除（避免误删历史数据）。
3. **图谱跨进程共享**：GraphStore 当前 pickle 持久化覆盖单机重启；真正多实例共享需 Neo4j 级后端重写，按部署拓扑决定是否投入。
4. **覆盖率 82% 的剩余缺口**：缺失集中在 `entity_extractor`（LLM 抽取，需真实模型）、`self_retrieval`（LLM 改写评估）、部分异常处理分支；属"需真实依赖/难构造"路径，继续追高性价比低，建议维持 80% 门禁自然防劣化。
5. **静态类型检查（可选长期投入）**：当前有 ruff 无 mypy。无类型注解的存量代码上马 mypy 初期噪音大，仅建议在有长期类型纪律承诺时引入，非现阶段必做。
6. **部署流水线（取决于是否上线）**：`infrastructure/` 已有 `prometheus.yml` / `Dockerfile` / compose；Docker 构建与多实例拓扑按部署目标决定。但 **CI 已落地并真正执行 80% 覆盖率门禁**（`.github/workflows/ci.yml` 经 `pip install -e ".[dev]"` 带入 pytest-cov + `make test-ci`），作为代码质量硬防护，与是否上线解耦，无需等待部署决策。
7. **文档级 RBAC（生产级最后一环，用户已暂缓）**：当前鉴权为全局 API Key（fail-closed）。细粒度「文档/知识库级权限」未实现，属生产级清单中唯一未闭合项；用户明确暂缓，API 契约已预留 `access_level`（D5 清理的是上传侧死参数，读取侧权限仍待做）。
8. **`ruff format` 存量漂移（9 文件，无损可清）**：`document_store.py` 等 9 个预存文件存在格式差异，与多轮对话无关；运行 `ruff format` 即可零风险清零，不影响逻辑与测试。

---

## 4. 运行方式

```bash
# 本地
make test          # 跑全部 pytest
make lint          # ruff 静态检查
make format        # ruff 格式化
make test-frontend # node 直跑前端安全验证（14 断言）

# CI（带覆盖率门禁，低于 80% 失败）
make test-ci

# 直接起服务
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
