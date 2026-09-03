# 多智能体客服框架 · 修订版实施计划（v2）

> 目标：把前一轮审查发现的 **3 个结构性缺陷 + 5 个缺失能力** 落进 `rag-2.0` 的实施计划，
> 精确替换原 P0–P5 中偏弱的步骤。所有改动点都锚定到真实存在的文件/函数（已 grep 核实）。
>
> 配套审查：缺陷 1＝4 Agent 全并行逻辑不自洽；缺陷 2＝置信度闸门太晚且缺"拒答/转人工"终态；
> 缺陷 3＝记忆层坍塌成向量库。缺失 #1＝HITL 交接协议；#2＝成本护栏；#3＝在线学习闭环；
> #4＝Agent 统一接口契约；#5＝多模态融合的视觉真值损失。

---

## 0. 已核实的现有代码锚点（不重写，只增强/新增）

| 模块 | 真实符号（行号） | 在计划中的角色 |
|---|---|---|
| 意图路由 | `engines/router/intent_router.py:32` `class RoutingResult`；`:43` `async def route(self, question) -> RoutingResult` | 增强为带 `intent`/`ambiguous` 的路由（缺陷修复） |
| 路由规则 | `engines/router/routing_rules.py` | 加 `INTENT_MAP` + 歧义阈值 |
| RAG 内核 | `api/core/orchestrator.py:238` `async def ask(...)`；`:919` `ask_stream(...)`；`:614` `def _build_context(docs, top_k=5)`；`:362` `_retrieve_context(...)` | 封装为 **ConsultAgent 内核**，被新编排器调用，不重写 |
| 混合检索 | `engines/retrieval/hybrid_retriever.py:17` `retrieve(query, top_k=20)`；`:39` `_expand_parents(docs)` | 复用（父子文档已落地） |
| 配置 | `api/config.py:6` `class Settings(BaseSettings)` | 新增 `rerank_mode` / 置信度阈值 / 成本预算 等字段（不硬编码） |
| 问答入口 | `api/main.py:114` `app.include_router(qa.router)` → `api/routers/qa.py` 的 `/api/v1/qa/ask` | 加 `image` 入参 + 新增 handoff/feedback 端点 |
| 解析 | `engines/parsing/ocr.py`（RapidOCR） | 与新增 `vision.py` 并列，形成"图→文"通道 |
| 评测 | `scripts/evaluate.py` 的 `_llm_ragas`（已验证可用） | 扩展指标 + 基线校准 |

---

## 1. 修订后阶段总览（替换原 P0–P5）

| 阶段 | 名称 | 修复项 | 是否碰 RAG 内核 |
|---|---|---|---|
| **P0'** | 质量地基（KB 匹配评测集 + 指标） | 缺口#3 前置 | 否 |
| **P1'** | 路由增强 + 选择性扇出 + 置信度 early-exit | **缺陷1、缺陷2** | 否（封装调用） |
| **P2'** | 阈值校准 + 在线学习闭环 | 缺口#3 | 否 |
| **P3'** | 多模态入口（图→qwen-vl 描述） | 缺口#5（部分） | 否（Embedder 不变） |
| **P4'** | 三新 Agent 骨架 + HITL 交接 | 缺口#1 | 否 |
| **P5'** | 三层记忆（working/episodic/semantic） | **缺陷3** | 否 |
| **P6'** | 可观测性 + 成本护栏 + Agent 契约 | 缺口#2、#4 | 否 |

> 关键原则：**P0'–P6' 全部不重写现有 RAG 内核**，只在它之外加编排层与新模块。风险最低、可灰度。

---

## 2. 逐阶段明细

### P0' 质量地基（修订：原 P0 只说"建评测集"，现明确生成方法与基线）
- **新增 `scripts/gen_kb_eval.py`**
  - 读 `lancedb_data` 的 `rag_policy` / `rag_service` / `rag_tech` 表 chunk；
  - 用 qwen-plus 对每批 chunk 生成 `(question, reference_answer, expected_chunk_ids)`；
  - 按 KB **分层抽样**（每库 ≥20 题），写 `tests/eval_dataset_kb.json`。
- **扩展 `scripts/evaluate.py`**
  - 复用 `_llm_ragas` 算 RAGAS 4 项；
  - 新增 `faithfulness`（小模型判答案是否被来源支持）、`recall@k`（expected_chunk_ids 命中率）、`hallucination_rate`；
  - 输出 `data/eval-summary.json`（逐题明细 + 校准基线）。
- **验收**：跑出非 0 的真实质量分（用自身场景，杜绝之前"通用 QA 集→全 0"的坑）。

### P1' 路由增强 + 选择性扇出 + 置信度 early-exit（最高 ROI）
- **`engines/router/intent_router.py`**
  - `RoutingResult`（:32）加字段 `intent: str`（consult/operate/complaint/chitchat）、`ambiguous: bool`；
  - `route()`（:43）在规则/LLM 判定后映射到 intent 维度；当 top1–top2 置信差 < `AMBIGUITY_MARGIN` 或规则/LLM 冲突 → `ambiguous=True`。
- **`engines/router/routing_rules.py`**
  - 加 `INTENT_MAP: dict[skill, intent]` + `AMBIGUITY_MARGIN` 常量。
- **新增 `engines/orchestration/orchestrator.py`**
  - `Orchestrator.select_agent(intent, ambiguous) -> list[Agent]`：**默认 `[ConsultAgent]`**，complaint 恒附加 `SentimentSidecar` 旁路，ambiguous→先反问澄清（**修复缺陷1：选择性扇出，不跑满 4 个**）；
  - `Orchestrator.run(query, history)`：路由 → 扇出 → 聚合 → 交置信度 → 输出 `(answer, sources, confidence, action)`；
  - `ConfidenceEvaluator.evaluate(...)`：**early-exit 分阶**（路由层先判置信 → 生成后再判），返回 `tier` + 显式终态 `abstain` / `handoff`（**修复缺陷2：闸门前移 + 缺拒答终态**）。
- **复用**：`api/core/orchestrator.py` 的 `ask`(238)/`ask_stream`(919)/`_build_context`(614) 作为 `ConsultAgent` 内核被调用，**不修改**。

### P2' 阈值校准 + 在线学习闭环（新增，缺口#3）
- **新增 `engines/evaluation/calibration.py`**
  - 用 P0' 基线画 `置信度 → 实际正确率` 校准曲线，切分点写入 `api/config.py` 的 `Settings`（:6）字段，**不硬编码**。
- **新增 `scripts/collect_failures.py`**（接 `gen_kb_eval.py`）
  - 低分 / 转人工样本自动回流 `tests/eval_dataset_kb.json` → 触发重校准闭环（**修复缺口#3：系统越用越好**）。

### P3' 多模态入口（修订，缺口#5 部分）
- **`api/core/orchestrator.py`** 的 `ask`(238)/`ask_stream`(919) 入参加 `image: str | None`（base64/url）；
- **新增 `engines/parsing/vision.py`** `describe_image()` 调 qwen-vl 生成文字描述，与 `ocr.py` 并列成"图→文"通道；文本+描述拼接进 query；**Embedder 不变**；
- **`api/routers/qa.py`** 的 `/api/v1/qa/ask` 加 `image` 字段；前端 `web/index.html`+`web/common.js` chat 区加图片上传（现有 `fileInput` 是灌库用，不混用）；
- **标注 limitation**：图描述丢视觉真值，严重场景（残次商品照/报错码截图）后续上图搜/visual grounding（**修复缺口#5 的已知上限**）。

### P4' 三新 Agent 骨架 + HITL 交接（修订，缺口#1）
- **新增 `engines/agents/complaint_agent.py`**：小模型情绪识别（愤怒/正常）→ 升级工单（接口预留）→ 返回情绪标签；作为 P1' 的 `SentimentSidecar` 常驻旁路；
- **新增 `engines/agents/chitchat_agent.py`**：小模型轻量回复（不耗 qwen-plus）；
- **新增 `engines/agents/operation_agent.py`**：意图+槽位抽取（小模型）→ `engines/tools/registry.py`（声明式 `name/scope/危险等级/需人工确认`）；无外部系统时走 mock + 降级转人工（按你确认的"暂无外部系统"）；
- **新增 `engines/agents/handoff.py`**：**HITL 交接协议** —— `pack_context(session)`（上下文打包）+ `resume(session_id)`（续接状态）+ 队列/SLA 占位（**修复缺口#1**）。

### P5' 三层记忆（修订，缺陷#3）
- **新增 `engines/memory/working.py`**（当轮上下文窗口，迁移 `api/core/orchestrator.py` 现有 history 用法）；
- **新增 `engines/memory/episodic.py`**（会话时序，LanceDB 新表）；
- **新增 `engines/memory/semantic.py`**（用户画像结构化 KV / 可选向量）；
- **新增 `engines/privacy/pii.py`** `mask_pii()`：长期向量入库前脱敏（**修复缺陷3：三层架构，不纯向量**）。

### P6' 可观测性 + 成本护栏 + Agent 契约（修订，缺口#2/#4）
- **新增 `api/core/observability.py`**
  - TraceID（扩展现有请求 ID 为 trace）串联各 span；
  - 基础指标：路由分布 / 延迟 p50–p95 / Agent 成功率 / 成本 token；
  - **`CostGuard`**：per-turn token 预算 + 熔断（**修复缺口#2**）；
  - **`AgentContract`**：统一输入输出 schema（上下文+意图+记忆 → 回答+置信度+溯源+动作）（**修复缺口#4**）；
  - A-B flag（`api/config.py` `Settings` 字段）。
- **新增 `api/routers/qa.py` 端点**：`/api/v1/handoff`（转人工上下文包）、`/api/v1/feedback`（失败样本回流 P2'）。

---

## 3. 配置默认值（`api/config.py` 的 `Settings` 新增字段）

```toml
rerank_mode          = "rrf"     # 语料实证 bge-reranker-base 伤排序, 默认 rrf, 保留开关待 A/B
retrieve_candidate_k = 20
rerank_top_k         = 10
context_parent_k     = 5
vision_model         = "qwen-vl-max"   # 仅用于图描述, Embedder 仍是 bge-small-zh
conf_high            = 0.75            # 高置信直回 (P2' 校准后覆盖)
conf_mid             = 0.45            # 中置信附来源
ambiguous_margin     = 0.12            # 路由歧义阈值
pii_mask             = true
audit_log            = true
cost_token_budget    = 4000            # per-turn token 预算, 超预算熔断 (缺口#2)
stream_first_token_s = 1.5
e2e_p95_s            = 3.0
parallel_agents      = false           # 4 Agent 不并行, 仅咨询默认路径
```

---

## 4. 风险与回滚
- 所有新增模块独立目录（`engines/orchestration`、`engines/agents`、`engines/memory`、`engines/tools`、`engines/privacy`、`api/core/observability.py`），不动现有 RAG 内核 → 单模块出错可独立回滚；
- `IntentRouter` 只在 `RoutingResult` 加可选字段 + `route()` 末尾映射，向后兼容；
- 多模态为追加 `image` 入参（默认 None），不影响现有纯文本链路；
- 配置全部走 `Settings` 环境变量，快速失败，无硬编码密钥。

---

## 5. 落地建议
**从 P0' 开工**（已踩过"无自身场景评测→全 0"的坑，P0' 是后面一切的前提且零风险）。
之后 P1' 是最高 ROI 且仍不碰 RAG 内核的增强，可紧接着做。
P3'–P6' 按你优先级灰度推进。
