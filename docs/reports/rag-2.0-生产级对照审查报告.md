# RAG 2.0 生产级对照审查报告

> 对照标尺：**「AI知识库别乱搭！RAG向量库实战讲透」视频要点**（文档预处理 / 切分策略 / Embedding与向量库 / 检索链路 / 评估体系 / 工程化）
> 审查方法：逐条对照项目真实代码（标注 `文件:行号`），不基于假设。
> 判定符号：✅ 符合 ｜ ⚠️ 部分符合 ｜ ❌ 不符合

---

## 一、文档预处理

| 视频要求 | 现状 | 判定 | 证据 |
|---|---|---|---|
| 多格式统一转 MD/JSON：PDF/Word/HTML/Markdown/PPT/Excel | 支持 PDF/MD/TXT/DOCX；**缺 HTML/PPT/Excel** | ❌ | `engines/parsing/registry.py:8` 仅注册 4 种 |
| 自动去噪：页眉/页脚/页码 | 按 bbox 位置识别 header/footer（页码通常在 footer 内） | ✅ | `pdf_parser.py:151-154`（`bbox[1]<50`→header，>`page_height-50`→footer） |
| 自动去噪：水印 | 未实现 | ❌ | 无 watermark 检测逻辑 |
| 自动去噪：重复声明 | 未实现 | ❌ | — |
| 保留标题层级、章节编号 | 保留 heading_level + 编号正则识别 | ✅ | `structure_chunker.py:151-190`（_heading_chains / _estimate_level / 编号正则） |
| 保留表格行列结构 | PDF 用 pdfplumber 提取，保留表头 | ✅ | `pdf_parser.py:108-123`；`strategies.py:146-177`（_TableAware 表格独立成块） |
| 保留代码块（含语言标识） | 代码围栏内容被当段落保留，但**丢失 ```` ``` ```` 围栏与语言标识**，且会被递归切分腰斩 | ⚠️ | `markdown_parser.py:48-51`（fence 内按 paragraph）；`structure_chunker.py:142-149`（仅保留 paragraph/title） |
| 每条文本携带 6 项元数据：doc_id/file_name/page/section/chunk_id/update_time | 有 doc_id ✅、chunk_id ✅、page(区间) ✅、section(title_chain) ✅、update_time(created_at) ✅；**缺 file_name（仅 doc_title）** | ⚠️ | `vector_store.py:35-78`（列：doc_id/title_chain/doc_title/page_range/created_at，无 file_name） |

**小结**：PDF 解析相当成熟（双引擎+OCR+去噪+表格）。最大缺口是 **HTML/PPT/Excel 三种格式完全无解析器**，以及 **file_name 没落到 chunk 级元数据**。

---

## 二、切分策略

| 视频要求 | 现状 | 判定 | 证据 |
|---|---|---|---|
| 语义优先、长度兜底 | StructureChunker：标题边界 + 递归字符回退 | ✅ | `structure_chunker.py:94-121` |
| 普通文档 300–800 字、重叠 10%–20% | 默认 800 字 / 重叠 128（16%） | ✅ | `config.py:102-103`；`structure_chunker.py:19` |
| FAQ 以整条问答为 chunk 不拆分 | FaqChunker 整对成块 | ✅ | `strategies.py:22-59` |
| 合同按条款切分并保留条款编号 | ClauseChunker 按「第X条/1.1/（一）」切，保留编号 | ✅ | `strategies.py:62-143`（_CLAUSE_SPLIT 覆盖 第一条/第一章/1.1/一、/（一）） |
| 代码按函数/类/模块整块保留 | **无**代码专用切分 | ❌ | 无 CodeChunker；代码走通用递归切分 |
| 表格保留表头 + 若干行 | _TableAware 表格独立成块 | ✅ | `strategies.py:146-177` |
| 父子文档机制（检索 child、返回 parent 上下文） | **无**；仅用 title_chain 内联上下文近似 | ❌ | 无 parent/child 结构 |

**注意**：ClauseChunker 默认 `max_chars=1200`（`strategies.py:81`），超过 800 字「普通文档」上限——属合同条款的合理例外，但值得在配置里标注。

**小结**：切分策略是项目强项（semantic/faq/clause/table 四态齐全）。缺口是 **代码整块切分** 与 **形式化父子文档机制**。

---

## 三、Embedding 与向量库

| 视频要求 | 现状 | 判定 | 证据 |
|---|---|---|---|
| 中文场景优先评测 BGE/GTE/E5，以业务数据实测 | 用 bge-small-zh-v1.5（属 BGE 系，视频推荐）；**仅 1 个模型，无 GTE/E5 对照评测 harness** | ⚠️ | `config.py:25`；embedder 无多模型评测入口 |
| 支持商用 API 与本地部署两种模式 | **仅本地 SentenceTransformer**；无商用 embedding API 路径 | ❌ | `embedder.py:47-57`（仅 `_get_model` 本地加载）；config 无 embed API 配置 |
| 向量库按规模选型：POC Chroma/FAISS，生产 Milvus/Qdrant/Weaviate | 用 **LanceDB 嵌入式**（介于 POC 与生产之间） | ⚠️ | `vector_store.py:1,14`（lancedb.connect） |
| 生产环境需 ANN 索引 | **未显式建 ANN 索引**（默认暴力/flat 检索） | ❌ | `vector_store.py:100-105`（search 无 create_index） |
| 元数据标量索引 | 支持 where 过滤，但**未建标量索引** | ⚠️ | `vector_store.py:104` |
| 高可用副本 | 单节点嵌入式，无副本 | ❌ | — |
| 延迟/召回/错误率监控 | Prometheus 全埋点 | ✅ | `api/core/metrics.py`（latency/recall/degradation/error） |

**小结**：Embedding 模型选型正确（BGE 系），且查询有 TTL 缓存（`embedder.py:85-111`）。主要短板是 **缺商用 API 双模 + 模型对照评测**，以及 **向量库停留在 POC 级（无 ANN 索引/HA）**。

---

## 四、检索链路

| 视频要求 | 现状 | 判定 | 证据 |
|---|---|---|---|
| Query 清洗与改写 | 规则清洗（去敬语/同义扩展）+ LLM 改写；**默认 `query_rewrite_enabled=False`** | ⚠️ | `query_preprocessor.py`（normalize/expand）；`config.py:127` |
| 多路召回：向量 + BM25 + 元数据过滤 | 向量 + BM25 并行召回 + 图谱增强；按 KB 检索层过滤 | ✅ | `orchestrator.py:532-533`（_parallel_retrieve）；`orchestrator.py:520-521`（per-kb bm25） |
| RRF 分数融合 | RRF + 插值融合（默认 interp，数据驱动选定） | ✅ | `fusion.py:17-97`；`config.py:114`（fusion_method="interp"） |
| Rerank 精排（Top 3–8 进 LLM） | 已实现 bge-reranker-base + 候选 10；**默认 `rerank_enabled=False`**（本语料 CPU 上关比开 Recall@3 更高） | ⚠️ | `orchestrator.py:557-558`；`config.py:81`（注释给出诊断数据：开 71.4% / 关 74.0%） |
| 上下文组装 | ✅ | `orchestrator.py` 检索→重排→组装 | ✅ |
| 生成时强制返回 doc_id/chunk_id 引用 | `[来源N]` 标签 + chunk_id/doc_id 透传 | ✅ | `orchestrator.py:643`（[来源i+1]）；sources 含 chunk_id |
| 低于相关性阈值兜底「未找到可靠信息」 | 置信度<0.50 直接拒答，文案「知识库中未找到与您问题高度相关的内容」 | ✅ | `orchestrator.py:784,1138`；`config.py:152`（answer_confidence_floor=0.50） |

**小结**：检索链路是**最贴合视频要求**的部分——多路召回 + RRF/interp 融合 + 跨库兜底 + 阈值拒答 + 强制引用，链路完整。两点「默认关闭」（改写、重排）是**基于本语料实测的数据驱动决策**，不是缺失，上 GPU/语义型语料后应开启。

---

## 五、评估体系

| 视频要求 | 现状 | 判定 | 证据 |
|---|---|---|---|
| Recall@K | ✅（按 expected_chunk_ids 精确匹配） | ✅ | `evaluate.py:144-148` |
| MRR | **未直接计算**（context_precision 近似） | ⚠️ | `evaluate.py` 无 MRR |
| 忠实度（faithfulness） | ✅ RAGAS faithfulness + 运行时 fidelity_guardrail(0.60) | ✅ | `evaluate.py:41-55`；`orchestrator.py:820-827`（fidelity_threshold） |
| 引用准确率 | ✅ chunk_id 级 recall/precision | ✅ | `evaluate.py:145-148` |
| 延迟/成本指标 | ✅ qa_metrics(latency/tokens) + Prometheus | ✅ | `api/core/qa_metrics.py`；`metrics.py` |
| 固定评测集 100+ 条真实业务问答 | ✅ **390 条**（eval_clean）+ 75 条（tests） | ✅ | `data/eval_clean/eval_dataset.json`（390 条） |
| A/B 对比 | 可对不同 URL/配置跑 evaluate.py 对比，但**无内置 A/B 框架** | ⚠️ | `scripts/evaluate.py` 单跑；对比需手动 |
| 知识库更新后自动回归 | 有 evaluate.py / export_qa_logs.py，但**无 CI 自动触发** | ⚠️ | 无 hook 自动回归 |

**小结**：评估体系扎实（RAGAS 4 指标 + chunk 级引用准确率 + 390 条固定集）。短板是 **MRR 缺位、A/B 与更新后回归未自动化**。

---

## 六、工程化

| 视频要求 | 现状 | 判定 | 证据 |
|---|---|---|---|
| 增量更新（文档哈希变更才重建） | **无内容哈希增量**；populate_kb 全量重切（仅按 chunk_id 去重重建） | ❌ | `populate_kb.py` 无 hash 比对；`vector_store.py:83-98`（insert 全量删旧加新） |
| 多租户权限隔离（检索层过滤，非前端隐藏） | KB 级 RBAC：allowed_kbs 作用域 + 缓存键分片 + 文档库 kb_in 过滤 | ✅ | `orchestrator.py:149-159,263`（_candidate_kbs）；`qa_cache.py:32-34`（键含 allowed_kbs）；`document_store.py:173-190`（kb_in 过滤） |
| 完整链路日志（含失败与超时） | ✅ logging 中间件 + qa_logs + 结构化日志 | ✅ | `api/middleware/logging_middleware.py`；`document_store.py:130-163`（log_qa） |
| 高频 Query 缓存 | ✅ RBAC 分片 + TTL + 内存/Redis 可插拔 | ✅ | `qa_cache.py`；`config.py:161-162` |
| 重复内容去重 | 入库按 chunk_id 去重 + 检索层按 id 去重；**语义近重复去重生产默认关闭**（P0-1 结论：重排前去重伤 Recall） | ⚠️ | `vector_store.py:83`（dedup）；`hybrid_retriever.py:69-77`；`dedup.py:53-89`（redundancy_reduce 未接生产） |

**小结**：工程化基础好（RBAC 检索层隔离、缓存、日志都到位）。最明确的缺口是 **增量更新靠哈希跳过未变文档**。

---

## 优先级优化清单（影响大且易实施优先）

> 评分维度：影响（业务/质量收益）× 实施难度（低/中/高）。

### P0 — 高影响 / 低-中难度（建议立即做）

**1. 补全缺失格式解析器（HTML / PPT / Excel）**
- 原因：视频明确要求 6 种格式，当前缺 3 种，业务文档接入面不全。
- 影响：可接入文档类型 +60%，避免「上传就失败」。
- 方向：新增 `html_parser.py`(html2text/bs4)、`pptx_parser.py`(python-pptx)、`excel_parser.py`(openpyxl)，注册到 `registry.py`。
- 难度：**中** ｜ 收益：**高**

**2. chunk 级补 file_name / update_time 元数据**
- 原因：视频要求 6 项元数据，当前缺 file_name，update_time 仅 created_at 近似。
- 影响：溯源、审计、前端展示来源文件名。
- 方向：解析层把 `source.path` 文件名写入 chunk.metadata.file_name；向量库加 `update_time` 列（复用 created_at 或文档 updated_at）。
- 难度：**低** ｜ 收益：**中**

**3. 默认开启查询改写 / 自检索（或按质量敏感场景默认开）**
- 原因：`query_rewrite_enabled=False` / `enable_self_retrieval=False`，而诊断显示改写是突破 fusion/rerank 调参天花板的关键杠杆（context_precision≈0.318 瓶颈）。
- 影响：检索排序质量上限。
- 方向：评估延迟/成本后，将默认改 True，或对 `QARequest.enable_self_retrieval=true` 的场景默认启用。
- 难度：**低** ｜ 收益：**中-高**

**4. 增量更新（内容哈希跳过未变文档）**
- 原因：populate_kb 每次全量重切，大库更新成本高、易误伤未变文档。
- 影响：更新耗时、一致性。
- 方向：解析前算 `sha256(content)`，与 `documents.db` 已存 hash 比对，未变则跳过；或基于 mtime+size 快速判定。
- 难度：**低-中** ｜ 收益：**中-高**

### P1 — 高影响 / 中难度

**5. 代码块专用切分（函数/类/模块整块 + 语言标识）**
- 原因：代码被当段落切，可能腰斩函数，丢失语言标识。
- 影响：代码类文档答不准、引用错位。
- 方向：`markdown_parser.py` 将 fence 标为 `type="code"` 并带 `lang`；新增 `CodeChunker` 按函数/类边界整块（AST 或正则）。
- 难度：**中** ｜ 收益：**中-高**

**6. 形式化父子文档机制**
- 原因：当前只用 title_chain 内联上下文近似，无「小 child 检索 + 大 parent 喂 LLM」。
- 影响：长文档召回率 + 上下文完整性。
- 方向：建 parent chunk（大段）+ child chunk（小段，带 parent_id）；检索 child，组装时回拉 parent 全文。
- 难度：**中** ｜ 收益：**中-高**

**7. Embedding 商用 API 双模 + 模型对照评测**
- 原因：仅本地模型，无商用 API 弹性；BGE/GTE/E5 未在业务数据上实测对比。
- 影响：中文效果上限 + 部署弹性。
- 方向：embedder 增加 API backend（DashScope/OpenAI embedding）；跑 `eval_retrieval.py` 在 390 条集上对比 BGE/GTE/E5 的 Recall@K。
- 难度：**中** ｜ 收益：**中-高**

**8. 去噪增强（水印 / 重复声明）**
- 原因：当前仅页眉页脚。
- 影响：噪声召回、答案污染。
- 方向：基于文本指纹/版面特征识别水印与重复声明段，解析期剔除。
- 难度：**低-中** ｜ 收益：**中**

### P2 — 生产加固 / 高难度

**9. 向量库 ANN 索引 + 标量索引 + 生产选型**
- 原因：LanceDB 未建 ANN 索引、单节点无 HA，规模上来检索延迟/可用性受限。
- 影响：规模、延迟、可用性。
- 方向：为 LanceDB 建 IVF/HNSW ANN 索引 + 标量索引；或迁移 Milvus/Qdrant（视规模）。
- 难度：**高** ｜ 收益：**高（规模化时）**

**10. 评估补全 MRR + A/B 框架 + 更新后自动回归 CI**
- 原因：MRR 缺位、A/B 与回归靠手动。
- 影响：可观测性、防回归。
- 方向：脚本加 MRR；封装 A/B 对比（同问题双配置跑分）；KB 更新 hook 触发 `evaluate.py`。
- 难度：**中-高** ｜ 收益：**中**

**11. Rerank 默认策略定夺**
- 原因：CPU 上 bge-reranker-base 伤排序，故默认关；但语义型/上 GPU 后应开。
- 影响：精排质量上限。
- 方向：上 GPU 后切 `bge-reranker-v2-m3` 并设 `rerank_enabled=True`；保留 adaptive_alpha 抗近重复。
- 难度：**低（配置）/ 高（需 GPU）** ｜ 收益：**中**

---

## 已做得出色、建议保持的部分

- **检索链路**：向量+BM25+图谱多路召回、RRF/interp 融合、跨库兜底、置信度阈值拒答、强制 [来源N] 引用——完整且数据驱动。
- **切分四态**：semantic/faq/clause/table-aware 类型化工作流，贴合「按文档类型选策略」。
- **RBAC 检索层隔离**：allowed_kbs 作用域贯穿路由/缓存/文档库，符合「检索层过滤而非前端隐藏」。
- **评估基础**：RAGAS 4 指标 + chunk 级引用准确率 + 390 条固定业务集。
- **工程化**：RBAC 缓存分片、全链路日志、QA 缓存、Prometheus 监控均到位。

> 结论：项目在「检索链路 / 切分 / RBAC / 评估基础 / 工程化」上已接近视频的生产级标准；最需补齐的是 **3 种格式解析器、增量更新哈希、chunk 元数据完整化、代码整块切分、向量库 ANN/HA 与 Embedding 双模**。
