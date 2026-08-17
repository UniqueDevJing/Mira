# 设计决策文档 — RAG 2.0

每个技术选型回答三个问题：**为什么选它？为什么不用替代方案？代价是什么？**

---

## 1. Embedding 模型: BAAI/bge-small-zh-v1.5 (512d)

### 为什么选它？

- **国产中文优化**: BGE 系列在 C-MTEB 中文基准上排名前列，语义区分度好
- **维度适中**: 512 维在精度和性能间取得平衡。768/1024 维模型检索精度提升 <3%，但内存和计算开销翻倍
- **部署友好**: small 版本 ~100MB，CPU 推理延迟 ~20ms/条，单机可承载
- **与 Reranker 同系列**: BGE Embedding + BGE Reranker 搭配，上游表示空间一致

### 为什么不用这些？

| 替代方案 | 排除原因 |
|---------|---------|
| text2vec-large-chinese (1024d) | 精度收益 <3%，内存翻倍，LanceDB 索引体积翻倍 |
| m3e-base (768d) | C-MTEB 排名低于 BGE，社区活跃度下降 |
| text-embedding-3-small (OpenAI) | API 延迟 + 成本不可控，离线不可用 |
| BGE-m3 (1024d, 多语言) | 项目只需中文，多语言能力冗余，维度翻倍 |

### 代价

- 512 维对细粒度语义区分可能不足（长尾查询召回率偏低）
- 缓解: Cross-Encoder 精排 + Self-Retrieval 多轮改写

### 参数依据

- `normalize_embeddings=True`: 余弦相似度需要归一化向量
- `passage:` / `query:` 前缀: BGE 模型训练时使用的前缀格式，不加会导致精度下降 ~5%

---

## 2. 分块策略: 标题树硬切 + 递归字符回退（含中文分隔符）

### 为什么选它？

- **文档结构感知**: 大多数 PDF 有标题层级，利用标题树保持语义边界可以避免在段落中间切分
- **自适应阈值**: 不用固定 chunk_size。相似度低于 `mean - 0.5*std` 时切分，适应不同文档密度
- **鲁棒性强**: 对排版变化（单栏/双栏/无标题）不敏感，退化为纯语义切分

### 为什么不用这些？

| 替代方案 | 排除原因 |
|---------|---------|
| RecursiveCharacterTextSplitter (LangChain) | 纯字符长度切分，无视语义边界，经常把一句话切成两半 |
| Agentic chunking (LLM 逐段判断切分点) | 每个 chunk 调一次 LLM，100 页 PDF 需要 ~50 次 LLM 调用 |
| 固定大小 sliding window | 无视文档结构，重叠区浪费 token |
| Semantic splitter (仅相似度) | 丢失文档结构信息，无标题链上下文 |

### 代价

- 依赖 PDF 解析器准确识别标题（字体大小启发式可能误判）
- 缓解: `_estimate_heading_level` 用字体大小推断层级，后续可增加 MLA/APA 等标准格式检测

### 参数依据

> **2026-08-16 更新**: 原"语义相似度切分"参数表 (min_tokens/similarity_threshold) 已随
> 2026-08-10 结构切分变更废弃, 下方为当前结构切分参数 (`api/config.py`, `RAG_` 前缀 env 可调)。

| 参数 | 值 | 理由 |
|------|----|------|
| chunk_max_chars | 800 | 结构切分单 chunk 最大字符；BGE-small-zh 512 token 上下文内，检索精准 |
| chunk_overlap | 128 | 递归字符切重叠量，防关键信息被切到两 chunk 边缘 |

### 2026-08-10 变更: PDF 分块从语义切分改为结构切分

原方案「语义相似度 + 标题树动态切分」废除，理由:
1. 语义边界检测每段落一次 embedding 调用，约为存储 embedding 的 2x 额外成本
2. 标题树（font_size/加粗/编号）纯规则零成本，已保留主要结构信息
3. 递归字符切用中文分隔符（。！？；，）断句，质量与语义切差距小

替代排除: RecursiveCharacterTextSplitter 单独使用（无标题上下文，弃）、全格式语义切（TXT 无结构白烧 embedding，弃）。

---

## 3. 向量存储: LanceDB (开发) / Milvus (生产)

### 为什么选 LanceDB？

- **零运维**: 嵌入式数据库，不需要单独部署服务，开发体验接近 SQLite
- **高并发友好**: 无锁读取（多版本并发控制），多进程读不冲突
- **与 PyArrow 原生集成**: 零拷贝数据转换，写入性能好
- **支持过滤**: `where()` 表达式支持 doc_id 等字段过滤

### 为什么生产切 Milvus？

- LanceDB 在百万级向量时检索延迟线性增长，Milvus 有 IVF/HNSW 索引
- Milvus 原生支持分布式、多副本、监控
- Milvus 有 GPU 索引加速

### 为什么不用这些？

| 替代方案 | 排除原因 |
|---------|---------|
| Chroma | 嵌入式模式在大数据量时性能下降明显 |
| Qdrant | 需要额外部署，本地开发体验不如 LanceDB |
| FAISS | 无持久化，重启丢失，无过滤表达式 |
| Pinecone/Weaviate Cloud | 云服务，数据不出境要求 |

---

## 4. 重排序: BAAI/bge-reranker-base (Cross-Encoder)

### 为什么选它？

- **真实语义交互**: Cross-Encoder 输入 (query, doc) 对，输出相关度分数，比 Embedding 余弦相似度准确
- **精度提升显著**: Cross-Encoder → Embedding 余弦，MRR 通常提升 5-15%
- **与 BGE Embedding 同系列**: 上游表示空间一致

### 为什么不用 bge-reranker-v2-m3？

- v2-m3 模型体积 ~2.2GB (base 版 ~1.1GB)
- 推理延迟翻倍 (~50ms vs ~25ms per pair)
- 精度差异在中文场景通常 <2%
- 后续可 A/B 测试验证，当前阶段 base 够用

### 降级策略

```
Cross-Encoder 可用 → 精排 (rerank_method=cross-encoder)
Cross-Encoder 不可用 → Embedding 余弦相似度 (rerank_method=embedding-cosine)
Embedding 也不可用 → 保持检索原始顺序
```

---

## 5. 知识图谱: 规则 + LLM 混合抽取

### 为什么混合？

- **规则优先**: 技术实体名称（Python、FastAPI、Milvus...）用正则匹配，零延迟、零成本
- **LLM 补充**: 非标准实体（人名、公司名、自定义术语）由 LLM 抽取
- **合并去重**: 规则和 LLM 结果合并，同一实体合并别名

### 为什么不全用 LLM？

- 每个 chunk 调一次 LLM 抽取实体 → 一份 50 页 PDF 需要 ~30 次 LLM 调用 → 成本高、延迟大
- 对于 "Python"、"FastAPI" 这类明显技术实体，规则匹配更可靠且零幻觉

### 为什么不全用规则？

- 无法覆盖非技术类实体（如法规名称 "欧盟 AI 法案"）
- 无法抽取实体间关系

### 关系类型设计

`uses, supplies, signs, references, contains, depends_on, employs, owns` — 这 8 种关系类型覆盖了企业文档和知识图谱的常见关联模式，同时约束 LLM 输出格式。

---

## 6. 查询改写: 策略驱动的自适应改写

### 为什么需要？

嵌入向量空间中对同一个问题的不同表述，检索结果可能差异很大。例如 "系统用了什么技术" vs "技术栈构成"，前者可能匹配到介绍性段落，后者匹配到技术选型文档。

### 策略选择逻辑

```
relevance < 0.5 → keyword_expand + synonym (检索方向偏了)
coverage < 0.5 → decompose + abstract_adjust (检索太窄)
其他           → keyword_expand + synonym (保守改写)
```

### 降级: LLM 改写 → 规则改写

LLM 不可用时用模板改写 (`"关于 {query} 的所有信息"`)，虽然简单但能改善部分召回问题。

---

## 7. Self-Retrieval 多轮检索

### 设计考量

- **最大轮数 = 3**: 超过 3 轮改写成边际效益递减，且延迟累积
- **每轮独立评估**: 评估不满足则改写重来，不依赖上一轮结果
- **结果合并去重**: 所有轮次的结果按 chunk_id 合并，保留 key 首次出现的结果

### 代价

- 最多 3x 检索延迟
- 缓解: 限制 `max_rounds=3`, 大部分查询 1 轮满足条件

---

## 8. LLM 客户端: 自定义 httpx (不依赖 LangChain)

### 为什么不用 LangChain LLM 抽象层？

- **抽象层过厚**: LangChain 的 LLMChain/BaseLLM 嵌套了多层 callback、prompt template、output parser，调试栈追踪时需要跳过几十层框架代码
- **版本不稳定**: 0.x 时代 API 频繁变更，升级成本高
- **按量付费模型需要精确控制**: 每次 API 调用的 token 消耗、重试逻辑、超时控制应当透明

### 自定义客户端的优势

- 重试逻辑完全可控: 4xx 不重试（除了 429），5xx 指数退避最多 3 次
- httpx 连接池复用: max_connections=20, max_keepalive=5
- 错误信息中文: 用户可见错误中文化，日志保留原始英文
- 支持 reasoning_content (DeepSeek 系列)

---

## 9. 工程化降级链总览

```
整个 RAG 流水线中每个外部依赖都至少有一级降级:

LLM API (生成)       → 返回检索结果原文 (不生成)
LLM API (改写)       → 规则改写模板
LLM API (实体抽取)   → 规则抽取
Cross-Encoder 重排   → Embedding 余弦重排
Embedding 模型       → 返回空结果 + 报错
Redis (限流)         → 内存限流 (slowapi 默认降级)
jieba (分词)         → 返回默认值 0.5
Milvus (向量存储)    → 不支持自动降级，需手动切换 LanceDB
```

---

## 10. Router + Skill 多技能架构

### 为什么引入 Router？

单一知识库问答无法支撑异构知识场景：客服话术、技术文档、直接寒暄混在一起，检索相互污染，纯闲聊也白白消耗检索+LLM 成本。按意图分诊后，各 Skill 只用自己知识库，可独立演进、独立降级。

### 路由方式：规则优先 + LLM 兜底（混合）

```
规则计分 conf ≥ 0.85 → 直通 (source=rule, 50ms 内)
conf < 0.85          → LLM 分类 (1.5s 超时, 复用 LLMClient 熔断)
LLM 失败/超时/熔断    → 默认 tech 库 (source=fallback, 绝不中断)
```

**为什么不用纯规则 / 纯 LLM？**

| 方案 | 排除原因 |
|------|---------|
| 纯规则关键词 | 覆盖不了模糊表达（"这个怎么退"无规则词）；词表膨胀难维护 |
| 纯 LLM 分类 | 每次提问都花 200-500ms + token；LLM 挂时全站瘫痪 |
| 先 LLM 后规则 | 反了：高频寒暄/明显业务词根本不需要 LLM，先规则省大部分延迟 |

**规则 conf 公式**：关键词带权重（强信号 0.9 / 中 0.7 / 弱 0.5），取命中最高分。寒暄词与业务词同现时 direct 让位业务，避免"你好，退款怎么弄"误判成闲聊。

### 代价

- LLM 分类仅在规则未命中时触发，大部分查询命中规则 50ms 内直通，延迟增量可控
- 规则词表需按业务维护；当前为关键词级，不做语义网/正则

### 多知识库隔离：LanceDB 多表 + 按库单例

| 组件 | 隔离方式 |
|------|---------|
| 向量 | 每 kb 一张 LanceDB 表 (`rag_<kb>`)，默认 `documents` 兼容旧数据 |
| BM25 | 每 kb 独立内存索引 |
| 图谱 | 每 kb 独立 GraphStore（实体/关系抽取器共享） |
| 文档元数据 | SQLite `knowledge_base` 列 |

### 跨库兜底 (top1_score < 0.50)

当前库检索结果置信过低时，花 +0.6s 预算查兄弟库合并结果。解决"问题描述偏向 A 库但答案在 B 库"的召回缺口。

阈值 0.55 → 0.50 由 `scripts/calibrate_threshold.py` 数据驱动校准 (2026-08-15):
18 条标注 (生产真路由, LanceDB+BM25 重建) 扫 t∈[0.3,0.8], F1 打平 (0.500) 但 0.50 精度 1.0 vs 0.4, 消除 3 次无谓兜底 (正确路由但 top1∈[0.5,0.55] 被误触发, 白花延迟 + 污染风险)。阈值可 env `RAG_CROSS_KB_THRESHOLD` 覆盖。

### 分阶段超时与降级（对齐全局 12s 预算）

| 阶段 | 预算 | 降级 |
|------|------|------|
| Router 规则 | 50ms | — |
| Router LLM 分类 | 1.5s | → fallback tech |
| Embedding | 2 次重试 | → 跳过向量，仅 BM25 |
| 向量+BM25 并行 | 0.8s | 向量超时 → 仅 BM25 (L2) |
| 图谱检索 | 0.5s | 超时/失败 → 跳过 |
| 跨库兜底 | +0.6s | 超时跳过 |
| Rerank | 0.5s | 超时/失败 → 跳过 (L1) |
| LLM 生成 | 8s | 失败 → 检索摘要 (L3) |

> 预算由 5s/LLM 2s 放宽至 12s/8s (2026-08-10): DeepSeek-v4-flash 强制 reasoning, 单次生成实测 6.2s, 2s 预算必然 100% L3 降级。实测 20 并发 P99 7.7s (见 `performance-benchmark.md`), 原 "P99 < 3s" SLA 不可达需重定标。

每阶段预算 = min(阶段值, 全局 12s 剩余)，保证总延迟不超预算。

### 为什么不用 LangGraph / Agent 框架？

当前需求是"固定技能分发"，不是自由 Agent 编排。LangGraph 引入状态图、检查点等抽象，对这个确定性场景是过度设计。Router + Skill 注册表已够，后续要扩 SQL Skill 也只需注册表加一项。

---

## 11. OCR: RapidOCR (onnxruntime) 替代 PaddleOCR

**为什么换？**
- paddlepaddle 3.x 不支持 Python 3.14（venv 实际版本），且 2.x→3.x API 断裂（`use_angle_cls`→`use_textline_orientation`、`ocr()`→`predict()`）
- RapidOCR 纯 onnxruntime CPU 底座，Py 版本兼容好，pip 直装无重量级依赖
- 中文识别精度与 paddle 相当（实测验证: 中文扫描 PDF 两条文本 0.88/0.76 置信度精确识别）

**代价**: 新增 onnxruntime 依赖；RapidOCR 仅 CPU（原 `use_gpu` 参数忽略）。扫描件是边缘场景，性能非瓶颈。

## 12. 关系抽取: LLM 缺失时的规则兜底

**为什么？** LLM Key 缺失或熔断时，`RelationExtractor` 原返回空列表 → 图谱只有节点没有边，多跳推理失效。

**规则兜底策略** (`_rule_extract`):
1. 动词模式定向建边（使用/依赖/包含/引用/提供/拥有 → uses/depends_on/contains/...）
2. 无动词同句共现 → `related_to` 弱关系
3. 跨句不建边（共现限于同句，控制噪声）

**取舍**: 规则边是弱信号，精度低于 LLM 但保证连通性；LLM 可用时仍优先 LLM，规则只兜底。

## 13. 限流: slowapi 内存存储（可选启用）

**为什么 `RAG_RATE_LIMIT_ENABLED` 默认关？** 内存存储单进程有效，多 worker 生产需 Redis 存储；默认开会误触本地测试（同一 TestClient IP 高频请求）。启用后按 IP 限 `RAG_RATE_LIMIT_PER_MINUTE`（默认 60）作用于 QA 端点。

---

## 工程化待办清单

### P0 — 上线前必须完成

**1. 性能基准测试**

现状: 不知道单机 QPS 上限。需要:
- 用不同文档量 (100/1000/10000 chunks) 压测各接口
- 记录 P50/P99 延迟
- 确定资源瓶颈 (CPU/内存/磁盘IO)

**2. Token 成本追踪**

现状: 每个请求的 token 用量不可见。需要:
- `LLMClient.chat()` 返回 time 时同时记录 `usage.total_tokens`
- 日志增加 `tokens_in` / `tokens_out` 字段
- 按日/周/月统计成本

```python
# 日志中需要增加的字段
logger.info(
    "LLM 调用完成: model=%s, tokens_in=%d, tokens_out=%d, latency=%sms",
    self.model,
    usage.prompt_tokens,
    usage.completion_tokens,
    elapsed_ms,
)
```

### P1 — 上线前建议完成

**3. 压测脚本**

```bash
# locust 示例 (locustfile.py)
from locust import HttpUser, task
class RAGUser(HttpUser):
    @task
    def ask(self):
        self.client.post("/api/v1/qa/ask", json={
            "question": "系统使用了哪些技术？", "mode": "hybrid"
        })
```

**4. LLM 熔断**

```
连续失败 N=3 次 → 打开熔断器 → 30s 内直接返回降级答案
30s 后 → 半开 → 1 次试探请求成功则关闭，失败则继续打开
```

**5. QA 缓存**

```
Redis key: qa_cache:{md5(question + top_k)}
TTL: 1 小时
相同问题命中 → 直接返回缓存, 跳过检索+LLM
```

**6. 跨库兜底阈值校准 (PR 曲线)** ✅ 2026-08-15

`scripts/calibrate_threshold.py` 完成, 阈值 0.55→0.50 (18 条标注 F1 打平, 0.50 精度更优)。标注集偏小, 后续扩充后重跑。

### P2 — 稳定迭代

**7. Prometheus + Grafana 监控指标**

需要暴露的指标:
- `rag_qa_requests_total` — QA 请求计数
- `rag_qa_latency_seconds` — QA 延迟分布
- `rag_llm_tokens_total` — Token 消耗累计
- `rag_retrieval_rounds` — Self-Retrieval 轮数分布
- `rag_errors_total` — 错误计数 (按类型)

**8. GraphRAG 批量抽取**

现状: 逐 chunk 调 LLM，N chunks = N 次 API 调用。改为:
- 收集所有 chunk 的文本
- 合并为一个批量请求: "从以下 N 段文本中抽取所有实体和关系"
- 预期减少 80% API 调用次数

**9. 补充测试**

缺失的测试文件:
- `tests/test_qa.py` — QA 接口测试 (正常回答、空结果、非法输入)
- `tests/test_reranker.py` — 重排对比测试 (CE vs Embedding 精度差异)
- `tests/test_self_retrieval.py` — Self-Retrieval 多轮逻辑测试
- `tests/test_api.py` — API 集成测试 (限流、认证、CORS)

---

## 参考

- BGE 模型系列: https://huggingface.co/BAAI/bge-small-zh-v1.5
- C-MTEB 中文基准: https://github.com/FlagOpen/FlagEmbedding
- LanceDB 文档: https://lancedb.github.io/lancedb/
- Milvus 文档: https://milvus.io/docs
- RAGAS 评估框架: https://docs.ragas.io/
