# RAG 2.0 术语表

**版本**: v1.0  
**日期**: 2026-08-04

---

## 核心概念

| 中文术语 | 英文/缩写 | 定义 | 项目中指代 |
|---------|----------|------|-----------|
| 检索增强生成 | RAG (Retrieval-Augmented Generation) | 从外部知识库检索相关文档后交由 LLM 生成答案的范式 | 项目整体方法论 |
| 统一中间表示 | UIR (Unified Intermediate Representation) | 文档解析后的标准化数据结构，屏蔽不同解析器的输出差异 | `engines/parsing/__init__.py` 中的 `UIRDocument` |
| 语义分块 | Semantic Chunking | 基于句子嵌入相似度低谷动态切分文本，而非固定字数切分 | `engines/chunking/semantic_chunker.py` |
| 向量嵌入 | Embedding / Vector | 文本的数值化表示（512 维浮点数组），用于语义相似度计算 | BGE-small-zh 输出的 512d vector |
| 混合检索 | Hybrid Retrieval | 同时使用稠密向量检索和图谱检索，合并去重后精排 | `engines/retrieval/hybrid_retriever.py` |
| 重排序 | Reranking | 对初步检索结果用更强模型重新排序，提升精度 | `engines/retrieval/reranker.py` |
| 自检索 | Self-Retrieval | 评估检索质量 → 自动改写查询 → 重检索的循环过程 | `engines/retrieval/self_retrieval.py` |
| 知识图谱 | Knowledge Graph | 实体 + 关系构成的有向图，支持多跳推理 | `engines/graph_rag/` |

---

## 模型与算法

| 术语 | 说明 |
|------|------|
| **BGE** (BAAI General Embedding) | 智源研究院 BAAI 开发的中文嵌入模型系列，C-MTEB 排名领先 |
| **BGE-small-zh-v1.5** | 项目中使用的嵌入模型，512 维，~100MB，CPU 推理 ~20ms |
| **Cross-Encoder** | 同时输入 (query, doc) 对计算相关度的模型，精度高于 Bi-Encoder 余弦相似度 |
| **bge-reranker-base** | 项目中使用的重排序 Cross-Encoder 模型 |
| **C-MTEB** (Chinese Massive Text Embedding Benchmark) | 中文嵌入模型评测基准 |
| **IVF** (Inverted File Index) | 向量近似最近邻索引算法，聚类后只搜索最近几个聚类中心 |
| **HNSW** (Hierarchical Navigable Small World) | 图结构向量索引，检索精度和速度均优于 IVF，但内存占用更高 |
| **MVCC** (Multi-Version Concurrency Control) | LanceDB 使用的并发控制机制，读写不互锁 |
| **MRR** (Mean Reciprocal Rank) | 检索评价指标，第一个相关文档排名倒数的均值 |

---

## 技术组件

| 术语 | 说明 | 角色 |
|------|------|------|
| **FastAPI** | Python 异步 Web 框架 | API 服务 |
| **Uvicorn** | ASGI 服务器 | 运行 FastAPI 应用 |
| **Pydantic** | 数据校验库 | 配置管理 + 请求模型 |
| **LanceDB** | 嵌入式向量数据库 | 开发环境向量存储 |
| **Milvus** | 分布式向量数据库 | 生产环境向量存储 |
| **Neo4j** | 图数据库 | 生产环境知识图谱持久化 |
| **Redis** | 内存 KV 数据库 | 限流 + 缓存 + Celery Broker |
| **Celery** | Python 分布式任务队列 | 异步文档处理 |
| **MinIO** | S3 兼容对象存储 | Docker 环境文件存储 |
| **PyMuPDF** (fitz) | C 扩展 PDF 解析库 | 原生 PDF 文本提取 |
| **PDFPlumber** | Python PDF 表格提取库 | 表格坐标级提取 |
| **PaddleOCR** | 百度 OCR 识别引擎 | 扫描件文字识别 |
| **Sentence-Transformers** | 句子嵌入框架 | BGE 模型加载与推理 |
| **httpx** | Python HTTP 客户端 | LLM API 调用 |
| **TokenHub** | 内部 LLM API 网关 | DeepSeek-V4-Flash 等模型接入 |
| **DeepSeek-V4-Flash** | MoE 大语言模型 | RAG 答案生成 |
| **pgvector** | PostgreSQL 向量扩展 | 生产环境备选向量存储 |

---

## 项目特有术语

| 术语 | 说明 |
|------|------|
| **Phase 1** | 基础管线：解析 + 分块 + 嵌入 |
| **Phase 2** | 检索增强：图谱构建 + 混合检索 + LLM 生成 |
| **Phase 3** | 体验提升：Web 前端 + Reranker + 安全基础 |
| **Phase 4** (待推进) | 生产加固：Neo4j + Milvus + Celery + 限流 |
| **Phase 5** (待推进) | 质量与监控：完整测试 + Prometheus + 压测 |
| **Phase 6** (待推进) | 前端重构 + 批量处理 + 文档删除 |

| 术语 | 说明 |
|------|------|
| **文档处理流水线** | PDF → 解析 → 分块 → 嵌入 → 存储 → 图谱构建 |
| **检索问答流水线** | 用户问题 → 嵌入 → 检索(向量+图谱) → 重排 → [Self-Retrieval] → LLM 生成 |
| **降级链** | 外部依赖不可用时的 fallback 策略层次，每级至少一个备选方案 |
| **TECH_PATTERNS** | 预定义的常见技术实体正则表达式集合，用于零成本规则抽取 |
| **heading_chain** | chunk 继承的标题层级路径 (如 "第一章 > 1.1 概述 > 1.1.1 背景") |
