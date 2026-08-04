# RAG 2.0 数据库设计

**版本**: v1.0  
**日期**: 2026-08-04

---

## 1. 数据架构全景

项目采用异构存储架构，不同数据类型使用最适合的存储引擎：

```
┌─────────────────────────────────────────────────┐
│                   RAG 2.0 数据层                  │
├─────────────┬─────────────┬──────────┬──────────┤
│  文档元数据  │  向量数据    │  图谱数据 │  缓存/会话 │
│  PostgreSQL │   LanceDB  │  Neo4j   │   Redis   │
│  (关系型)   │  (向量索引) │  (图)     │  (KV)     │
├─────────────┼─────────────┼──────────┼──────────┤
│ 开发替代:   │  开发/本地:  │ 开发替代:│ 开发替代: │
│ 内存 _docs  │  LanceDB    │  内存     │  slowapi   │
│ 字典        │  (嵌入式)   │  GraphStore│  内存模式  │
└─────────────┴─────────────┴──────────┴──────────┘
```

### 选型依据

| 存储引擎 | 用途 | 为什么选它 | 为什么不用单体DB |
|---------|------|-----------|----------------|
| PostgreSQL + pgvector | 文档元数据、用户、配置 | 关系型 ACID + 向量扩展，一库两用 | 单体 PG 的向量检索在大数据量下性能不如专用向量库 |
| LanceDB | 向量嵌入存储与检索 | 嵌入式、零运维、MVCC 并发无锁 | Chroma（性能差）、FAISS（无持久化） |
| Neo4j | 实体关系图谱 | 图遍历(O(1)边跳转)远比 SQL JOIN(O(N))高效 | 关系型做多跳图查询需自连接，SQL 可读性差且性能低 |
| Redis | 限流计数、QA 缓存、Celery Broker | 内存级延迟、支持多种数据结构 | Memcached（功能太少） |

---

## 2. 向量数据库 — LanceDB

### 2.1 Schema: `documents` 表

| 字段 | 类型 | 说明 | 来源 |
|------|------|------|------|
| `id` | string | chunk 唯一 ID (UUID) | SemanticChunker |
| `doc_id` | string | 所属文档 ID | PDFParser |
| `content` | string (max 65535) | chunk 文本内容 | SemanticChunker |
| `embedding` | list\<float32\> (512d) | BGE-small-zh 向量 | EmbeddingService |
| `created_at` | int64 | Unix 时间戳 | VectorStore.insert() |

### 2.2 索引策略

```
当前: 无额外索引（LanceDB 默认 IVF-PQ）
生产: 待评估 HNSW 索引参数（M=16, efConstruction=200）
```

### 2.3 分片与容量规划

| 规模 | 文档数 | 预估 Chunks | 预估存储 |
|------|--------|------------|---------|
| 原型 | < 100 | < 1,000 | < 10 MB |
| 小规模 | < 1,000 | < 10,000 | < 100 MB |
| 中规模 | < 10,000 | < 100,000 | < 1 GB |
| 大规模 | > 10,000 | > 100,000 | 切换 Milvus |

---

## 3. 图数据库 — Neo4j (目标) / 内存 GraphStore (当前)

### 3.1 节点类型

| 标签 | 属性 | 示例 |
|------|------|------|
| `Technology` | name, alias[], category | Python, FastAPI, Milvus |
| `Organization` | name, type | TokenHub, 传智 |
| `Person` | name, role | (从文档抽取) |
| `Document` | doc_id, filename | sample.pdf |
| `Chunk` | chunk_id | 分块引用 |

### 3.2 关系类型 (8 种)

| 关系 | 方向 | 语义 | 示例 |
|------|------|------|------|
| `uses` | (Org/Person)→Technology | 使用技术 | 传智 → FastAPI |
| `depends_on` | Technology→Technology | 技术依赖 | FastAPI → Starlette |
| `contains` | Technology→Technology | 包含关系 | Transformers → BGE |
| `supplies` | Organization→Technology | 提供技术 | TokenHub → DeepSeek |
| `references` | Document→Technology | 文档引用 | sample.pdf → Python |
| `employs` | Organization→Person | 雇佣关系 | TokenHub → 员工 |
| `owns` | Organization→Document | 拥有文档 | 传智 → sample.pdf |
| `signs` | Person→Document | 签署文档 | (未来) |

### 3.3 查询模式

```cypher
-- 多跳: 与 "FastAPI" 相关的所有上下游技术
MATCH (t:Technology {name: 'FastAPI'})-[r:uses|depends_on|contains*1..2]-(related)
RETURN t, r, related

-- 技术栈全景: 从文档出发的实体关系图
MATCH (d:Document {doc_id: 'xxx'})-[:references]->(t:Technology)
OPTIONAL MATCH (t)-[r]-(other)
RETURN d, t, r, other
```

---

## 4. 关系型数据库 — PostgreSQL (目标)

### 4.1 文档元数据表

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(500) NOT NULL,
    file_size BIGINT,
    page_count INT,
    chunk_count INT,
    status VARCHAR(20) DEFAULT 'processing',  -- processing / ready / failed
    department VARCHAR(100),
    tags TEXT[],
    access_level VARCHAR(20) DEFAULT 'internal',
    uploaded_at TIMESTAMPTZ DEFAULT now(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_department ON documents(department);
```

### 4.2 QA 日志表 (用于成本追踪)

```sql
CREATE TABLE qa_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    answer TEXT,
    mode VARCHAR(20),           -- hybrid / vector / graph
    retrieval_rounds INT,        -- Self-Retrieval 轮数
    tokens_in INT,               -- prompt tokens
    tokens_out INT,              -- completion tokens
    latency_ms INT,
    source_doc_ids TEXT[],
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_qa_logs_created ON qa_logs(created_at);
```

---

## 5. 缓存 — Redis (目标)

| Key Pattern | 用途 | TTL |
|-------------|------|-----|
| `qa_cache:{md5(question + top_k)}` | QA 结果缓存 | 3600s |
| `rate_limit:{ip}:{endpoint}` | 限流计数器 | 60s |
| `doc_status:{doc_id}` | 文档处理状态 | 无过期 |

---

## 6. 开发/生产对照

| 存储 | 开发环境 | 生产目标 | 切换方式 |
|------|---------|---------|---------|
| 向量 | LanceDB (嵌入式) | Milvus (独立服务) | 替换 VectorStore 实现 |
| 图谱 | 内存 GraphStore | Neo4j 5.x | 替换 GraphStore 实现 |
| 元数据 | 内存 `_docs` 字典 | PostgreSQL + pgvector | 新增 ORM 模型 |
| 缓存 | slowapi 内存 | Redis | 环境变量切换 |
| 对象存储 | 本地磁盘 | MinIO (S3) | 替换文件路径为 S3 URI |

---

## 7. 数据流向图

```
上传 PDF (HTTP multipart)
  │
  ├─ 1. 解析 → UIRDocument
  │     └─ pages: [{text, tables, images, heading_level}]
  │
  ├─ 2. 分块 → Chunk[]
  │     └─ {chunk_id, doc_id, content, heading_chain, embedding}
  │
  ├─ 3. 嵌入 → 512d float32 vector
  │     └─ 存入 LanceDB documents 表
  │
  └─ 4. 图谱抽取 → entities + relations
        └─ 存入内存 GraphStore (规则 → LLM 混合)

QA 请求 (POST /ask)
  │
  ├─ 查询嵌入 → 向量检索 (LanceDB cosine)
  ├─ 实体匹配 → 图谱检索 (多跳遍历)
  ├─ 合并去重 → Reranker 精排
  ├─ [Self-Retrieval 循环: 评估 → 改写 → 重检索]
  └─ LLM 生成 → 返回 {answer, sources, graph_context}
```
