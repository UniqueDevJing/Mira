

# RAG 2.0 API 设计文档

**版本**: v1.0  
**日期**: 2026-08-04  
**Base URL**: `http://localhost:8000`

---

## 1. 设计原则

| 原则 | 实践 |
|------|------|
| RESTful | 资源导向 URL，HTTP 方法语义正确 |
| 版本化 | 路径前缀 `/api/v1/`，兼容未来变更 |
| 中文错误 | 用户可见错误中文化，技术日志英文 |
| 降级友好 | 所有端点出错返回部分结果而非 500 |
| 无状态 | 通过环境变量配置，Session 不存储在服务端 |

---

## 2. 接口总览

```
/api/v1/
├── documents/                    # 文档管理
│   ├── POST   /upload            # 上传文档
│   ├── GET    /                  # 文档列表（分页）
│   ├── GET    /{doc_id}/status   # 查询处理状态
│   └── DELETE /{doc_id}          # 删除文档（待实现）
├── qa/                           # 知识问答
│   └── POST   /ask               # 提问
└── 系统
    ├── GET    /health            # 健康检查
    ├── GET    /docs              # Swagger API 文档 (debug 模式)
    └── GET    /                  # Web UI
```

---

## 3. 接口详细定义

### 3.1 健康检查

```
GET /health
```

**响应**: `200 OK`
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### 3.2 文档上传

```
POST /api/v1/documents/upload
Content-Type: multipart/form-data
```

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File (.pdf) | 是 | PDF 文档，最大 50MB |
| department | string | 否 | 所属部门 |
| tags | string | 否 | 逗号分隔标签 |
| access_level | string | 否 | 权限级别，默认 "internal" |

**响应**: `200 OK`
```json
{
  "doc_id": "a1b2c3d4e5f6",
  "status": "ready",
  "estimated_time": 0
}
```

**错误响应**:
```json
{
  "doc_id": "a1b2c3d4e5f6",
  "status": "failed",
  "error": "PDF 解析失败：文件损坏或为扫描件"
}
```

**设计决策**:
- 当前为同步处理（开发阶段），生产目标为异步（返回 `processing` → 轮询 status）
- doc_id 用 `uuid4()[:12]` 缩短，兼顾唯一性和可读性

---

### 3.3 文档列表

```
GET /api/v1/documents?page=1&size=20
```

**响应**: `200 OK`
```json
{
  "items": [
    {
      "doc_id": "a1b2c3d4e5f6",
      "filename": "技术方案.pdf",
      "status": "ready"
    }
  ],
  "total": 42
}
```

---

### 3.4 文档状态查询

```
GET /api/v1/documents/{doc_id}/status
```

**响应**: `200 OK`
```json
{
  "doc_id": "a1b2c3d4e5f6",
  "filename": "技术方案.pdf",
  "status": "ready",
  "page_count": 12,
  "chunk_count": 45
}
```

**状态枚举**: `processing` | `ready` | `failed` | `not_found`

---

### 3.5 知识问答 (核心接口)

```
POST /api/v1/qa/ask
Content-Type: application/json
```

**请求体**:
```json
{
  "question": "系统使用了哪些技术？",
  "mode": "hybrid",
  "enable_self_retrieval": true,
  "top_k": 10,
  "filters": null
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| question | string | 是 | — | 自然语言问题 |
| mode | string | 否 | "hybrid" | 检索模式: hybrid / vector / graph |
| enable_self_retrieval | bool | 否 | true | 启用多轮自适应检索 |
| top_k | int | 否 | 10 | 返回文档数 (1-50) |
| filters | object | 否 | null | 过滤条件: {"department": "技术部"} |

**响应**: `200 OK`
```json
{
  "answer": "系统使用了以下技术：FastAPI 作为 Web 框架、BGE-small-zh 作为嵌入模型、LanceDB 作为向量存储、PyMuPDF 解析 PDF...",
  "sources": [
    {
      "id": "chunk-uuid-1",
      "chunk_id": "chunk-uuid-1",
      "doc_id": "a1b2c3d4e5f6",
      "content": "技术栈方面，系统采用 FastAPI 作为...",
      "score": 0.87
    }
  ],
  "graph_context": {
    "entities": ["FastAPI", "BGE-small-zh", "LanceDB"],
    "relations": [{"source": "RAG 2.0", "relation": "uses", "target": "FastAPI"}]
  },
  "retrieval_rounds": 1,
  "rewritten_queries": [],
  "latency_ms": 1234.56
}
```

**降级行为**:
| 场景 | 响应 |
|------|------|
| 知识库为空 | `"未在知识库中找到相关信息，请先上传文档。"` |
| LLM API 不可用 | `"（LLM 暂时不可用）检索到 N 条相关内容。"` |
| reasoning_content 超限 | fallback 到 `reasoning_content` 前 200 字符 |

---

## 4. 错误码规范

| HTTP Status | 场景 | 响应格式 |
|-------------|------|---------|
| 200 | 成功或业务级错误 | `{"error": "..."}` 内嵌 |
| 422 | Pydantic 校验失败 | FastAPI 默认格式 |
| 500 | 未捕获异常 | FastAPI 默认 + 脱敏处理 |

**设计决策**: 不在 200 外使用自定义错误码，减少前后端协议复杂度。业务错误在 200 内通过 `error` 字段传递。

---

## 5. 检索流水线

```
POST /ask
  │
  ├─ 1. EmbeddingService.embed_query(question)
  │     └─ BGE-small-zh → 512d vector
  │
  ├─ 2. HybridRetriever.retrieve(query, top_k * 2)
  │     ├─ VectorStore.search(query_emb, top_k=40)  → LanceDB cosine
  │     └─ GraphRAGRetriever.retrieve(query, top_k)  → 多跳图遍历
  │
  ├─ 3. Reranker.rerank(query, docs, top_k)
  │     └─ Embedding 余弦相似度排序
  │
  ├─ 4. [可选] SelfRetrieval.retrieve(query, top_k)
  │     └─ 循环: 评估 → 改写 → 重检索，最多 3 轮
  │
  └─ 5. LLM 生成
        ├─ 拼接 context (Top-5 docs)
        ├─ POST TokenHub /chat/completions
        └─ 解析 response → answer
```

---

## 6. 待实现接口

| 方法 | 路径 | 说明 | 优先级 |
|------|------|------|--------|
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档及关联向量/图谱数据 | P1 |
| POST | `/api/v1/documents/upload-batch` | 批量上传 | P2 |
| GET | `/api/v1/evaluate/status` | 检索质量评估 | P2 |
| GET | `/api/v1/stats` | Token 消耗统计 | P1 |
| GET | `/metrics` | Prometheus 指标端点 | P2 |
