# RAG 2.0 部署文档

**版本**: v1.0  
**日期**: 2026-08-04

---

## 1. 部署方式概览

| 方式 | 适用场景 | 复杂度 | 状态 |
|------|---------|--------|------|
| 本地 Python | 开发 / 调试 | 低 | ✅ 就绪 |
| Docker Compose | 单机生产 / 集成测试 | 中 | ⚠️ 引擎兼容问题 |
| Kubernetes | 集群生产 | 高 | ⏳ 待实现 |

---

## 2. 本地 Python 部署

### 2.1 环境要求

| 组件 | 最低版本 | 用途 |
|------|---------|------|
| Python | 3.12+ | 运行环境 |
| pip | 24+ | 依赖管理 |
| Git | 2.40+ | 版本管理 |

### 2.2 安装步骤

```powershell
# 1. 克隆项目
cd C:\Users\Dominion\Desktop
git clone <repo-url> rag-2.0

# 2. 创建虚拟环境
cd rag-2.0
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install fastapi uvicorn[standard] pydantic pydantic-settings \
    pymupdf pdfplumber paddleocr paddlepaddle sentence-transformers \
    lancedb httpx jieba

# 4. 配置环境变量
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:RAG_LLM_API_KEY="your-tokenhub-api-key"

# 5. 启动
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2.3 验证

```powershell
# 健康检查
curl http://127.0.0.1:8000/health
# → {"status":"healthy","version":"1.0.0"}

# 打开 Web UI
start http://127.0.0.1:8000
```

### 2.4 首次使用

1. 打开 `http://127.0.0.1:8000`
2. 切换到"文档管理"标签
3. 上传 PDF 文件
4. 等待处理完成（状态变为 ready）
5. 切换到"知识问答"标签
6. 输入问题测试

---

## 3. Docker Compose 部署

### 3.1 架构

```
┌──────────────────────────────────────────┐
│              Docker Network              │
│                                          │
│  ┌──────┐  ┌──────┐  ┌──────┐          │
│  │ API  │  │Worker│  │Neo4j │          │
│  │:8000 │  │Celery│  │:7687 │          │
│  └──┬───┘  └──┬───┘  └──────┘          │
│     │         │                          │
│  ┌──┴─────────┴──────────────────────┐  │
│  │  Redis :6379  (Broker + Cache)    │  │
│  └───────────────────────────────────┘  │
│                                          │
│  ┌──────────┐  ┌──────────┐            │
│  │ Milvus   │  │MinIO     │            │
│  │ :19530   │  │:9000     │            │
│  └──┬───────┘  └──────────┘            │
│     │                                    │
│  ┌──┴──────┐  ┌──────────┐            │
│  │  etcd   │  │PostgreSQL│            │
│  │  :2379  │  │  :5432   │            │
│  └─────────┘  └──────────┘            │
└──────────────────────────────────────────┘
```

### 3.2 启动

```powershell
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止
docker-compose down
```

### 3.3 服务端口映射

| 服务 | 外部端口 | 内部端口 | 用途 |
|------|---------|---------|------|
| API | 8000 | 8000 | Web UI + REST API |
| Milvus | 19530 | 19530 | 向量数据库 |
| Milvus Metrics | 9091 | 9091 | Prometheus 指标 |
| etcd | 2379 | 2379 | Milvus 元数据 |
| MinIO | 9000 / 9001 | 9000 / 9001 | S3 存储 / Console |
| Neo4j | 7474 / 7687 | 7474 / 7687 | HTTP / Bolt |
| Redis | 6379 | 6379 | 缓存 |
| PostgreSQL | 5432 | 5432 | 关系型数据库 |

### 3.4 已知问题

| 问题 | 影响 | 状态 |
|------|------|------|
| Docker Desktop v29.5.3 引擎超时 | 无法启动 | ⏳ 需重置引擎或降级 |
| etcd 缺少 command 配置 | 容器崩溃 | ✅ 已修复 |
| MinIO 认证环境变量缺失 | Milvus 无法连接 MinIO | ✅ 已修复 |

---

## 4. 配置参考

### 4.1 环境变量完整列表

```bash
# LLM (必填)
RAG_LLM_API_KEY=sk-xxx                    # TokenHub API Key
RAG_LLM_BASE_URL=https://tokenhub.itcast.cn/v1  # API 地址
RAG_LLM_MODEL=deepseek-v4-flash           # 模型名称

# Embedding
RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5  # 嵌入模型
RAG_EMBEDDING_DEVICE=cpu                  # 推理设备 (cpu/cuda)

# 向量库 (生产环境)
RAG_MILVUS_HOST=localhost
RAG_MILVUS_PORT=19530

# 图数据库 (生产环境)
RAG_NEO4J_URI=bolt://localhost:7687
RAG_NEO4J_USER=neo4j
RAG_NEO4J_PASSWORD=password123

# Redis
RAG_REDIS_HOST=localhost
RAG_REDIS_PORT=6379

# 对象存储
RAG_MINIO_ENDPOINT=localhost:9000
RAG_MINIO_ACCESS_KEY=minioadmin
RAG_MINIO_SECRET_KEY=minioadmin

# 数据库
RAG_DATABASE_URL=postgresql+asyncpg://raguser:ragpass@localhost:5432/rag20

# OCR
RAG_OCR_LANG=ch
RAG_OCR_USE_GPU=false

# HuggingFace 镜像 (中国大陆必需)
HF_ENDPOINT=https://hf-mirror.com
```

### 4.2 .env 文件模板

复制 `.env.example` 为 `.env`，填写必填项：

```bash
cp .env.example .env
# 编辑 .env，替换 RAG_LLM_API_KEY
```

---

## 5. 运维手册

### 5.1 日志

当前使用 `print` 输出到 stdout。生产目标为结构化日志（JSON 格式），方便日志聚合系统解析。

```python
# 当前
print(f"[{doc_id}] 解析完成: {len(uir.pages)} 页")

# 目标
logger.info("文档解析完成", extra={"doc_id": doc_id, "pages": len(uir.pages)})
```

### 5.2 健康检查

```bash
curl http://localhost:8000/health
# 期望: {"status":"healthy","version":"1.0.0"}
```

### 5.3 常见故障排查

| 故障 | 排查步骤 |
|------|---------|
| 端口占用 | `netstat -ano | findstr 8000`，kill 占用进程 |
| 模型下载失败 | 检查 `HF_ENDPOINT=https://hf-mirror.com` 是否设置 |
| LLM 调用超时 | 检查 TokenHub API Key 是否有效，网络是否可达 |
| LanceDB 数据损坏 | 删除 `./lancedb_data` 目录重新生成 |
| 内存不足 | BGE-small 需 ~500MB RAM，关闭其他应用 |

### 5.4 备份策略

| 数据 | 路径 | 备份方式 |
|------|------|---------|
| LanceDB 向量 | `./lancedb_data/` | 目录整体复制 |
| 上传的 PDF | 内存中（当前）| 待切换到 MinIO 后 S3 sync |
| 配置文件 | `.env` | Git 不纳入，手动备份 |

---

## 6. 生产部署检查清单

部署到生产前逐项确认：

- [ ] API Key 通过环境变量注入（不硬编码）
- [ ] CORS 限制特定域名（非 `*`）
- [ ] API Key 认证中间件已启用
- [ ] 向量库切换为 Milvus（或 LanceDB 已压测验证）
- [ ] 图谱存储切换为 Neo4j（数据持久化）
- [ ] 文档处理走 Celery 异步（不阻塞 HTTP 线程）
- [ ] Redis 集中限流已配置
- [ ] 结构化日志已启用
- [ ] 健康检查端点已配置监控告警
- [ ] 数据备份策略已就绪
- [ ] 降级链已验证（逐一断掉外部依赖测试）
- [ ] 压测通过（目标 QPS 下的 P99 延迟达标）
