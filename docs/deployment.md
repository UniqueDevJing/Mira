# RAG 2.0 部署文档

**版本**: v2.0（诚实修订版）
**日期**: 2026-08-17
**修订说明**: 本文档已对照代码实际状态修订。原 v1.0 描述了 Celery / Milvus / Neo4j / Redis / MinIO / etcd / PostgreSQL 微服务架构，**这些组件在代码中均未实现**（PRD 列为 Phase 4–6 规划项）。本文档只描述"当前真实可部署形态"，并把规划项明确标注为"待实现"。

---

## ⚠️ 当前真实状态（先读这段）

| 维度 | 真实状态 |
|------|----------|
| 部署形态 | **单 API 容器**（FastAPI + uvicorn），`docker-compose.yml` 仅启动 `api` 一个服务 |
| 向量库 | **LanceDB（本地文件 `./lancedb_data`）**，非 Milvus |
| 图谱 | **内存态（重启即丢）**，无 Neo4j 持久化 |
| 限流 / QA 缓存 | **进程内存**，非 Redis |
| 文档处理 | **FastAPI `BackgroundTasks`**，非 Celery 异步队列 |
| 鉴权 | **单一静态 API Key**（`X-API-Key`），无用户体系 / RBAC |
| TLS / HTTPS | **代码不提供**，需自配反向代理（见 §6） |
| SLA | 目标 P99<3s，**实测 P99≈7.7s@20 并发**，单机 QPS≈3 |

> 结论：当前是**内部 MVP / 演示级可部署**，不是文档化微服务生产架构。对外宣称"生产级微服务部署"会与代码不符。

---

## 1. 部署方式概览

| 方式 | 适用场景 | 复杂度 | 状态 |
|------|---------|--------|------|
| 本地 Python | 开发 / 调试 / 演示 | 低 | ✅ 就绪 |
| Docker Compose（单容器） | 单机内网 / 演示 | 低 | ✅ 就绪 |
| Docker Compose（生产 profile） | 单机生产（非 root + 资源限制） | 低 | ✅ 就绪 |
| Kubernetes | 集群生产 | 高 | ⏳ 待实现 |
| 微服务（Milvus/Neo4j/Celery/Redis） | 规模化生产 | 高 | ⏳ 设计完成 / 待实现（Phase 4–6） |

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
cd rag-2.0
python -m venv venv
.\venv\Scripts\Activate.ps1

# 安装项目（含全部运行依赖）
pip install -e .

# 配置环境变量
$env:HF_ENDPOINT="https://hf-mirror.com"   # 中国大陆 HuggingFace 镜像（必需）
$env:RAG_LLM_API_KEY="your-tokenhub-api-key"

# 启动
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2.3 验证

```powershell
curl http://127.0.0.1:8000/health
# → {"status":"healthy","version":"1.0.0"}

# 打开 Web UI
start http://127.0.0.1:8000
```

### 2.4 首次使用

1. 打开 `http://127.0.0.1:8000`
2. 切到"文档管理"→ 上传 PDF
3. 等待状态变为 `ready`
4. 切到"知识问答"→ 输入问题

---

## 3. Docker Compose 部署（单容器，真实形态）

### 3.1 当前架构（与代码一致）

```
┌──────────────────────────────────────┐
│           单机 Docker Host           │
│                                      │
│   ┌──────────────────────────────┐   │
│   │  api 容器 (FastAPI+uvicorn)  │   │
│   │  :8000  Web UI + REST + SSE  │   │
│   └───────────┬──────────────────┘   │
│               │ 挂载卷                 │
│   ┌───────────┴──────────────────┐   │
│   │  ./lancedb_data  向量（文件） │   │
│   │  ./data          上传/DB      │   │
│   │  内存: 图谱 / 限流 / QA缓存   │   │
│   └──────────────────────────────┘   │
└──────────────────────────────────────┘
```

> 注意：图谱、限流、QA 缓存均为**进程内存**，容器重启后清空；多副本部署会导致状态不一致——当前**不支持水平扩展**。

### 3.2 启动（开发）

```powershell
docker-compose up -d --build
docker-compose logs -f api
```

### 3.3 启动（生产 profile，非 root + 资源限制 + 健康检查）

```powershell
docker-compose -f docker-compose.prod.yml up -d --build
```

`docker-compose.prod.yml` 已包含：非 root 用户、healthcheck、`/metrics` 暴露、日志轮转、CPU/内存限制。

### 3.4 端口

| 服务 | 端口 | 用途 |
|------|------|------|
| API | 8000 | Web UI + REST API + SSE 流式 |
| Prometheus 指标 | 8000/metrics | 可观测性（Grafana dashboard 见 `docs/`） |

---

## 4. 配置参考（当前真实生效项）

### 4.1 实际生效的环境变量

```bash
# LLM（必填）
RAG_LLM_API_KEY=sk-xxx                    # TokenHub API Key
RAG_LLM_BASE_URL=https://tokenhub.itcast.cn/v1
RAG_LLM_MODEL=deepseek-v4-flash           # 注意：该模型强制 reasoning，单次生成 ~6s，是 P99 偏高主因

# Embedding
RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
RAG_EMBEDDING_DEVICE=cpu

# 鉴权
RAG_API_KEY_ENABLED=true                  # 生产必须 true；.env.example 默认 false
RAG_API_KEY=change_me                     # 生产必须改为强随机值并轮转

# HuggingFace 镜像（中国大陆必需）
HF_ENDPOINT=https://hf-mirror.com
```

### 4.2 以下变量为"规划项占位"，当前代码不消费

> 仅当实现 Phase 4–6 后才生效，请勿在简历/文档中声称已接入。

```bash
RAG_MILVUS_HOST / RAG_MILVUS_PORT     # ⏳ 规划：向量库切换 Milvus
RAG_NEO4J_URI / _USER / _PASSWORD     # ⏳ 规划：图谱持久化 Neo4j
RAG_REDIS_HOST / _PORT                # ⏳ 规划：集中限流 + QA 缓存
RAG_MINIO_ENDPOINT / _ACCESS/SECRET   # ⏳ 规划：对象存储
RAG_DATABASE_URL                      # ⏳ 规划：PostgreSQL 关系库
```

### 4.3 .env 模板

```bash
cp .env.example .env
# 编辑 .env：替换 RAG_LLM_API_KEY，生产环境设 RAG_API_KEY_ENABLED=true 并修改 RAG_API_KEY
```

> 安全：`.env` 已被 `.gitignore` 排除（实测未入库）。但本地 `.env` 若含真实 Key，需手动轮转，勿提交。

---

## 5. 运维手册

### 5.1 日志

代码使用 `print` 输出到 stdout（结构化日志为规划项）。`docker-compose.prod.yml` 已配置日志轮转。

### 5.2 健康检查

```bash
curl http://localhost:8000/health
# → {"status":"healthy","version":"1.0.0"}
```

### 5.3 故障排查

| 故障 | 排查 |
|------|------|
| 端口占用 | `netstat -ano \| findstr 8000` |
| 模型下载失败 | 检查 `HF_ENDPOINT=https://hf-mirror.com` |
| LLM 超时 | 检查 TokenHub Key / 网络；deepseek-v4-flash 强制 reasoning 致 ~6s/次 |
| LanceDB 损坏 | 删除 `./lancedb_data` 重建 |
| 内存不足 | BGE-small 约 500MB RAM |

### 5.4 备份

| 数据 | 路径 | 方式 |
|------|------|------|
| LanceDB 向量 | `./lancedb_data/` | 目录整体复制 |
| 上传 PDF / SQLite | `./data/` | 目录整体复制 |
| 图谱 / 限流 / QA 缓存 | 内存 | ⚠️ 重启即丢，规划切换 Neo4j/Redis 后解决 |

---

## 6. TLS / 反向代理（代码不提供，需自配）

生产环境**必须在 API 前加反向代理**做 TLS 终止与基础防护。示例（Caddyfile）：

```caddy
rag.your-domain.com {
    encode gzip
    reverse_proxy localhost:8000
}
```

或使用 Nginx：`location / { proxy_pass http://127.0.0.1:8000; proxy_set_header X-Forwarded-Proto $scheme; }`。
反向代理层可同时承担：HTTPS、限流、访问日志、基础 WAF。

---

## 7. 生产部署检查清单（诚实版）

部署到**外部/生产**前逐项确认：

- [ ] `RAG_API_KEY_ENABLED=true` 且 `RAG_API_KEY` 已改为强随机值并轮转
- [ ] 反向代理已配置 TLS（HTTPS），非裸 8000 暴露
- [ ] CORS 限制特定域名（非 `*`）
- [ ] 数据卷 `./lancedb_data` 与 `./data` 已配置定期备份
- [ ] `/health` 与 `/metrics` 已接入监控告警
- [ ] 已认知并接受：当前**单机、不支持水平扩展、图谱/缓存重启即丢**
- [ ] SLA 已对外如实说明（P99≈7.7s@20 并发，非 <3s）
- [ ] 降级链已验证（逐一断外部依赖测试 L0–L3）

### 若要达到"规模化生产"，需先实现（Phase 4–6，当前均为设计态）

- [ ] 向量库切换 Milvus（或 LanceDB 压测验证单机容量）
- [ ] 图谱持久化 Neo4j
- [ ] 文档处理改 Celery + Redis 异步队列
- [ ] 限流 / QA 缓存改 Redis 集中式
- [ ] PostgreSQL 关系库 + 多租户
- [ ] K8s 编排 + 多副本
