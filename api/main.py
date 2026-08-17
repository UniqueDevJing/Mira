"""RAG 2.0 API 入口"""

import logging
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.middleware.security import APIKeyMiddleware, error_sanitization_handler
from api.routes import documents, qa

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.config import settings

    logger.info("启动 RAG 2.0 服务 — 模型: %s", settings.llm_model)

    # 预热模型（确保所有模型在请求到来前加载完毕）
    try:
        from engines.embedding.embedder import EmbeddingService

        EmbeddingService().embed_query("warmup")
        logger.info("模型预热完成")
    except Exception as e:  # noqa: BLE001 — 预热是尽力而为, 失败不阻塞启动
        logger.warning("模型预热跳过: %s", str(e)[:200])

    # 重置上次中断遗留的 processing 文档 (后台任务随进程死亡, 状态卡死)
    from api.core.document_store import get_document_store

    get_document_store().reset_stale_processing()

    # 生产安全自检: 鉴权/限流默认关闭属 fail-open, 仅适合本地开发 (A3)
    if not settings.api_key_enabled:
        logger.warning(
            "⚠️ API Key 鉴权未启用 (RAG_API_KEY_ENABLED=false) — 仅适合本地开发；"
            "生产环境请设 RAG_API_KEY_ENABLED=1 并配置 RAG_API_KEY"
        )
    if not settings.rate_limit_enabled:
        logger.warning("⚠️ 限流未启用 (RAG_RATE_LIMIT_ENABLED=false) — 生产环境建议启用 RAG_RATE_LIMIT_ENABLED=1")

    yield
    # 关闭共享抽取器的 LLM 客户端连接 (keep-alive 释放)
    from api.state import close_extractors

    close_extractors()
    logger.info("服务关闭")


app = FastAPI(
    title="RAG 2.0 API",
    description="智能文档处理与知识库构建系统",
    version="1.0.0",
    lifespan=lifespan,
)

# 中间件顺序: 错误脱敏 → API Key 认证 → CORS
app.middleware("http")(error_sanitization_handler)
app.add_middleware(APIKeyMiddleware)

# CORS: 默认仅前端 origin, 通过 RAG_CORS_ORIGINS 覆盖
from api.config import settings

cors_origins = settings.cors_origins
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"])

# 结构化日志 (JSON + trace_id): 让日志机器可解析、可按 trace_id 串联一次请求
from api.middleware.logging_middleware import RequestLoggingMiddleware, configure_json_logging

configure_json_logging()
app.add_middleware(RequestLoggingMiddleware)

# 限流 (slowapi): RAG_RATE_LIMIT_ENABLED=1 启用; 默认内存存储单进程有效, 配 RAG_SHARED_STATE_BACKEND=redis 后多 worker 共享
from api.core.limiter import limiter

if limiter is not None:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("限流已启用: %d 次/分钟/IP", settings.rate_limit_per_minute)

app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点 — 通过 to_thread 避免同步 IO 阻塞事件循环"""
    import asyncio

    from fastapi.responses import PlainTextResponse

    from api.core.metrics import get_metrics

    body = await asyncio.to_thread(get_metrics)
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4")


@app.get("/")
async def root():
    import os

    web_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "index.html")
    return FileResponse(web_path)
