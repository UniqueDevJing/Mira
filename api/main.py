"""RAG 2.0 API 入口"""
import os
import logging

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from api.routes import documents, qa
from api.middleware.security import APIKeyMiddleware, error_sanitization_handler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.config import settings
    logger.info("启动 RAG 2.0 服务 — 模型: %s", settings.llm_model)

    # 预热模型（确保所有模型在请求到来前加载完毕）
    try:
        from engines.embedding.embedder import EmbeddingService
        EmbeddingService().embed_query('warmup')
        logger.info("模型预热完成")
    except Exception as e:
        logger.warning("模型预热跳过: %s", str(e)[:200])

    yield
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

# CORS: 开发环境允许全部，生产环境通过 RAG_CORS_ORIGINS 环境变量限制
from api.config import settings
cors_origins = settings.cors_origins
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"])

app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    from api.core.metrics import get_metrics
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(get_metrics(), media_type="text/plain; version=0.0.4")


@app.get("/")
async def root():
    import os
    web_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "index.html")
    return FileResponse(web_path)
