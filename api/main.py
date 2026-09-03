"""RAG 2.0 API 入口"""

import logging
import os
import threading

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.middleware.security import APIKeyMiddleware, error_sanitization_handler
from api.routes import auth, documents, qa

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.config import settings

    logger.info("启动 RAG 2.0 服务 — 模型: %s", settings.llm_model)

    # 后台预热模型: 不阻塞服务启动, 让 /health 与前端立即可用;
    # 模型在独立线程加载, 首个查询若未就绪会自动触发懒加载(幂等单例)。
    def _warmup():
        try:
            from api.state import get_embedder

            # 预热当前配置的真实单例模型 (get_embedder 用 settings.embedding_model)
            get_embedder().embed_query("warmup")
            logger.info("Embedding 模型预热完成")
        except Exception as e:  # noqa: BLE001 — 预热是尽力而为, 失败不阻塞启动
            logger.warning("Embedding 模型预热跳过: %s", str(e)[:200])

        # Reranker 预热: CE 模型约 600MB, 不预热首个查询会卡在下载上并超时降级
        if settings.reranker_model:
            try:
                from api.state import get_reranker

                reranker = get_reranker()  # 未生效(CPU/显式关闭)时返回 None, 跳过预热
                if reranker is not None and reranker.warmup():
                    logger.info("Reranker 模型预热完成: %s", settings.reranker_model)
                else:
                    logger.warning(
                        "⚠️ Reranker 模型不可用, 重排降级为不重排: %s", settings.reranker_model
                    )
            except Exception as e:  # noqa: BLE001 — 预热失败不阻塞启动
                logger.warning("Reranker 预热跳过: %s", str(e)[:200])

    threading.Thread(target=_warmup, daemon=True).start()

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

# 安全响应头 (OPT-S1): nosniff / X-Frame-Options / Referrer-Policy 全响应;
# CSP 仅 HTML, HSTS 仅 https。注册在最后 = 最外层, 保证含错误响应在内都有安全头。
from api.middleware.security import SecurityHeadersMiddleware

app.add_middleware(SecurityHeadersMiddleware)

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
app.include_router(auth.router)

# 前端静态资源 (common.js / icons.js / markdown.js / common.css) — 去除 Font Awesome CDN 依赖, 离线可用
from fastapi.staticfiles import StaticFiles

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web-static")


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点 — 通过 to_thread 避免同步 IO 阻塞事件循环"""
    import asyncio

    from fastapi.responses import PlainTextResponse

    from api.core.metrics import get_metrics
    from api.core.system_stats import update_system_gauges

    def _collect():
        update_system_gauges()  # 先刷新系统级 Gauge, 再导出
        return get_metrics()

    body = await asyncio.to_thread(_collect)
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4")


@app.get("/api/v1/metrics/summary", tags=["metrics"])
async def metrics_summary_view():
    """只读: 指标汇总 JSON — 与 /metrics 同源, 供前端指标/降级可视化面板消费。

    覆盖请求量与成功率、延迟与阶段耗时、质量信号、降级等级分布、路由来源分布、
    缓存命中率、跨库兜底、按文档类型的切分统计与系统规模。
    与 /metrics 一致先刷新系统级 Gauge 再导出, 且采集在线程池执行避免阻塞事件循环。
    """
    import asyncio

    from api.core.metrics import metrics_summary
    from api.core.system_stats import update_system_gauges

    def _collect():
        update_system_gauges()  # 先刷新系统级 Gauge (向量库行数/图谱规模), 再折算
        return metrics_summary()

    return await asyncio.to_thread(_collect)


@app.get("/api/v1/metrics/history", tags=["metrics"])
async def metrics_history_view():
    """只读: 历史问答基线 — 跨重启的项目级历史指标。

    与 /api/v1/metrics/summary (进程内实时累计, 重启清零) 互补: 本接口读
    data/qa_export.json 导出快照, 覆盖降级分布、路由来源、知识库分布、延迟
    精确分位数、忠实度、按天趋势与数据质量说明。带 mtime 缓存, 解析在线程池
    执行避免阻塞事件循环。文件缺失时返回 available=False, 不抛错。
    """
    import asyncio

    from api.core.qa_history import history_summary

    return await asyncio.to_thread(history_summary)


@app.get("/")
async def root():
    import os

    web_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "index.html")
    return FileResponse(web_path)


@app.get("/admin")
async def admin():
    import os

    web_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "admin.html")
    return FileResponse(web_path)
