"""结构化日志中间件 — 零依赖 (标准库 logging), 让日志可被 Loki/ELK 机器解析。

- trace_id: 每个 HTTP 请求分配一个, 存 ContextVar, 全链路日志自动携带, 可按 trace_id 串联一次请求
- JsonFormatter: 输出单行 JSON (ts/level/logger/msg/trace_id + 任意 extra.structured 字段)
- RequestLoggingMiddleware: 记录每个请求 method/path/status/latency

用法 (在 api/main.py 启动时):
    from api.middleware.logging_middleware import RequestLoggingMiddleware, configure_json_logging
    configure_json_logging()
    app.add_middleware(RequestLoggingMiddleware)
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware

# 每个请求一个 trace_id; 默认 "-" 避免无请求上下文 (如 lifespan) 时报错
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="-")


class JsonFormatter(logging.Formatter):
    """单行 JSON 格式化; 自动附加 trace_id, 支持 record.structured 额外字段。"""

    def format(self, record: logging.LogRecord) -> str:
        extra = getattr(record, "structured", None)
        # trace_id 优先取 record 显式传入 (中间件路径), 否则回退 contextvar (请求上下文内可见)
        tid = trace_id_ctx.get()
        if isinstance(extra, dict) and extra.get("trace_id"):
            tid = extra["trace_id"]
        log = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": tid,
        }
        if isinstance(extra, dict):
            log.update({k: v for k, v in extra.items() if k != "trace_id"})
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        return json.dumps(log, ensure_ascii=False)


def configure_json_logging() -> None:
    """将根 logger 的所有 handler 切换为 JsonFormatter (幂等, 可重复调用)。

    仅在尚未配置时生效, 避免重复设置或覆盖 uvicorn 的初次配置意图。
    """
    root = logging.getLogger()
    already = any(isinstance(h.formatter, JsonFormatter) for h in root.handlers if h.formatter)
    if already:
        return
    fmt = JsonFormatter()
    for h in root.handlers:
        h.setFormatter(fmt)
    # 若根 logger 无任何 handler (非 uvicorn 启动场景), 补一个 stderr handler
    if not root.handlers:
        import sys

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(fmt)
        root.addHandler(handler)
    # 应用代码 logger (如 api.*) 通常 propagate 到 root, 无需单独改


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """每个请求分配 trace_id, 结束时记录结构化访问日志 (含 status/latency)。"""

    async def dispatch(self, request, call_next):
        tid = uuid.uuid4().hex[:16]
        token = trace_id_ctx.set(tid)
        start = time.time()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception:
            status = 500
            raise
        finally:
            trace_id_ctx.reset(token)
            logging.getLogger("api.access").info(
                "request",
                extra={
                    "structured": {
                        "trace_id": tid,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "ms": round((time.time() - start) * 1000, 1),
                    }
                },
            )
