"""API Key 认证中间件 + 错误脱敏"""

import logging
import os
import secrets
import traceback

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


EXEMPT_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc"}

API_KEY_HEADER = "X-API-Key"
BEARER_PREFIX = "Bearer "


def _is_exempt(path: str) -> bool:
    return path in EXEMPT_PATHS or path.startswith(("/docs", "/openapi"))


def _is_local_loopback(request: Request) -> bool:
    """本机直连 (127.0.0.1) 免鉴权, 局域网/公网仍需 Key。

    隧道 (cloudflared) 转发到本机时 client 也是 127.0.0.1, 但带
    Cf-Connecting-Ip / X-Forwarded-For 头 → 视为外部请求, 不豁免。
    """
    host = request.client.host if request.client else ""
    has_forward_header = bool(request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for"))
    return host in ("127.0.0.1", "::1", "localhost") and not has_forward_header


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API Key 认证中间件。

    仅在 RAG_API_KEY_ENABLED=true 时激活。
    支持两种传递方式:
      - Authorization: Bearer <key>
      - X-API-Key: <key>
    本机直连 (127.0.0.1) 免鉴权; 局域网/公网/隧道需 Key。
    """

    async def dispatch(self, request: Request, call_next):
        # 仅在启用时验证. os.environ 优先 (测试/运行时动态覆盖), 否则读 settings (.env 注入)
        from api.config import settings

        enabled = os.environ.get("RAG_API_KEY_ENABLED") or ("true" if settings.api_key_enabled else "false")
        if enabled.lower() != "true":
            return await call_next(request)

        if _is_exempt(request.url.path):
            return await call_next(request)

        if _is_local_loopback(request):
            return await call_next(request)

        expected = os.environ.get("RAG_API_KEY") or settings.api_key
        if not expected:
            # fail-closed: 启用鉴权却未配置 Key 属部署错误, 拒绝服务而非静默放行
            logger.error("RAG_API_KEY_ENABLED=true 但 RAG_API_KEY 为空 — 拒绝所有外部请求 (fail-closed)")
            return JSONResponse(
                status_code=503,
                content={"detail": "服务端未配置 API Key，请联系管理员"},
            )

        provided = request.headers.get(API_KEY_HEADER) or _extract_bearer(request.headers.get("Authorization", ""))

        if not provided:
            return JSONResponse(
                status_code=401,
                content={"detail": "缺少 API Key，请通过 X-API-Key 请求头或 Authorization: Bearer <key> 提供"},
            )
        if not secrets.compare_digest(provided, expected):
            return JSONResponse(
                status_code=403,
                content={"detail": "API Key 无效"},
            )

        return await call_next(request)


def _extract_bearer(auth_header: str) -> str | None:
    if auth_header.startswith(BEARER_PREFIX):
        return auth_header[len(BEARER_PREFIX) :]
    return None


async def error_sanitization_handler(request: Request, call_next):
    """全局异常处理 — 捕获未处理异常并脱敏后返回。

    不暴露内部堆栈，日志保留完整 traceback 用于排查。
    """
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 — 全局错误边界: 吞所有异常返回脱敏 500
        tb = traceback.format_exc()
        logger.error("请求异常: %s %s\n%s", request.method, request.url.path, tb)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "服务器内部错误，请稍后重试。如持续出现请联系管理员。",
            },
        )
