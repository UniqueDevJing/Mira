"""API Key 认证中间件 + 错误脱敏"""

import logging
import os
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
        from api.core.auth import authenticate, anonymous_admin, loopback_principal

        enabled = os.environ.get("RAG_API_KEY_ENABLED") or ("true" if settings.api_key_enabled else "false")
        if enabled.lower() != "true":
            # 鉴权关闭: 注入 anonymous admin 主体, 路由层 RBAC 逻辑仍可一致运行
            request.state.principal = anonymous_admin()
            return await call_next(request)

        if _is_exempt(request.url.path):
            request.state.principal = anonymous_admin()
            return await call_next(request)

        if _is_local_loopback(request):
            # 本机直连免鉴权, 但仍是 admin 主体
            request.state.principal = loopback_principal()
            return await call_next(request)

        # 启用鉴权且为外部请求: 必须有合法 Key
        provided = request.headers.get(API_KEY_HEADER) or _extract_bearer(request.headers.get("Authorization", ""))

        if not provided:
            return JSONResponse(
                status_code=401,
                content={"detail": "缺少 API Key，请通过 X-API-Key 请求头或 Authorization: Bearer <key> 提供"},
            )

        principal = authenticate(provided)
        if principal is None:
            # fail-closed: 提供了 Key 但不在白名单 → 拒绝 (不泄露是否启用)
            logger.warning("API Key 校验失败 (不在白名单): %s", request.url.path)
            return JSONResponse(
                status_code=403,
                content={"detail": "API Key 无效或无访问权限"},
            )

        request.state.principal = principal
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
