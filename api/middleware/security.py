"""API Key 认证中间件 + 错误脱敏"""
import os
import traceback
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


EXEMPT_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc"}

API_KEY_HEADER = "X-API-Key"
BEARER_PREFIX = "Bearer "


def _is_exempt(path: str) -> bool:
    return path in EXEMPT_PATHS or path.startswith("/docs") or path.startswith("/openapi")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API Key 认证中间件。

    仅在 RAG_API_KEY_ENABLED=true 时激活。
    支持两种传递方式:
      - Authorization: Bearer <key>
      - X-API-Key: <key>
    """

    async def dispatch(self, request: Request, call_next):
        # 仅在启用时验证
        if os.environ.get("RAG_API_KEY_ENABLED", "").lower() != "true":
            return await call_next(request)

        if _is_exempt(request.url.path):
            return await call_next(request)

        expected = os.environ.get("RAG_API_KEY", "")
        if not expected:
            return await call_next(request)

        provided = (
            request.headers.get(API_KEY_HEADER)
            or _extract_bearer(request.headers.get("Authorization", ""))
        )

        if not provided:
            return JSONResponse(
                status_code=401,
                content={"detail": "缺少 API Key，请通过 X-API-Key 请求头或 Authorization: Bearer <key> 提供"},
            )
        if provided != expected:
            return JSONResponse(
                status_code=403,
                content={"detail": "API Key 无效"},
            )

        return await call_next(request)


def _extract_bearer(auth_header: str) -> str | None:
    if auth_header.startswith(BEARER_PREFIX):
        return auth_header[len(BEARER_PREFIX):]
    return None


async def error_sanitization_handler(request: Request, call_next):
    """全局异常处理 — 捕获未处理异常并脱敏后返回。

    不暴露内部堆栈，日志保留完整 traceback 用于排查。
    """
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR] {request.method} {request.url.path}\n{tb}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "服务器内部错误，请稍后重试。如持续出现请联系管理员。",
            },
        )


def sanitize_error_response(detail: str) -> dict:
    """脱敏通用错误响应"""
    return {"detail": detail}
