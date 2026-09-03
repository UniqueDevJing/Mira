"""API Key 认证中间件 + 错误脱敏"""

import logging
import os
import traceback

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# 管理后台页面 (/admin) 与 Prometheus 指标 (/metrics) 免鉴权直出:
# - /admin 仅交付静态控制台 HTML, 其内数据接口仍走 X-API-Key 鉴权;
# - /metrics 供同栈 Prometheus 抓取 (跨容器非 loopback 客户端), 内部监控标准做法;
# - /web 前端静态资源 (index/admin 引用的 JS/CSS/图标) 公开直出, 数据接口仍在 /api/v1 下鉴权。
EXEMPT_PATHS = {"/health", "/", "/metrics", "/web", "/admin"}


def _is_bound_to_all_interfaces() -> bool:
    return os.environ.get("UVICORN_HOST", "127.0.0.1") == "0.0.0.0"

API_KEY_HEADER = "X-API-Key"
BEARER_PREFIX = "Bearer "


def _is_exempt(path: str) -> bool:
    return path in EXEMPT_PATHS or path.startswith(("/docs", "/openapi", "/web"))


def _is_local_loopback(request: Request) -> bool:
    """本机直连 (127.0.0.1) 免鉴权 — 显式开关 (RAG_LOOPBACK_EXEMPT), 生产默认关 (S4)。

    安全背景: 部署在 nginx/Caddy 等反向代理后, 若代理未设转发头 (proxy_pass 默认不带 XFF),
    所有外部请求会被当成 loopback 直连 → 全 API 匿名 admin 绕过 (含 DELETE 端点)。
    因此 loopback 豁免必须显式开启; 且即便开启, 带 Cf-Connecting-Ip / X-Forwarded-For 头的
    请求仍视为外部 (隧道/代理转发场景), 不豁免。
    """
    from api.config import settings

    if not settings.loopback_exempt:
        return False
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
        from api.core.auth import anonymous_admin, authenticate, loopback_principal

        enabled = os.environ.get("RAG_API_KEY_ENABLED") or ("true" if settings.api_key_enabled else "false")
        if enabled.lower() != "true":
            if _is_bound_to_all_interfaces() and not _is_exempt(request.url.path):
                return JSONResponse(status_code=403, content={"detail": "鉴权已关闭，请勿绑定 0.0.0.0"})
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


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """安全响应头中间件 (OPT-S1)。

    - X-Content-Type-Options / X-Frame-Options / Referrer-Policy: 全响应下发, 零兼容风险。
    - CSP: 仅对 HTML 页面下发 —— index.html 为单文件内联脚本/样式, 必须允许
      'unsafe-inline', 否则前端直接白屏; 对 API JSON/SSE 不下发, 避免无谓干扰。
    - HSTS: 仅当请求经 TLS (X-Forwarded-Proto=https, cloudflared 隧道场景) 时下发,
      避免 http://127.0.0.1 本地调试被浏览器强制升级。
    """

    CSP_HTML = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            response.headers.setdefault("Content-Security-Policy", self.CSP_HTML)
        if request.headers.get("x-forwarded-proto", "").lower() == "https":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


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
