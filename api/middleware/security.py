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
    """判断路径是否免鉴权。

    注意：这里必须用「路径段边界」匹配，不能直接 startswith 裸前缀。
    裸前缀会把同名开头的业务路径一起豁免，例如
        startswith("/web")  → /webassets/app.js、/webhook/callback 被误豁免
        startswith("/docs") → /docs-internal/config 被误豁免
        startswith("/openapi") → /openapi-secret/keys 被误豁免
    一旦将来新增以这些前缀命名的业务路由，鉴权会被静默跳过，
    而中间件是 fail-closed 设计，这个漏洞不会有任何报错提示。

    正确做法：路径**完全相等**，或前缀后紧跟 `/`（子路径）：
        /web/index.html  → 豁免（/web 后紧跟 /）
        /docs            → 豁免（完全相等；FastAPI 的 Swagger UI 根路径）
        /openapi.json    → 豁免（完全相等）
        /webhooks/cb     → 不豁免（/web 后紧跟 h，不是 /）
        /docs-internal/  → 不豁免
    """
    if path in EXEMPT_PATHS:
        return True
    # FastAPI 默认端点：两个根路径本身不是子路径形式，必须单独列出
    if path in ("/docs", "/openapi.json"):
        return True
    for prefix in ("/docs", "/openapi", "/web"):
        if path.startswith(prefix + "/"):
            return True
    return False


def _is_local_loopback(request: Request) -> bool:
    """判定「本机直连」(开发/运维放行): host 必须是 loopback 且**不能带反向代理转发头**。

    安全背景 (S5 漏洞根因): 部署在 nginx/Caddy/cloudflared 后, 反向代理把公网请求转发到
    127.0.0.1, 若仅按 host 判断会被伪装成内部, 导致未启用鉴权时公网请求获得 admin。
    因此「带 X-Forwarded-For / Cf-Connecting-Ip / X-Real-IP 即视为外部」**无条件生效**,
    不依赖 RAG_LOOPBACK_EXEMPT 开关 —— 即便本机直连豁免, 经代理转发的请求也一律走鉴权分支。
    """
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        return False
    has_forward_header = bool(
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-forwarded-for")
        or request.headers.get("x-real-ip")
    )
    return not has_forward_header


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
