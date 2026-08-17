"""鉴权与身份 API"""

from fastapi import APIRouter, Request

from api.core.auth import get_principal

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.get("/me")
async def whoami(request: Request):
    """返回当前请求主体的身份与权限范围, 供管理后台展示。"""
    p = get_principal(request)
    return {
        "key_id": p.key_id,
        "name": p.name,
        "role": p.role,
        "allowed_kbs": p.allowed_kbs,
        "is_admin": p.is_admin(),
    }
