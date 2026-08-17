"""API Key 白名单 + 基于知识库(KB)的 RBAC 主体抽象。

设计要点:
- 多 Key 白名单: 每个 key 映射到一组可访问的知识库 (allowed_kbs) 与角色 (role)。
- allowed_kbs=None 表示「全部知识库」(admin 语义); [] 表示无权访问任何库; ["a","b"] 受限。
- Principal 经安全中间件注入 request.state, 各路由用 get_principal 读取并做 KB 级拦截。
- 兼容旧版单 Key (api_key): 视为 admin, 可访问全部知识库。
"""

import json
import os
import secrets
from dataclasses import dataclass

from fastapi import Request

from api.config import settings

ADMIN = "admin"


class KBForbiddenError(Exception):
    """路由命中了 principal 无权访问的知识库, 由路由层转换为 403。"""


@dataclass
class Principal:
    """请求主体: 身份 + 可访问知识库范围。"""

    key_id: str  # API Key 原文或 "loopback"/"anonymous"
    name: str
    role: str  # "admin" | "reader"
    allowed_kbs: list[str] | None  # None=全部; []=无; [...] = 受限子集

    def is_admin(self) -> bool:
        return self.role == ADMIN or self.allowed_kbs is None

    def can_access_kb(self, kb: str) -> bool:
        if self.allowed_kbs is None:
            return True
        return kb in self.allowed_kbs


def _parse_whitelist(raw: str) -> dict[str, dict]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"api_key_whitelist 不是合法 JSON: {e}") from e
    if not isinstance(data, dict):
        raise TypeError("api_key_whitelist 必须是对象 (key -> {name,kbs,role})")
    return data


def _principal_from_entry(key: str, entry: dict) -> Principal:
    kbs = entry.get("kbs")
    if kbs is None or kbs == "*" or kbs == ["*"]:
        allowed, role = None, ADMIN  # 全部知识库 = admin 语义
    else:
        allowed = [str(k) for k in kbs]
        role = entry.get("role", "reader")
    return Principal(key_id=key, name=entry.get("name", key[:8]), role=role, allowed_kbs=allowed)


def load_api_keys() -> dict[str, Principal]:
    """加载白名单为 key -> Principal 映射 (含 legacy 单 Key 作为 admin)。"""
    keys: dict[str, Principal] = {}
    wl = _parse_whitelist(os.environ.get("RAG_API_KEY_WHITELIST") or settings.api_key_whitelist)
    for k, v in wl.items():
        keys[k] = _principal_from_entry(k, v if isinstance(v, dict) else {"name": v})
    # 兼容旧版单 Key: 未列入白名单时视为 admin (全部知识库)
    legacy = os.environ.get("RAG_API_KEY") or settings.api_key
    if legacy and legacy not in keys:
        keys[legacy] = Principal(key_id=legacy, name="legacy-admin", role=ADMIN, allowed_kbs=None)
    return keys


_KEYS: dict[str, Principal] | None = None


def get_api_keys() -> dict[str, Principal]:
    global _KEYS
    if _KEYS is None:
        _KEYS = load_api_keys()
    return _KEYS


def authenticate(provided: str | None) -> Principal | None:
    """常量时间比对校验 Key; 命中返回对应 Principal, 否则 None。"""
    if not provided:
        return None
    for k, p in get_api_keys().items():
        if secrets.compare_digest(provided, k):
            return p
    return None


def loopback_principal() -> Principal:
    return Principal(key_id="loopback", name="loopback", role=ADMIN, allowed_kbs=None)


def anonymous_admin() -> Principal:
    return Principal(key_id="anonymous", name="anonymous", role=ADMIN, allowed_kbs=None)


def get_principal(request: Request) -> Principal:
    """路由依赖: 读取中间件注入的 principal, 缺失时回退 anonymous admin。"""
    p = getattr(request.state, "principal", None)
    return p if isinstance(p, Principal) else anonymous_admin()
