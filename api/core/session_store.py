"""多轮对话会话存储 — 服务端按 session_id 维护历史, 刷新/换设备(同浏览器)不丢上下文。

复用 shared_state 后端 (InMemory 默认 / Redis-ready), 与 QA 缓存/限流共享同一可插拔抽象。
历史以 JSON 列表持久化, TTL 惰性过期; 仅存最近 20 轮。
"""

from __future__ import annotations

import json

from api.core.shared_state import CacheBackend, get_cache_backend
from api.schemas.qa import ChatTurn

SESSION_TTL_S = 1800  # 30 分钟无活动则过期
_MAX_TURNS = 20
_KEY_PREFIX = "rag:session:"


def _key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


def _normalize(turn) -> dict | None:
    """统一 ChatTurn / dict 为 {role, content}。"""
    if isinstance(turn, ChatTurn):
        return {"role": turn.role, "content": turn.content}
    if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and turn.get("content"):
        return {"role": turn["role"], "content": turn["content"]}
    return None


def load_session(session_id: str, backend: CacheBackend | None = None) -> list[ChatTurn]:
    """读取会话历史; 不存在/损坏返回空列表。

    兼容两种存储格式: 新格式 {"owner": str|None, "turns": [...]} 与旧格式 [turn, ...]。
    """
    if not session_id:
        return []
    be = backend or get_cache_backend()
    raw = be.get(_key(session_id))
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    turns_data = data.get("turns") if isinstance(data, dict) else data
    turns = []
    for t in turns_data or []:
        n = _normalize(t)
        if n:
            turns.append(ChatTurn(role=n["role"], content=n["content"]))
    return turns


def session_owner(session_id: str, backend: CacheBackend | None = None) -> str | None:
    """读取会话归属者 key_id (S6 IDOR 防护用); 旧格式/不存在返回 None (视为无归属)。"""
    if not session_id:
        return None
    be = backend or get_cache_backend()
    raw = be.get(_key(session_id))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return data.get("owner") or None
    return None


def save_session(session_id: str, turns: list, backend: CacheBackend | None = None, owner: str | None = None) -> None:
    """写入会话历史(截断最近 20 轮)。turns 元素可为 ChatTurn 或 {role,content}。

    owner: 创建者 key_id (S6 归属绑定); None = 未绑定 (旧调用方, 保持向后兼容)。
    """
    if not session_id:
        return
    be = backend or get_cache_backend()
    trimmed = [_normalize(t) for t in (turns or [])]
    trimmed = [t for t in trimmed if t][-_MAX_TURNS:]
    if not trimmed:
        be.delete(_key(session_id))
        return
    payload = json.dumps({"owner": owner, "turns": trimmed}, ensure_ascii=False)
    be.set(_key(session_id), payload, SESSION_TTL_S)


def clear_session(session_id: str, backend: CacheBackend | None = None) -> None:
    if not session_id:
        return
    (backend or get_cache_backend()).delete(_key(session_id))
