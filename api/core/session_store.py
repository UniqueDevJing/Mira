"""多轮对话会话存储 — 服务端按 session_id 维护历史, 刷新/换设备(同浏览器)不丢上下文。

复用 shared_state 后端 (InMemory 默认 / Redis), 与 QA 缓存/限流共享同一可插拔抽象。
历史以 JSON 列表持久化, TTL 惰性过期; 仅保留最近 _MAX_TURNS 条消息, 被截掉的部分累计为
`dropped` —— 供前端提示「更早的对话已超出上下文窗口」, 而不是静默丢弃 (静默截断会让用户
以为模型"忘了", 却看不到任何解释)。

存储格式: {"owner": str|None, "turns": [{role, content}, ...], "dropped": int}
兼容旧格式: [turn, ...] 与 {"owner":..., "turns": [...]}。
"""

from __future__ import annotations

import json

from api.config import settings
from api.core.shared_state import CacheBackend, get_cache_backend
from api.schemas.qa import ChatTurn

_MAX_TURNS = 20  # 保留的消息条数 (user + assistant 合计)
MAX_TURNS = _MAX_TURNS  # 公开别名: 调用方(编排层)据此把注入上下文裁剪到与持久化一致的窗口
_KEY_PREFIX = "rag:session:"


def _ttl_s() -> int:
    """会话存活时长 (秒)。下限 60s, 防误配成 0 导致会话永不生效。"""
    return max(60, int(getattr(settings, "session_ttl_s", 7200) or 7200))


def _key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


def _normalize(turn) -> dict | None:
    """统一 ChatTurn / dict 为 {role, content}。"""
    if isinstance(turn, ChatTurn):
        return {"role": turn.role, "content": turn.content}
    if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and turn.get("content"):
        return {"role": turn["role"], "content": turn["content"]}
    return None


def normalize_turns(turns) -> list[dict]:
    """批量规范化, 丢弃无法识别的元素。供会话检索与请求历史回退共用。"""
    out: list[dict] = []
    for t in turns or []:
        n = _normalize(t)
        if n:
            out.append(n)
    return out


def _read_raw(session_id: str, be: CacheBackend) -> dict:
    """读取并解析存储载荷; 不存在/损坏返回空结构。"""
    raw = be.get(_key(session_id)) if session_id else None
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    # 旧格式: 裸列表
    return {"turns": data}


def load_session(session_id: str, backend: CacheBackend | None = None) -> list[ChatTurn]:
    """读取会话历史; 不存在/损坏返回空列表。"""
    if not session_id:
        return []
    data = _read_raw(session_id, backend or get_cache_backend())
    return [ChatTurn(role=n["role"], content=n["content"]) for n in normalize_turns(data.get("turns"))]


def session_meta(session_id: str, backend: CacheBackend | None = None) -> dict:
    """会话元信息: {"turns": 现存条数, "dropped": 累计被截掉的条数}。不存在返回全 0。"""
    if not session_id:
        return {"turns": 0, "dropped": 0}
    data = _read_raw(session_id, backend or get_cache_backend())
    turns = normalize_turns(data.get("turns"))
    try:
        dropped = int(data.get("dropped") or 0)
    except (TypeError, ValueError):
        dropped = 0
    return {"turns": len(turns), "dropped": max(0, dropped)}


def session_owner(session_id: str, backend: CacheBackend | None = None) -> str | None:
    """读取会话归属者 key_id (S6 IDOR 防护用); 旧格式/不存在返回 None (视为无归属)。"""
    if not session_id:
        return None
    return _read_raw(session_id, backend or get_cache_backend()).get("owner") or None


def save_session(
    session_id: str,
    turns: list,
    backend: CacheBackend | None = None,
    owner: str | None = None,
) -> int:
    """写入会话历史 (仅保留最近 _MAX_TURNS 条), 返回累计被截掉的条数。

    owner: 创建者 key_id (S6 归属绑定); None = 未绑定 (旧调用方, 保持向后兼容)。
    """
    if not session_id:
        return 0
    be = backend or get_cache_backend()
    normalized = normalize_turns(turns)
    overflow = max(0, len(normalized) - _MAX_TURNS)
    trimmed = normalized[-_MAX_TURNS:]
    if not trimmed:
        be.delete(_key(session_id))
        return 0
    # 累计截断数: 会话生命周期内被挤出窗口的总条数, 供前端如实提示
    dropped = session_meta(session_id, be)["dropped"] + overflow
    payload = json.dumps({"owner": owner, "turns": trimmed, "dropped": dropped}, ensure_ascii=False)
    be.set(_key(session_id), payload, _ttl_s())
    return dropped


def clear_session(session_id: str, backend: CacheBackend | None = None) -> None:
    if not session_id:
        return
    (backend or get_cache_backend()).delete(_key(session_id))
