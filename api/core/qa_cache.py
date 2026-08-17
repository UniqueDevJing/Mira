"""QA 结果缓存 — 相同问题命中直接返回, 跳过路由+检索+LLM。

存储后端可插拔 (api/core/shared_state.py): 默认内存 (单进程), 配置 Redis 后跨 worker 共享。
缓存值以 JSON 字符串存后端; 本类负责 dict <-> 字符串的序列化与 TTL 语义。
"""

import hashlib
import json
import threading

from api.core.shared_state import backend_deserialize, backend_serialize, get_cache_backend


class QACache:
    def __init__(self, backend=None):
        # backend=None → 默认全局共享后端 (内存/Redis 按 config 选择)
        self._backend = backend if backend is not None else get_cache_backend()

    @staticmethod
    def make_key(
        question: str,
        skill: str | None,
        top_k: int,
        enable_self_retrieval: bool,
        temperature: float,
        mode: str = "hybrid",
        history=None,
    ) -> str:
        """缓存键: 完整输入指纹。temperature/mode/history 计入, 尊重用户参数, 代价是缓存分片。"""
        hist = [(getattr(t, "role", None), getattr(t, "content", None)) if not isinstance(t, dict) else (t.get("role"), t.get("content")) for t in (history or [])]
        raw = json.dumps(
            [question, skill, top_k, enable_self_retrieval, temperature, mode, hist], ensure_ascii=False, sort_keys=True
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        return backend_deserialize(self._backend.get(key))

    def set(self, key: str, value: dict, ttl_s: int) -> None:
        self._backend.set(key, backend_serialize(value), ttl_s)

    def delete(self, key: str) -> None:
        self._backend.delete(key)

    def clear(self) -> None:
        # 后端支持 clear 时清空 (内存后端本地清; Redis 后端跨进程清本前缀)。测试默认走内存后端。
        clr = getattr(self._backend, "clear", None)
        if clr is not None:
            clr()


_cache_instance: QACache | None = None
_cache_lock = threading.Lock()


def get_qa_cache() -> QACache:
    """全局单例 (与 api/state.py 的单例模式一致)。"""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = QACache()
    return _cache_instance
