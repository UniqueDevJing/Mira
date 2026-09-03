"""QA 结果缓存 — 相同问题命中直接返回, 跳过路由+检索+LLM。

存储后端可插拔 (api/core/shared_state.py): 默认内存 (单进程), 配置 Redis 后跨 worker 共享。
缓存值以 JSON 字符串存后端; 本类负责 dict <-> 字符串的序列化与 TTL 语义。

语义近重复命中 (qa_cache_semantic_enabled):
  在精确哈希之上, 对"同参数作用域"内的近义/改写问题做 cosine 命中, 省 LLM+rerank 调用。
  - 作用域 (scope) = 除 question 外所有输入指纹, 防止跨 temperature/mode/history/RBAC 误命中。
  - 阈值偏高 (默认 0.92) 仅命中近重复, 防误答。
  - 语义索引为 in-process: 内存后端完整生效; Redis 后端下仍为 best-effort (精确命中照常跨 worker)。
"""

import copy
import hashlib
import json
import threading
import time

from api.config import settings
from api.core.shared_state import backend_deserialize, backend_serialize, get_cache_backend


class QACache:
    def __init__(self, backend=None, embed_fn=None, semantic_enabled: bool | None = None):
        # backend=None → 默认全局共享后端 (内存/Redis 按 config 选择)
        self._backend = backend if backend is not None else get_cache_backend()
        # embed_fn(question:str)->list[float] | None。None 时生产环境惰性取 get_embedder().embed_query
        self._embed_fn = embed_fn
        self._semantic = settings.qa_cache_semantic_enabled if semantic_enabled is None else semantic_enabled
        self._sem_threshold = settings.qa_cache_semantic_threshold
        # 语义索引: scope -> {cache_key: (embedding, expiry_ts, value)}; 仅 in-process 生效
        self._sem_index: dict[str, dict[str, tuple[list[float], float, dict]]] = {}
        self._sem_order: list[tuple[str, str]] = []  # (scope, key) 插入序, 用于超额驱逐
        self._sem_cap = 2000
        self._sem_lock = threading.Lock()

    @staticmethod
    def make_key(
        question: str,
        skill: str | None,
        top_k: int,
        enable_self_retrieval: bool,
        temperature: float,
        mode: str = "hybrid",
        history=None,
        allowed_kbs=None,
    ) -> str:
        """缓存键: 完整输入指纹。temperature/mode/history 计入, 尊重用户参数, 代价是缓存分片。

        allowed_kbs 计入键 (RBAC 作用域分片): None=全部(admin)、[]=无权限、[...]=受限子集。
        防止受限 reader 命中 admin 权限下生成的含越权 KB 内容的缓存 (S1 越权读修复)。
        """
        hist = [
            (getattr(t, "role", None), getattr(t, "content", None))
            if not isinstance(t, dict)
            else (t.get("role"), t.get("content"))
            for t in (history or [])
        ]
        scope = sorted(allowed_kbs) if allowed_kbs is not None else None
        raw = json.dumps(
            [question, skill, top_k, enable_self_retrieval, temperature, mode, hist, scope],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def make_scope(
        question: str,
        skill: str | None,
        top_k: int,
        enable_self_retrieval: bool,
        temperature: float,
        mode: str = "hybrid",
        history=None,
        allowed_kbs=None,
    ) -> str:
        """语义命中作用域: 输入指纹中排除 question 文本, 使同参数下的不同问法落入同一桶。

        与 make_key 完全相同, 仅 question 固定为空串 → 相同参数集得相同 scope。
        """
        return QACache.make_key(
            "", skill, top_k, enable_self_retrieval, temperature, mode, history, allowed_kbs
        )

    # ────────────────────────── 语义嵌入 ──────────────────────────

    def _sem_embed(self, question: str) -> list[float] | None:
        """取问题 embedding (归一化)。无 embed_fn 且无法惰性取模型 → 返回 None (禁用语义)。"""
        if self._embed_fn is not None:
            try:
                return self._embed_fn(question)
            except Exception:  # noqa: BLE001 — 嵌入失败则退化为仅精确命中
                return None
        try:
            from api.state import get_embedder

            return get_embedder().embed_query(question)
        except Exception:  # noqa: BLE001 — 模型不可用 → 仅精确命中
            return None

    # ────────────────────────── 精确 / 语义 查询 ──────────────────────────

    def get(
        self, key: str, question: str | None = None, scope: str | None = None, stats: dict | None = None
    ) -> dict | None:
        """命中返回值; 传入 stats 时回填命中类型供指标上报。

        stats 回填: {"kind": "exact"|"semantic", "sim": float}
        semantic 时 sim 为近重复余弦相似度(用于观测阈值松紧是否合适)。
        """
        exact = backend_deserialize(self._backend.get(key))
        if exact is not None:
            if stats is not None:
                stats["kind"] = "exact"
            return copy.deepcopy(exact)
        # 精确未命中 → 尝试同作用域近重复语义命中
        if not self._semantic or not question or scope is None:
            return None
        emb = self._sem_embed(question)
        if emb is None:
            return None
        with self._sem_lock:
            bucket = self._sem_index.get(scope)
            if not bucket:
                return None
            now = time.time()
            best_key, best_val, best_sim = None, None, -1.0
            expired = []
            for k, (e, exp, val) in bucket.items():
                if exp < now:
                    expired.append(k)
                    continue
                sim = sum(a * b for a, b in zip(emb, e))  # 已归一化 → 点积即 cosine
                if sim > best_sim:
                    best_sim = sim
                    best_key, best_val = k, val
            for k in expired:
                bucket.pop(k, None)
            if best_key is not None and best_sim >= self._sem_threshold:
                if stats is not None:
                    stats["kind"] = "semantic"
                    stats["sim"] = best_sim
                return copy.deepcopy(best_val)
        return None

    def set(self, key: str, value: dict, ttl_s: int, question: str | None = None, scope: str | None = None) -> None:
        self._backend.set(key, backend_serialize(value), ttl_s)
        # 同作用域近重复语义索引 (best-effort, in-process)
        if not self._semantic or not question or scope is None:
            return
        emb = self._sem_embed(question)
        if emb is None:
            return
        with self._sem_lock:
            bucket = self._sem_index.setdefault(scope, {})
            if key not in bucket:
                self._sem_order.append((scope, key))
            bucket[key] = (emb, time.time() + ttl_s, copy.deepcopy(value))
            # 超额驱逐最旧一条 (跨作用域)
            while len(self._sem_order) > self._sem_cap:
                old_scope, old_key = self._sem_order.pop(0)
                ob = self._sem_index.get(old_scope)
                if ob and old_key in ob:
                    del ob[old_key]
                if not ob:
                    self._sem_index.pop(old_scope, None)

    def delete(self, key: str) -> None:
        self._backend.delete(key)
        with self._sem_lock:
            for bucket in self._sem_index.values():
                bucket.pop(key, None)

    def clear(self) -> None:
        # 后端支持 clear 时清空 (内存后端本地清; Redis 后端跨进程清本前缀)。测试默认走内存后端。
        clr = getattr(self._backend, "clear", None)
        if clr is not None:
            clr()
        with self._sem_lock:
            self._sem_index.clear()
            self._sem_order.clear()


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
