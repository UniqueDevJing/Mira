"""跨进程共享状态后端 — 可插拔, 默认内存, 可切 Redis。

解决多 worker (gunicorn -k uvicorn.workers.UvicornWorker -w N) 部署下, QA 缓存/限流各自
进程独立导致共享态失效的问题:
- InMemoryBackend: 默认, 零依赖, TTL 支持, 线程安全 (仅单进程有效)
- RedisBackend: shared_state_backend=redis 且 redis_url 非空时启用, 懒加载 redis 客户端
  (未安装 redis 或连接失败 → 自动回退 InMemoryBackend, 不阻断启动)
- get_cache_backend(): 全局单例工厂, 按 settings 选择后端

QACache 与限流器复用此后端实现多 worker 共享态。
"""

import json
import threading
import time
from typing import Protocol

from api.config import settings


class CacheBackend(Protocol):
    """共享缓存后端接口 (值以 JSON 字符串存储, TTL 由后端控制)。"""

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_s: int) -> None: ...

    def delete(self, key: str) -> None: ...


class InMemoryBackend:
    """内存后端 — 默认实现。TTL 惰性过期, 容量上限防无限增长, LRU 淘汰保活跃项。

    LRU 语义: get/set 命中既存 key 时将其移到末尾 (最近使用), 满则逐出最久未访问项。
    否则频繁命中的活跃会话/QA 缓存会被插入序淘汰提前清掉 (共享 4096 上限下尤甚)。
    """

    def __init__(self, max_entries: int = 4096):
        self._data: dict[str, tuple[str, float]] = {}  # key -> (value, expires_at)
        self._lock = threading.Lock()
        self._max = max_entries

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            val, exp = item
            if exp < time.monotonic():
                del self._data[key]
                return None
            # 命中且未过期 → 移到末尾标记最近使用 (LRU)
            self._data[key] = self._data.pop(key)
            return val

    def set(self, key: str, value: str, ttl_s: int) -> None:
        with self._lock:
            if len(self._data) >= self._max:
                self._evict_expired_locked()
                if len(self._data) >= self._max:
                    self._data.pop(next(iter(self._data)))  # 满且无过期项 → 丢最久未访问
            # 既存 key 重写 → 移到末尾标记最近使用 (LRU)
            self._data.pop(key, None)
            self._data[key] = (value, time.monotonic() + ttl_s)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        for k in [k for k, (_, e) in self._data.items() if e < now]:
            del self._data[k]


class RedisBackend:
    """Redis 后端 — 值存 JSON 字符串, TTL 由 Redis 原生 EX 控制。

    懒加载 redis 客户端 (仅构造时 import), 未安装 redis 或不可达时由 get_cache_backend 回退内存。
    """

    def __init__(self, redis_url: str, key_prefix: str = "rag:"):
        import redis

        self._r = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=5, retry_on_timeout=True)
        self._prefix = key_prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> str | None:
        return self._r.get(self._k(key))

    def set(self, key: str, value: str, ttl_s: int) -> None:
        self._r.set(self._k(key), value, ex=max(1, int(ttl_s)))

    def delete(self, key: str) -> None:
        self._r.delete(self._k(key))

    def clear(self) -> None:
        # 跨进程清空本前缀键 (SCAN 非阻塞, 避免 KEYS 阻塞 Redis)
        cursor = 0
        while True:
            cursor, keys = self._r.scan(cursor, match=f"{self._prefix}*", count=200)
            if keys:
                self._r.delete(*keys)
            if cursor == 0:
                break


_backend: CacheBackend | None = None
_backend_lock = threading.Lock()


def get_cache_backend() -> CacheBackend:
    """全局共享缓存后端单例 — 按 settings 选择, 进程内复用同一实例。"""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                if settings.shared_state_backend == "redis" and settings.redis_url:
                    try:
                        backend = RedisBackend(settings.redis_url)
                        backend._r.ping()  # 启动探活: 不可达 → 回退内存, 避免运行期每请求 500
                        _backend = backend
                    except Exception:  # noqa: BLE001 — Redis 不可用/不可达回退内存, 不阻断启动
                        _backend = InMemoryBackend()
                else:
                    _backend = InMemoryBackend()
    return _backend


def backend_serialize(value: dict) -> str:
    """缓存值序列化 (dict → JSON 字符串, 供后端存储)。"""
    return json.dumps(value, ensure_ascii=False)


def backend_deserialize(raw: str | None) -> dict | None:
    """缓存值反序列化 (JSON 字符串 → dict); 损坏/类型错误返回 None。"""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
