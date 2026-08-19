"""跨进程共享态后端测试 — 内存实现 + Redis 回退 + QACache 接线 (不依赖 Redis 服务)。"""

import time

from api.config import settings
from api.core import shared_state
from api.core.qa_cache import QACache
from api.core.shared_state import InMemoryBackend, get_cache_backend


def _reset_backend(monkeypatch):
    """重置后端单例, 使 get_cache_backend 重新按 settings 选择。"""
    monkeypatch.setattr(shared_state, "_backend", None)


def test_inmemory_set_get_delete():
    b = InMemoryBackend()
    b.set("k", "v", ttl_s=60)
    assert b.get("k") == "v"
    b.delete("k")
    assert b.get("k") is None


def test_inmemory_ttl_expiry():
    b = InMemoryBackend()
    b.set("k", "v", ttl_s=1)
    assert b.get("k") == "v"
    time.sleep(1.1)
    assert b.get("k") is None  # 过期惰性清除


def test_inmemory_clear():
    b = InMemoryBackend()
    b.set("a", "1", ttl_s=60)
    b.set("b", "2", ttl_s=60)
    b.clear()
    assert b.get("a") is None
    assert b.get("b") is None


def test_inmemory_lru_evicts_least_recently_used():
    """LRU: 容量满时逐出最久未访问项, 而非最旧插入项。

    覆盖共享 4096 上限下活跃会话被插入序提前淘汰的隐患。
    """
    b = InMemoryBackend(max_entries=2)
    b.set("a", "1", ttl_s=600)
    b.set("b", "2", ttl_s=600)
    # 访问 a → a 变为最近使用, b 成为最久未访问
    assert b.get("a") == "1"
    b.set("c", "3", ttl_s=600)  # 满 → 应逐出 b
    assert b.get("b") is None  # 最久未访问被逐
    assert b.get("a") == "1"  # 活跃项保留
    assert b.get("c") == "3"


def test_default_backend_is_inmemory(monkeypatch):
    _reset_backend(monkeypatch)
    assert isinstance(get_cache_backend(), InMemoryBackend)


def test_redis_unavailable_falls_back_to_memory(monkeypatch):
    """配置 redis 但 redis 不可达 → 启动探活失败回退内存, 不阻断启动。"""
    _reset_backend(monkeypatch)
    monkeypatch.setattr(settings, "shared_state_backend", "redis")
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6399")  # 无服务
    backend = get_cache_backend()
    assert isinstance(backend, InMemoryBackend)  # 探活失败 → 回退内存


def test_qa_cache_via_backend_roundtrip(monkeypatch):
    cache = QACache(backend=InMemoryBackend())
    key = QACache.make_key("什么是退款", "service", 5, False, 0.0, "hybrid")
    cache.set(key, {"answer": "买家申请退款", "sources": []}, ttl_s=60)
    hit = cache.get(key)
    assert hit == {"answer": "买家申请退款", "sources": []}
    cache.clear()
    assert cache.get(key) is None


def test_qa_cache_expired(monkeypatch):
    cache = QACache(backend=InMemoryBackend())
    key = QACache.make_key("Q", "tech", 5, False, 0.0)
    cache.set(key, {"answer": "x"}, ttl_s=1)
    time.sleep(1.1)
    assert cache.get(key) is None
