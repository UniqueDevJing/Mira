"""GraphStore Redis 共享后端测试。

不依赖真实 Redis 服务:
- 用进程内 FakeRedis 验证整图 pickle 双写/恢复路径 (_save / _load_redis)。
- 用不可达的 redis_url 验证 Redis 不可用时优雅回退 (不抛异常, 内存态仍可用)。
"""

from engines.graph_rag.graph_store import GraphStore


class FakeRedis:
    """最小化 redis-py 兼容桩: 仅实现 GraphStore 用到的 set/get (二进制)。"""

    def __init__(self):
        self._data: dict[str, bytes] = {}

    def set(self, name, value, **kwargs):
        self._data[name] = value
        return True

    def get(self, name):
        return self._data.get(name)


def _build_sample(gs: GraphStore) -> None:
    gs.upsert_entity("FastAPI", "framework", "c1", aliases=["fastapi"])
    gs.upsert_entity("Starlette", "framework", "c1")
    gs.add_relation("FastAPI", "uses", "Starlette", "c1")


def test_redis_roundtrip_with_fake_backend():
    """首个 worker 建图写入 Redis, 第二个 worker 从 Redis 恢复, 图谱一致。"""
    fake = FakeRedis()
    gs = GraphStore()  # redis_url=None → _redis 不初始化
    gs._redis = fake
    gs._redis_key = "rag:graph:test"
    _build_sample(gs)
    gs._save()  # 双写: persist_path=None 跳过文件, 写 Redis

    gs2 = GraphStore()
    gs2._redis = fake
    gs2._redis_key = "rag:graph:test"
    assert gs2._load_redis() is True

    assert "FastAPI" in gs2.nodes
    assert gs2.nodes["FastAPI"]["aliases"] == ["fastapi"]
    assert len(gs2.edges) == 1
    assert gs2.edges[0]["object"] == "Starlette"
    # 小写索引在恢复后可用 (O(1) 反查)
    canon, _ = gs2.find_node("fastapi")
    assert canon == "FastAPI"


def test_redis_unavailable_falls_back_gracefully():
    """redis_url 指向不可达服务: 初始化/落盘/恢复均不抛异常, 内存态可用。"""
    gs = GraphStore(persist_path=None, redis_url="redis://127.0.0.1:6399")
    # _load_redis 连接失败被吞 → 回退空图, 不抛
    _build_sample(gs)
    assert "FastAPI" in gs.nodes  # 内存态仍正常
    # _save 的 Redis set 失败被吞
    gs._save()
    assert len(gs.edges) == 1


def test_redis_key_default_from_persist_path():
    """未显式给 redis_key 时, 回退 persist_path 的 basename, 避免多 KB 串图。"""
    gs = GraphStore(persist_path="/tmp/x/graph_kb1.pkl")
    assert gs._redis_key == "graph_kb1.pkl"
