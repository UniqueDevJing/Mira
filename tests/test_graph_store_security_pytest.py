"""GraphStore 持久化安全回归测试 — 杜绝 pickle 反序列化 RCE + HMAC 完整性。

防护目标 (对应企业级质检 P1#5):
- Redis/文件中的不可信字节绝不被 unpickle (旧 pickle.loads 是 RCE 入口)。
- 内容经 HMAC-SHA256 签名, 篡改/错密钥的 blob 必须拒绝 (回退空图/文件), 不静默加载。
"""

import json

from engines.graph_rag.graph_store import _GRAPH_VERSION, GraphStore


class FakeRedis:
    """最小化 redis-py 兼容桩: 仅实现 GraphStore 用到的 set/get (二进制)。"""

    def __init__(self, store=None):
        self._d = store or {}

    def set(self, name, value, **kwargs):
        self._d[name] = value
        return True

    def get(self, name):
        return self._d.get(name)


def _make_store(redis=None, key="rk"):
    gs = GraphStore()
    if redis is not None:
        gs._redis = redis
        gs._redis_key = key
    return gs


def test_untrusted_pickle_in_redis_fails_closed(monkeypatch):
    """Redis 被写入恶意 pickle: 旧代码 pickle.loads 会执行 __reduce__ 触发 RCE; 现在必须拒绝且绝不加载。"""
    import pickle

    monkeypatch.delenv("RAG_GRAPH_HMAC_SECRET", raising=False)

    class _Evil:
        def __reduce__(self):
            # 若被 unpickle 会执行命令; 测试靠它不被触发来证明 RCE 已消除
            return (exec, ("raise SystemExit('PWNED_RCE_EXECUTED')",))

    evil_blob = pickle.dumps(
        {
            "version": _GRAPH_VERSION,
            "nodes": {"x": {"type": "t", "aliases": [], "chunks": [], "properties": {}}},
            "edges": [],
            "adj_out": {},
            "adj_in": {},
            "edge_keys": [],
            "lower_index": {},
        }
    )
    fake = FakeRedis({"rag:graph:evil": evil_blob})
    gs = _make_store(redis=fake, key="rag:graph:evil")
    ok = gs._load_redis()
    assert ok is False, "不可信 pickle 不应被加载"
    assert "x" not in gs.nodes, "恶意 pickle 内容绝不应进入节点表 (RCE 防护失效)"


def test_tampered_hmac_rejected(monkeypatch):
    """有效 JSON + 错误 HMAC 签名: 必须验签失败并回退, 不加载。"""
    monkeypatch.delenv("RAG_GRAPH_HMAC_SECRET", raising=False)

    body = json.dumps(
        {"version": _GRAPH_VERSION, "nodes": {}, "edges": [], "adj_out": {}, "adj_in": {}, "edge_keys": [], "lower_index": {}}
    ).encode()
    tampered = b"deadbeef" + b"|" + body
    fake = FakeRedis({"k": tampered})
    gs = _make_store(redis=fake, key="k")
    assert gs._load_redis() is False


def test_hmac_sealed_roundtrip_succeeds(monkeypatch):
    """合法 HMAC 信封: 写入后另一实例能完整恢复图谱 (节点/边/小写索引)。"""
    monkeypatch.delenv("RAG_GRAPH_HMAC_SECRET", raising=False)

    redis = FakeRedis()
    w = _make_store(redis=redis, key="rk")
    w.upsert_entity("FastAPI", "fw", "c1", aliases=["fastapi"])
    w.add_relation("FastAPI", "uses", "Starlette", "c1")
    w._save()

    r = _make_store(redis=redis, key="rk")
    assert r._load_redis() is True
    assert "FastAPI" in r.nodes
    assert r.nodes["FastAPI"]["aliases"] == ["fastapi"]
    assert len(r.edges) == 1
    assert r.edges[0]["object"] == "Starlette"
    assert r.find_node("fastapi")[0] == "FastAPI"  # 小写索引恢复


def test_cross_instance_wrong_key_rejected(monkeypatch):
    """多 worker 共享图谱: 写/读密钥不一致必须拒绝 (强制生产配置同一 RAG_GRAPH_HMAC_SECRET)。"""
    monkeypatch.delenv("RAG_GRAPH_HMAC_SECRET", raising=False)
    import engines.graph_rag.graph_store as gs_mod

    redis = FakeRedis()
    # 写方用密钥 A
    monkeypatch.setattr(gs_mod, "_hmac_key", lambda: b"worker-A-secret")
    w = _make_store(redis=redis, key="rk5")
    w.upsert_entity("A", "t", "c")
    w._save()
    # 读方用密钥 B —— 必须验签失败
    monkeypatch.setattr(gs_mod, "_hmac_key", lambda: b"worker-B-secret")
    r = _make_store(redis=redis, key="rk5")
    assert r._load_redis() is False


def test_corrupted_file_quarantined_as_corrupt(monkeypatch, tmp_path):
    """损坏的落盘文件 (非 JSON 信封) 必须隔离为 .corrupt 并回退空图, 不抛异常。"""
    monkeypatch.delenv("RAG_GRAPH_HMAC_SECRET", raising=False)

    p = tmp_path / "graph.pkl"
    p.write_bytes(b"\x80\x02corrupted-pickle-bytes")
    gs = GraphStore(persist_path=str(p))
    assert gs.stats()["nodes"] == 0
    assert (tmp_path / "graph.pkl.corrupt").exists(), "损坏文件应被隔离为 .corrupt"
