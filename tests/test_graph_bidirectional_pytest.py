"""GraphRAG 双向多跳 + find_node O(1) 索引 (C2/C4) 回归测试"""

from engines.graph_rag.graph_store import GraphStore


def _build():
    g = GraphStore()
    g.upsert_entity("FastAPI", "framework", "c1")
    g.upsert_entity("uvicorn", "server", "c1")
    g.upsert_entity("缓存", "concept", "c2", aliases=["Cache", "CACHE"])
    g.add_relation("FastAPI", "uses", "uvicorn", "c1")
    g.add_relation("FastAPI", "contains", "缓存", "c2")
    return g


def test_bidirectional_reachability():
    """从宾语侧 (uvicorn) 应能经入边反向到达 FastAPI (C2)"""
    g = _build()
    hops = g.multi_hop("uvicorn", relations=["uses", "contains", "depends_on", "owns"])
    targets = {h["from"] if h["to"] == "uvicorn" else h["to"] for h in hops}
    assert "FastAPI" in targets
    # 语义方向保留: 入边 from=FastAPI, to=uvicorn
    assert any(h["from"] == "FastAPI" and h["to"] == "uvicorn" for h in hops)


def test_unidirectional_off_still_works():
    g = _build()
    # 关闭双向: 从 FastAPI 出发仍能沿出边到达 uvicorn
    hops = g.multi_hop("FastAPI", relations=["uses"], bidirectional=False)
    assert any(h["to"] == "uvicorn" for h in hops)


def test_find_node_case_insensitive_and_alias():
    """大小写不敏感 + 别名反查 (C4 索引路径)"""
    g = _build()
    assert g.find_node("fastapi")[0] == "FastAPI"
    assert g.find_node("CACHE")[0] == "缓存"
    assert g.find_node("Cache")[0] == "缓存"
    assert g.find_node("不存在") == (None, None)


def test_lower_index_o1_no_linear_fallback_needed():
    """大批量小写反查走索引, 结果正确 (不依赖线性扫描)"""
    g = GraphStore()
    for i in range(200):
        g.upsert_entity(f"Entity{i}", "t", f"c{i}", aliases=[f"Alias{i}"])
    assert g.find_node("entity150")[0] == "Entity150"
    assert g.find_node("alias50")[0] == "Entity50"
