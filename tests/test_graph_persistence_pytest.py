"""知识图谱持久化测试 — pickle 落盘/重载/容错 (不依赖 LLM/模型)。"""

from engines.graph_rag.graph_store import GraphStore


def _populate(gs: GraphStore) -> None:
    gs.upsert_entity("FastAPI", "Technology", "c1", aliases=["Fast API"])
    gs.upsert_entity("Python", "Technology", "c1")
    gs.upsert_entity("Uvicorn", "Technology", "c2")
    gs.add_relation("FastAPI", "uses", "Python", "c1")
    gs.add_relation("FastAPI", "depends_on", "Uvicorn", "c2")
    gs.save()


def test_save_then_reload_restores_graph(tmp_path):
    p = tmp_path / "graph.pkl"
    gs = GraphStore(persist_path=str(p))
    _populate(gs)
    assert p.exists()

    # 新实例从磁盘恢复
    gs2 = GraphStore(persist_path=str(p))
    assert gs2.stats()["nodes"] == 3
    assert gs2.stats()["edges"] == 2
    # 多跳遍历仍可工作
    hops = gs2.multi_hop("FastAPI", ["uses", "depends_on"])
    assert len(hops) == 2
    # 入边双向遍历也还原: 邻节点在 from 字段, 起点 Python 在 to 字段
    back = gs2.multi_hop("Python", ["uses"], bidirectional=True)
    assert any(h["from"] == "FastAPI" for h in back)


def test_aliases_and_lower_index_survive_round_trip(tmp_path):
    p = tmp_path / "graph.pkl"
    gs = GraphStore(persist_path=str(p))
    gs.upsert_entity("FastAPI", "Technology", "c1", aliases=["Fast API"])
    gs.save()

    gs2 = GraphStore(persist_path=str(p))
    # 大小写不敏感反查 (O(1) 索引还原)
    canon, _ = gs2.find_node("fastapi")
    assert canon == "FastAPI"
    # 别名反查
    canon2, _ = gs2.find_node("Fast API")
    assert canon2 == "FastAPI"


def test_no_persist_path_does_not_write_file(tmp_path):
    p = tmp_path / "absent" / "graph.pkl"  # 父目录不存在, 若误写会报错
    gs = GraphStore(persist_path=None)  # 纯内存
    gs.upsert_entity("X", "T", "c1")
    gs.save()  # 应为空操作, 不创建任何文件
    assert not p.exists()


def test_corrupted_pickle_falls_back_to_empty(tmp_path):
    p = tmp_path / "graph.pkl"
    p.write_bytes(b"\x80\x02corrupted-pickle-bytes")  # 非法 pickle
    gs = GraphStore(persist_path=str(p))  # 不应抛异常
    assert gs.stats()["nodes"] == 0
    assert gs.stats()["edges"] == 0
    # 仍可正常写入新数据
    gs.upsert_entity("After", "T", "c1")
    gs.save()
    gs2 = GraphStore(persist_path=str(p))
    assert "After" in gs2.nodes


def test_default_init_is_in_memory():
    gs = GraphStore()
    assert gs._persist_path is None
