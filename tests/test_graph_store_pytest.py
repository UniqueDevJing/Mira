"""知识图谱存储测试 — 实体 / 关系 / 多跳遍历"""
import pytest
from engines.graph_rag.graph_store import GraphStore


@pytest.fixture
def graph():
    """空图谱"""
    return GraphStore()


@pytest.fixture
def populated_graph(graph):
    """预填充图谱"""
    # 添加实体
    graph.upsert_entity("FastAPI", "Technology", "chunk_001")
    graph.upsert_entity("Python", "Technology", "chunk_001")
    graph.upsert_entity("Uvicorn", "Technology", "chunk_002")
    graph.upsert_entity("Starlette", "Technology", "chunk_002")

    # 添加关系
    graph.add_relation("FastAPI", "uses", "Python", "chunk_001")
    graph.add_relation("FastAPI", "depends_on", "Uvicorn", "chunk_001")
    graph.add_relation("FastAPI", "depends_on", "Starlette", "chunk_002")
    graph.add_relation("Uvicorn", "uses", "Python", "chunk_002")

    return graph


class TestGraphStore:
    """图谱存储测试"""

    def test_upsert_entity(self, graph):
        """插入实体"""
        graph.upsert_entity("FastAPI", "Technology", "chunk_001")
        entity = graph.get_entity("FastAPI")
        assert entity is not None
        assert entity["type"] == "Technology"
        assert "chunk_001" in entity["chunks"]

    def test_upsert_entity_merge(self, graph):
        """重复插入应合并"""
        graph.upsert_entity("FastAPI", "Technology", "chunk_001")
        graph.upsert_entity("FastAPI", "Technology", "chunk_002", aliases=["Fast API"])
        entity = graph.get_entity("FastAPI")
        assert "chunk_001" in entity["chunks"]
        assert "chunk_002" in entity["chunks"]
        assert "Fast API" in entity["aliases"]

    def test_add_relation(self, graph):
        """添加关系"""
        graph.upsert_entity("FastAPI", "Technology")
        graph.upsert_entity("Python", "Technology")
        graph.add_relation("FastAPI", "uses", "Python", "chunk_001")

        relations = graph.get_relations(subject="FastAPI")
        assert len(relations) == 1
        assert relations[0]["predicate"] == "uses"
        assert relations[0]["object"] == "Python"

    def test_get_entity_not_found(self, graph):
        """不存在的实体应返回 None"""
        assert graph.get_entity("NonExistent") is None

    def test_get_relations_filter(self, populated_graph):
        """关系过滤"""
        # 按 subject
        rels = populated_graph.get_relations(subject="FastAPI")
        assert len(rels) == 3

        # 按 predicate
        rels = populated_graph.get_relations(predicate="uses")
        assert len(rels) == 2

        # 按 object
        rels = populated_graph.get_relations(object="Python")
        assert len(rels) == 2

    def test_multi_hop(self, populated_graph):
        """多跳遍历"""
        hops = populated_graph.multi_hop("FastAPI", ["uses", "uses"])
        assert len(hops) > 0
        # FastAPI -> Python (uses), 然后从 Python 出发的 uses 关系
        first_hop = hops[0]
        assert first_hop["from"] == "FastAPI"
        assert first_hop["relation"] == "uses"

    def test_multi_hop_no_path(self, populated_graph):
        """无路径时应返回空"""
        hops = populated_graph.multi_hop("Python", ["owns"])
        assert hops == []

    def test_get_context_for_entity(self, populated_graph):
        """获取实体上下文"""
        ctx = populated_graph.get_context_for_entity("FastAPI")
        assert "FastAPI" in ctx
        assert "Technology" in ctx

    def test_get_context_not_found(self, graph):
        """不存在实体的上下文应返回空"""
        assert graph.get_context_for_entity("NonExistent") == ""

    def test_stats(self, populated_graph):
        """统计信息"""
        stats = populated_graph.stats()
        assert stats["nodes"] == 4
        assert stats["edges"] == 4
        assert "Technology" in stats["node_types"]
        assert "uses" in stats["relation_types"]
        assert "depends_on" in stats["relation_types"]
