"""GraphRAG 批量抽取测试 — 实体/关系批量方法 + build_from_chunks 批量路径。

规则模式 (无 LLM) 验证行为与单次 extract 一致; LLM 模式用 FakeLLM 模拟批量响应。
不依赖真实 LLM/网络。
"""

import types

from engines.graph_rag.entity_extractor import Entity, EntityExtractor, Relation, RelationExtractor
from engines.graph_rag.graph_retriever import GraphRAGRetriever
from engines.graph_rag.graph_store import GraphStore


def _ent(name: str) -> Entity:
    return Entity(name=name, type="Technology")


class FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def chat(self, messages, **kwargs):
        return types.SimpleNamespace(content=self._content)


def _extractor_with_llm(extractor_cls, content: str):
    """构造带 LLM Key 的抽取器, 并把 LLM 客户端替换为 FakeLLM。"""
    inst = extractor_cls(llm_url="http://fake", llm_model="m", llm_key="k")
    inst._get_llm_client = lambda: FakeLLM(content)
    return inst


# ────────────────────────── 规则模式 (无 LLM) ──────────────────────────


def test_entity_batch_rule_mode():
    ex = EntityExtractor()
    result = ex.extract_batch([("Python 使用 RAG", "c0"), ("Docker 和 Kubernetes 编排", "c1")])
    names0 = {e.name for e in result["c0"]}
    names1 = {e.name for e in result["c1"]}
    assert "Python" in names0 and "RAG" in names0
    assert "Docker" in names1 and "Kubernetes" in names1


def test_relation_batch_rule_mode():
    rx = RelationExtractor()
    entities = {"c1": [_ent("FastAPI"), _ent("Python")]}
    result = rx.extract_batch([("FastAPI 使用 Python 构建 API 服务", "c1")], entities)
    rels = result["c1"]
    assert any(r.subject == "FastAPI" and r.predicate == "uses" and r.object == "Python" for r in rels)


def test_relation_batch_empty_entities_returns_empty():
    rx = RelationExtractor()
    result = rx.extract_batch([("任意文本", "c9")], {"c9": []})
    assert result["c9"] == []


# ────────────────────────── LLM 批量模式 ──────────────────────────


def test_entity_batch_with_llm_parses_per_chunk():
    content = '{"c0":[{"name":"FastAPI","type":"Technology","aliases":["FA"]}],"c1":[{"name":"Docker","type":"Technology","aliases":[]}]}'
    ex = _extractor_with_llm(EntityExtractor, content)
    result = ex.extract_batch([("文本A", "c0"), ("文本B", "c1")])
    assert [e.name for e in result["c0"]] == ["FastAPI"]
    assert [e.name for e in result["c1"]] == ["Docker"]
    assert result["c0"][0].aliases == ["FA"]


def test_entity_batch_llm_failure_falls_back_rule():
    ex = _extractor_with_llm(EntityExtractor, "这不是JSON，乱码")
    result = ex.extract_batch([("Python 教程", "c0")])
    assert any(e.name == "Python" for e in result["c0"])  # 规则兜底
    assert ex._fail_count == 1


def test_relation_batch_with_llm_parses_per_chunk():
    content = '{"c0":[{"subject":"FastAPI","predicate":"uses","object":"Python"}],"c1":[]}'
    rx = _extractor_with_llm(RelationExtractor, content)
    entities = {"c0": [_ent("FastAPI"), _ent("Python")], "c1": []}
    result = rx.extract_batch([("文本A", "c0"), ("文本B", "c1")], entities)
    rels0 = result["c0"]
    assert len(rels0) == 1 and rels0[0].subject == "FastAPI" and rels0[0].predicate == "uses"
    assert result["c1"] == []


def test_relation_batch_llm_failure_falls_back_rule():
    rx = _extractor_with_llm(RelationExtractor, "坏数据")
    entities = {"c0": [_ent("FastAPI"), _ent("Python")]}
    result = rx.extract_batch([("FastAPI 使用 Python 构建 API", "c0")], entities)
    assert any(r.predicate == "uses" for r in result["c0"])  # 规则兜底
    assert rx._fail_count == 1


# ────────────────────────── build_from_chunks 批量路径 ──────────────────────────


class _FakeExtractor:
    def __init__(self, entities, relations):
        self._entities = entities
        self._relations = relations

    def extract_batch(self, batch, entities_map=None):
        if entities_map is None:
            return self._entities
        return self._relations

    def extract(self, *a, **kw):  # retrieve 路径仍可单条调用
        return []


def test_build_from_chunks_batches_and_writes_graph():
    class Chunk:
        def __init__(self, content, chunk_id):
            self.content = content
            self.chunk_id = chunk_id

    chunks = [Chunk("Python 使用 RAG", "c0"), Chunk("Docker 编排", "c1")]
    entity_map = {
        "c0": [Entity(name="Python", type="Technology", source_chunk_id="c0")],
        "c1": [Entity(name="Docker", type="Technology", source_chunk_id="c1")],
    }
    relation_map = {
        "c0": [Relation(subject="Python", predicate="uses", object="RAG", source_chunk_id="c0")],
        "c1": [],
    }
    store = GraphStore()
    retriever = GraphRAGRetriever(
        _FakeExtractor(entity_map, relation_map), _FakeExtractor(entity_map, relation_map), store
    )

    result = retriever.build_from_chunks(chunks)

    assert result["entities"] == 2
    assert result["relations"] == 1
    assert store.get_entity("Python")["type"] == "Technology"
