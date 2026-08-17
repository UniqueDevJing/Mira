"""RelationExtractor 规则兜底测试 — LLM 不可用/熔断时仍产出边 (P0)"""

from engines.graph_rag.entity_extractor import Entity, RelationExtractor


def _ent(name):
    return Entity(name=name, type="Technology")


def test_verb_pattern_builds_directed_edge():
    """动词模式: "X 使用 Y" → (X, uses, Y), 不反向。"""
    r = RelationExtractor()  # 无 LLM Key → 规则兜底
    rels = r.extract("FastAPI 使用 Python 构建 API 服务", [_ent("FastAPI"), _ent("Python")], "c1")
    assert ("FastAPI", "uses", "Python") in [(x.subject, x.predicate, x.object) for x in rels]


def test_cooccurrence_fallback():
    """同句共现但无动词 → related_to 弱关系。"""
    r = RelationExtractor()
    rels = r.extract(
        "系统对比了 Redis、LanceDB 与 Milvus 的存储方案", [_ent("Redis"), _ent("LanceDB"), _ent("Milvus")], "c2"
    )
    pairs = {(x.subject, x.object) for x in rels}
    assert pairs == {("Redis", "LanceDB"), ("Redis", "Milvus"), ("LanceDB", "Milvus")}
    assert all(x.predicate == "related_to" for x in rels)


def test_no_relation_across_sentences():
    """不同句的实体不建边 (规则边界: 共现限于同句)。"""
    r = RelationExtractor()
    rels = r.extract("Docker 负责容器化。Kubernetes 负责编排。", [_ent("Docker"), _ent("Kubernetes")], "c3")
    assert rels == []


def test_empty_entities_no_relation():
    """无实体输入 → 空关系。"""
    r = RelationExtractor()
    assert r.extract("任何文本", [], "c4") == []


def test_llm_unavailable_never_raises():
    """熔断/无 Key 场景不抛异常, 稳定返回规则结果。"""
    r = RelationExtractor()
    r._fail_count = r._max_fails  # 模拟熔断后
    rels = r.extract("Milvus 依赖 MinIO 做对象存储", [_ent("Milvus"), _ent("MinIO")], "c5")
    assert rels
