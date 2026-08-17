"""多知识库隔离测试 — VectorStore 多表 + state 按库单例。"""

import tempfile

from engines.interfaces import Chunk
from engines.retrieval.vector_store import VectorStore


def _chunks(kb_tag: str):
    # 非零向量: LanceDB cosine 对零向量查询会返回空
    return [
        Chunk(
            chunk_id=f"{kb_tag}_c{i}",
            doc_id=f"{kb_tag}_doc{i}",
            content=f"{kb_tag} 内容 {i}",
            embedding=[0.1 * (i + 1)] * 512,
        )
        for i in range(3)
    ]


def test_vector_store_table_isolation():
    with tempfile.TemporaryDirectory() as uri:
        a = VectorStore(uri=uri, table_name="rag_test_a")
        b = VectorStore(uri=uri, table_name="rag_test_b")
        a.insert(_chunks("a"))
        b.insert(_chunks("b"))

        ra = a.search([0.2] * 512, top_k=5)
        rb = b.search([0.2] * 512, top_k=5)
        assert len(ra) == 3 and len(rb) == 3
        assert all(d["doc_id"].startswith("a_") for d in ra)
        assert all(d["doc_id"].startswith("b_") for d in rb)


def test_state_per_kb_singleton():
    from api.state import get_bm25_index, get_graph_rag, get_vector_store

    # 不同库返回不同实例与表
    sa, sb = get_vector_store("service"), get_vector_store("tech")
    assert sa is not sb
    assert sa.table_name == "rag_service" and sb.table_name == "rag_tech"
    # 同一库幂等
    assert get_vector_store("service") is sa
    # 图谱隔离
    assert get_graph_rag("service") is not get_graph_rag("tech")
    # BM25 隔离
    assert get_bm25_index("service") is not get_bm25_index("tech")
    # 默认库兼容旧表名
    assert get_vector_store("documents").table_name == "documents"
