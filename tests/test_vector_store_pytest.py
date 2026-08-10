"""向量存储测试 — 插入 / 检索 / 删除"""
import tempfile

import pytest

from engines.chunking.structure_chunker import Chunk
from engines.retrieval.vector_store import VectorStore


@pytest.fixture
def store():
    """临时向量存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(uri=tmpdir)
        yield store


@pytest.fixture
def sample_chunks():
    """测试文档块"""
    return [
        Chunk(
            chunk_id="chunk_001",
            doc_id="doc_001",
            content="FastAPI 是一个高性能的 Python Web 框架",
            embedding=[0.1] * 512,
        ),
        Chunk(
            chunk_id="chunk_002",
            doc_id="doc_001",
            content="LanceDB 是一个嵌入式向量数据库",
            embedding=[0.2] * 512,
        ),
        Chunk(
            chunk_id="chunk_003",
            doc_id="doc_002",
            content="Milvus 是一个分布式向量数据库",
            embedding=[0.3] * 512,
        ),
    ]


class TestVectorStore:
    """向量存储测试"""

    def test_insert_and_search(self, store, sample_chunks):
        """插入后应能检索到"""
        store.insert(sample_chunks)
        query_embedding = [0.15] * 512
        results = store.search(query_embedding, top_k=5)
        assert len(results) > 0
        assert all("id" in r for r in results)
        assert all("score" in r for r in results)

    def test_search_top_k(self, store, sample_chunks):
        """top_k 应限制返回数量"""
        store.insert(sample_chunks)
        query_embedding = [0.1] * 512
        results = store.search(query_embedding, top_k=2)
        assert len(results) <= 2

    def test_search_returns_content(self, store, sample_chunks):
        """检索结果应包含内容"""
        store.insert(sample_chunks)
        query_embedding = [0.1] * 512
        results = store.search(query_embedding, top_k=1)
        assert len(results) > 0
        assert "content" in results[0]
        assert results[0]["content"]

    def test_search_score_range(self, store, sample_chunks):
        """分数应在合理范围内"""
        store.insert(sample_chunks)
        query_embedding = [0.1] * 512
        results = store.search(query_embedding, top_k=5)
        for r in results:
            assert -1.0 <= r["score"] <= 1.0

    def test_search_empty_store(self, store):
        """空库检索应返回空列表"""
        query_embedding = [0.1] * 512
        results = store.search(query_embedding, top_k=5)
        assert results == []

    def test_delete_by_doc_id(self, store, sample_chunks):
        """按文档 ID 删除"""
        store.insert(sample_chunks)
        store.delete_by_doc_id("doc_001")
        query_embedding = [0.1] * 512
        results = store.search(query_embedding, top_k=5)
        doc_ids = {r["doc_id"] for r in results}
        assert "doc_001" not in doc_ids

    def test_filter_expression(self, store, sample_chunks):
        """过滤表达式应生效"""
        store.insert(sample_chunks)
        query_embedding = [0.1] * 512
        results = store.search(query_embedding, top_k=5, filter_expr='doc_id = "doc_002"')
        for r in results:
            assert r["doc_id"] == "doc_002"
