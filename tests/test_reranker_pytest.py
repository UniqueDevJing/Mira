"""重排序引擎测试 — 相似度排序 / top_k 截断"""
import pytest
from engines.retrieval.reranker import Reranker


@pytest.fixture
def reranker(embedder):
    """Reranker 实例"""
    return Reranker(embedder=embedder)


@pytest.fixture
def query():
    return "FastAPI 框架的性能特点"


@pytest.fixture
def mixed_docs():
    """混合相关性文档"""
    return [
        {"id": "d1", "chunk_id": "c1",
         "content": "FastAPI 是一个高性能的 Python Web 框架，基于 Starlette 和 Pydantic。它的性能与 Node.js 和 Go 相当。",
         "score": 0.5},
        {"id": "d2", "chunk_id": "c2",
         "content": "Django 是一个全栈 Web 框架，提供了 ORM、模板引擎和 admin 后台。",
         "score": 0.6},
        {"id": "d3", "chunk_id": "c3",
         "content": "FastAPI 支持异步处理、自动生成 OpenAPI 文档、依赖注入系统，并且性能优于大多数 Python 框架。",
         "score": 0.4},
        {"id": "d4", "chunk_id": "c4",
         "content": "Flask 是一个轻量级的 Web 框架，适合小型应用和微服务。",
         "score": 0.55},
        {"id": "d5", "chunk_id": "c5",
         "content": "FastAPI 使用 Uvicorn 作为 ASGI 服务器，支持 WebSocket 和后台任务。",
         "score": 0.45},
    ]


class TestReranker:
    """Reranker 测试"""

    def test_rerank_changes_order(self, reranker, query, mixed_docs):
        """重排应改变文档顺序"""
        original_order = [d["id"] for d in mixed_docs]
        reranked = reranker.rerank(query, mixed_docs, top_k=5)
        reranked_order = [d["id"] for d in reranked]
        assert original_order != reranked_order, "重排未改变文档顺序"

    def test_rerank_relevant_docs_rank_higher(self, reranker, query, mixed_docs):
        """FastAPI 相关文档应排在前面"""
        reranked = reranker.rerank(query, mixed_docs, top_k=5)
        fastapi_ids = {"d1", "d3", "d5"}
        top3_ids = {d["id"] for d in reranked[:3]}
        assert fastapi_ids & top3_ids, "FastAPI 相关文档未进入 Top-3"

    def test_rerank_top_k_truncation(self, reranker, query, mixed_docs):
        """top_k 应正确截断结果"""
        reranked = reranker.rerank(query, mixed_docs, top_k=2)
        assert len(reranked) == 2

    def test_rerank_empty_docs(self, reranker, query):
        """空文档列表应返回空"""
        result = reranker.rerank(query, [], top_k=5)
        assert result == []

    def test_rerank_no_embedder(self, query, mixed_docs):
        """无 embedder 时应返回原文档"""
        reranker = Reranker(embedder=None)
        result = reranker.rerank(query, mixed_docs, top_k=5)
        assert len(result) == 5
        assert result[0]["id"] == "d1"  # 保持原始顺序

    def test_rerank_updates_scores(self, reranker, query, mixed_docs):
        """重排后应更新分数"""
        reranked = reranker.rerank(query, mixed_docs, top_k=5)
        for doc in reranked:
            assert "score" in doc
            assert isinstance(doc["score"], float)
