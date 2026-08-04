"""Embedding 服务测试 — 嵌入维度 / 批量处理 / 归一化"""
import pytest
import numpy as np
from engines.embedding.embedder import EmbeddingService


@pytest.fixture
def embedder():
    return EmbeddingService()


class TestEmbeddingService:
    """Embedding 服务测试"""

    def test_embed_query_dimension(self, embedder):
        """查询嵌入维度应为 512"""
        embedding = embedder.embed_query("测试查询")
        assert len(embedding) == 512

    def test_embed_query_normalized(self, embedder):
        """查询嵌入应已归一化"""
        embedding = embedder.embed_query("测试查询")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"向量未归一化: norm={norm}"

    def test_embed_batch_returns_list(self, embedder):
        """批量嵌入应返回列表"""
        texts = ["文本1", "文本2", "文本3"]
        embeddings = embedder.embed_batch(texts)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3

    def test_embed_batch_dimension(self, embedder):
        """批量嵌入维度应为 512"""
        texts = ["文本1", "文本2"]
        embeddings = embedder.embed_batch(texts)
        for emb in embeddings:
            assert len(emb) == 512

    def test_embed_batch_empty(self, embedder):
        """空列表应返回空"""
        result = embedder.embed_batch([])
        assert result == []

    def test_embed_similarity(self, embedder):
        """相似文本应有高相似度"""
        emb1 = embedder.embed_query("FastAPI 性能")
        emb2 = embedder.embed_query("FastAPI 框架的性能特点")
        emb3 = embedder.embed_query("今天天气怎么样")

        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        assert sim_12 > sim_13, f"相似文本相似度 ({sim_12}) 应高于不相似文本 ({sim_13})"

    def test_embed_query_prefix(self, embedder):
        """查询应使用 query: 前缀"""
        # 内部实现会自动添加 query: 前缀
        embedding = embedder.embed_query("测试")
        assert len(embedding) == 512

    def test_embed_batch_prefix(self, embedder):
        """批量嵌入应使用 passage: 前缀"""
        # 内部实现会自动添加 passage: 前缀
        embeddings = embedder.embed_batch(["测试"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 512
