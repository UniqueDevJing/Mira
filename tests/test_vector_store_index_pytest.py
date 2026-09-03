"""向量存储索引回归 — ANN 向量索引 + 标量索引 + get_by_* 修复

P2#9: 大语料建 IvfHnswFlat(cosine) ANN 索引降低检索延迟; doc_id/parent_id 建标量索引加速过滤下推。
附带回归: 原 get_by_ids/get_by_doc_id 的零向量 + prefilter 写法在 ANN 索引下返回 0 行,
且 get_by_doc_id 有隐藏的 10 条上限 → 统一改为无向量 search().where() 全表过滤扫描。
"""

import tempfile

import numpy as np
import pytest

from engines.chunking.structure_chunker import Chunk
from engines.retrieval.vector_store import VectorStore

DIM = 64  # 测试用较小维度, 加速; 真实为 512
N_LARGE = 300  # >= 256 阈值, 触发建索引


def _dense_unit(rng):
    v = rng.standard_normal(DIM)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def large_store():
    """>=256 行的稠密向量存储, 触发索引创建。"""
    rng = np.random.default_rng(7)
    base = [_dense_unit(rng) for _ in range(N_LARGE)]
    with tempfile.TemporaryDirectory() as d:
        s = VectorStore(uri=d, dim=DIM)
        chunks = [
            Chunk(chunk_id=f"c{i}", doc_id=f"d{i % 7}", content=f"x{i}", embedding=base[i])
            for i in range(N_LARGE)
        ]
        s.insert(chunks)
        yield s


class TestIndexCreation:
    def test_ann_and_scalar_indexes_created_on_large_table(self, large_store):
        """>=256 行应建向量 ANN 索引 + doc_id/parent_id 标量索引。"""
        assert large_store._index_exists("embedding") is True
        assert large_store._index_exists("doc_id") is True
        assert large_store._index_exists("parent_id") is True

    def test_no_index_on_small_table(self):
        """<256 行不应建索引 (暴力检索足够快且规避小表边界)。"""
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            s = VectorStore(uri=d, dim=DIM)
            chunks = [
                Chunk(chunk_id=f"c{i}", doc_id="d0", content=f"x{i}", embedding=_dense_unit(rng))
                for i in range(5)
            ]
            s.insert(chunks)
            assert s._index_exists("embedding") is False
            assert s._index_exists("doc_id") is False


class TestSearchRecallWithIndex:
    def test_recall_top1_correct(self, large_store):
        """ANN 索引下稠密向量召回正确 (top1 应为查询对应的 chunk)。"""
        rng = np.random.default_rng(7)
        k = 42
        # 重新构造与构建时一致的向量: 用同 seed 重放
        base = [_dense_unit(rng) for _ in range(N_LARGE)]
        q = np.array(base[k]) + 1e-3 * rng.standard_normal(DIM)
        q = q / np.linalg.norm(q)
        res = large_store.search(q.tolist(), top_k=5)
        assert len(res) > 0
        assert res[0]["id"] == f"c{k}"


class TestGetByFixes:
    def test_get_by_doc_id_returns_all_chunks(self, large_store):
        """修复回归: get_by_doc_id 应返回文档全部 chunk, 不受 10 条限制。"""
        # d0 = i%7==0 → 300/7 ≈ 42 条, 远超原 10 条隐藏上限
        rows = large_store.get_by_doc_id("d0")
        assert len(rows) == (N_LARGE // 7) + (1 if N_LARGE % 7 > 0 else 0)
        assert all(r["doc_id"] == "d0" for r in rows)

    def test_get_by_doc_id_empty(self, large_store):
        assert large_store.get_by_doc_id("") == []

    def test_get_by_ids_works(self, large_store):
        rows = large_store.get_by_ids(["c7", "c13"])
        assert len(rows) == 2
        assert {r["id"] for r in rows} == {"c7", "c13"}

    def test_get_by_ids_empty(self, large_store):
        assert large_store.get_by_ids([]) == []
