"""vector_store / reranker 覆盖率补强 — 核心检索组件的薄弱路径（验证/降级/边界）。"""

from unittest.mock import MagicMock

from engines.retrieval.reranker import Reranker
from engines.retrieval.vector_store import VectorStore


class _FakeEmbedder:
    dim = 512

    def embed_query(self, text):
        return [1.0] * self.dim

    def embed_batch(self, texts):
        return [[1.0] * self.dim for _ in texts]


class _Chunk:
    def __init__(self, chunk_id, doc_id, content, embedding, context=None, metadata=None):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.content = content
        self.embedding = embedding
        self.context = context or {}
        self.metadata = metadata or {}


def _make_store(tmp_path):
    return VectorStore(uri=str(tmp_path / "lancedb"), dim=512, table_name="documents")


# ---------- VectorStore ----------


def test_vs_insert_then_search_returns_structured_fields(tmp_path):
    vs = _make_store(tmp_path)
    vs.insert(
        [
            _Chunk(
                "c1",
                "d1",
                "向量检索用于召回",
                [1.0] * 512,
                context={"title_chain": ["H1"], "doc_title": "Doc"},
                metadata={"page_range": [1, 2]},
            ),
            _Chunk("c2", "d1", "图谱检索补充实体", [0.5] * 512),
        ]
    )
    docs = vs.search([1.0] * 512, top_k=5)
    assert len(docs) == 2
    by_id = {d["chunk_id"]: d for d in docs}
    assert by_id["c1"]["content"] == "向量检索用于召回"
    assert by_id["c1"]["title_chain"] == ["H1"]
    assert by_id["c1"]["doc_title"] == "Doc"
    assert by_id["c1"]["page_range"] == [1, 2]
    assert 0.0 <= by_id["c1"]["score"] <= 1.0


def test_vs_insert_missing_embedding_raises(tmp_path):
    vs = _make_store(tmp_path)
    try:
        vs.insert([_Chunk("c1", "d1", "x", None)])
        raise AssertionError("应因缺 embedding 抛 ValueError")
    except ValueError:
        pass


def test_vs_insert_dim_mismatch_raises(tmp_path):
    vs = _make_store(tmp_path)
    try:
        vs.insert([_Chunk("c1", "d1", "x", [1.0] * 256)])
        raise AssertionError("应因维度不符抛 ValueError")
    except ValueError:
        pass


def test_vs_get_by_ids_empty(tmp_path):
    vs = _make_store(tmp_path)
    assert vs.get_by_ids([]) == []


def test_vs_get_by_ids_returns_content(tmp_path):
    vs = _make_store(tmp_path)
    vs.insert([_Chunk("c1", "d1", "召回内容", [1.0] * 512)])
    rows = vs.get_by_ids(["c1"])
    assert len(rows) == 1
    assert rows[0]["content"] == "召回内容"
    assert rows[0]["chunk_id"] == "c1"


def test_vs_search_with_filter(tmp_path):
    vs = _make_store(tmp_path)
    vs.insert(
        [
            _Chunk("c1", "d1", "A 文档内容", [1.0] * 512),
            _Chunk("c2", "d2", "B 文档内容", [1.0] * 512),
        ]
    )
    docs = vs.search([1.0] * 512, top_k=5, filter_expr="doc_id = 'd2'")
    assert [d["chunk_id"] for d in docs] == ["c2"]


def test_vs_delete_by_doc_id(tmp_path):
    vs = _make_store(tmp_path)
    vs.insert([_Chunk("c1", "d1", "内容", [1.0] * 512)])
    vs.delete_by_doc_id("d1")
    assert vs.search([1.0] * 512, top_k=5) == []


# ---------- Reranker ----------


def test_rerank_empty_returns_empty():
    assert Reranker(embedder=_FakeEmbedder()).rerank("q", []) == []


def test_rerank_no_embedder_returns_topk():
    docs = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
    out = Reranker(embedder=None).rerank("q", docs, top_k=2)
    assert out == docs[:2]


def test_rerank_no_embedder_returns_top_k():
    docs = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
    out = Reranker(embedder=None).rerank("q", docs, top_k=2)
    assert out == docs[:2]


def test_rerank_no_ce_returns_top_k():
    docs = [
        {"content": "a", "embedding": [1.0] * 512},
        {"content": "b", "embedding": [0.0] * 512},
    ]
    out = Reranker(embedder=_FakeEmbedder(), ce_model_name="").rerank("q", docs, top_k=2)
    assert out[0]["content"] == "a"  # 保持原始顺序


def test_rerank_no_ce_missing_vectors_no_crash():
    docs = [{"content": "a"}, {"content": "b"}]
    out = Reranker(embedder=_FakeEmbedder(), ce_model_name="").rerank("q", docs, top_k=2)
    assert len(out) == 2


def test_rerank_ce_path_uses_predict_scores():
    r = Reranker(embedder=_FakeEmbedder())
    ce = MagicMock()
    ce.predict.return_value = [0.2, 0.9]
    r._ce_model = ce
    out = r.rerank("q", [{"content": "a"}, {"content": "b"}], top_k=2)
    assert out[0]["content"] == "b"  # 高分在前
    assert out[0]["score"] == 0.9


def test_rerank_ce_exception_falls_back_to_bi_encoder():
    r = Reranker(embedder=_FakeEmbedder())
    ce = MagicMock()
    ce.predict.side_effect = RuntimeError("boom")
    r._ce_model = ce
    docs = [
        {"content": "a", "embedding": [1.0] * 512},
        {"content": "b", "embedding": [0.0] * 512},
    ]
    out = r.rerank("q", docs, top_k=2)
    assert out[0]["content"] == "a"  # 回退 Bi-Encoder: a 相似度更高
