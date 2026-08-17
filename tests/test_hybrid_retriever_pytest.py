"""HybridRetriever 单测 — 向量+图谱融合 / 去重 / 降级路径"""

from engines.retrieval.hybrid_retriever import HybridRetriever


class FakeVectorStore:
    """最小向量库替身: search 返回固定文档, get_by_ids 按 id 查回。"""

    def __init__(self, docs: list[dict]):
        self.docs = docs

    def search(self, emb, top_k: int = 20) -> list[dict]:
        return self.docs[:top_k]

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        by = {d.get("chunk_id") or d.get("id"): d for d in self.docs}
        return [by[i] for i in ids if i in by]


class FakeEmbedder:
    def embed_query(self, q: str) -> list[float]:
        return [0.0] * 8


class FakeReranker:
    def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        return documents[:top_k]


def _d(did: str) -> dict:
    return {"id": did, "chunk_id": did, "content": f"内容{did}", "score": 1.0}


def test_retrieve_returns_top_k_documents():
    vs = FakeVectorStore([_d("c1"), _d("c2"), _d("c3")])
    hr = HybridRetriever(vector_store=vs, embedder=FakeEmbedder())
    r = hr.retrieve("query", top_k=2)
    assert r["documents"]
    assert len(r["documents"]) <= 2
    assert r["graph_context"] is None


def test_retrieve_applies_reranker():
    vs = FakeVectorStore([_d("c1"), _d("c2")])
    hr = HybridRetriever(vector_store=vs, embedder=FakeEmbedder(), reranker=FakeReranker())
    r = hr.retrieve("query", top_k=1)
    assert len(r["documents"]) == 1


def test_merge_graph_chunks_str_and_dict():
    """图谱 source_chunks: str 从向量库查内容回填, dict 直接并入, 均不产生重复。"""
    vs = FakeVectorStore([_d("c1"), _d("c2")])
    hr = HybridRetriever(vector_store=vs)
    merged = hr._merge_graph_chunks(
        [_d("c1")],
        ["c2", {"id": "g1", "chunk_id": "g1", "content": "图谱片段"}],
    )
    keys = [d["chunk_id"] for d in merged]
    assert keys == ["c1", "c2", "g1"]
    assert merged[1]["content"] == "内容c2"  # str 回填走了向量库


def test_merge_graph_chunks_missing_content_placeholder():
    """图谱 chunk 向量库查不到 → 占位串, 不崩溃。"""
    vs = FakeVectorStore([])
    hr = HybridRetriever(vector_store=vs)
    merged = hr._merge_graph_chunks([], ["missing_chunk"])
    assert merged[0]["content"].startswith("[图谱关联片段")


def test_dedup_removes_duplicate_chunk_ids():
    hr = HybridRetriever(vector_store=None)
    out = hr._dedup([_d("a"), _d("a"), _d("b")])
    assert len(out) == 2


def test_retrieve_without_embedder_returns_empty():
    hr = HybridRetriever(vector_store=None)
    assert hr.retrieve("query")["documents"] == []


def test_lookup_chunk_content_empty_for_missing():
    class EmptyVS:
        def get_by_ids(self, ids: list[str]) -> list[dict]:
            return []

    hr = HybridRetriever(vector_store=EmptyVS())
    assert hr._lookup_chunk_content("missing") == ""


def test_lookup_chunk_content_returns_stored():
    vs = FakeVectorStore([_d("c1")])
    hr = HybridRetriever(vector_store=vs)
    assert hr._lookup_chunk_content("c1") == "内容c1"
