"""RRF 融合单测 — 验证排序融合公式与边界 (纯逻辑, 不依赖模型)。"""

from engines.retrieval.fusion import RRF_K, _tag_rrf, rrf_fuse


def _doc(cid, extra=None):
    d = {"chunk_id": cid, "content": f"doc {cid}"}
    if extra:
        d.update(extra)
    return d


def test_rrf_fuse_ranks_by_score_desc():
    v = [_doc("a"), _doc("b"), _doc("c")]
    b = [_doc("x"), _doc("y")]
    out = rrf_fuse(v, b)
    scores = [d["_rrf"] for d in out]
    assert scores == sorted(scores, reverse=True)
    assert len(out) == 5


def test_rrf_fuse_overlap_accumulates():
    v = [_doc("a"), _doc("b")]
    b = [_doc("a"), _doc("c")]  # "a" 两路都有 → 排名分累加
    out = rrf_fuse(v, b)
    a = next(d for d in out if d["chunk_id"] == "a")
    # 向量路 rank0 + bm25 路 rank0 => 2/(k+1)
    assert a["_rrf"] == round(2.0 / (RRF_K + 1), 6)


def test_rrf_fuse_empty_vector_passthrough():
    b = [_doc("x"), _doc("y")]
    out = rrf_fuse([], b)
    assert [d["chunk_id"] for d in out] == ["x", "y"]
    assert out[0]["_rrf"] == round(1.0 / (RRF_K + 1), 6)


def test_rrf_fuse_empty_bm25_passthrough():
    v = [_doc("a")]
    out = rrf_fuse(v, [])
    assert out[0]["chunk_id"] == "a"


def test_rrf_fuse_both_empty():
    assert rrf_fuse([], []) == []


def test_rrf_fuse_ignores_docs_without_id():
    v = [{"content": "no id"}, _doc("a")]
    b = [_doc("b")]
    out = rrf_fuse(v, b)
    assert {d["chunk_id"] for d in out} == {"a", "b"}


def test_rrf_fuse_custom_k():
    v = [_doc("a")]
    b = [_doc("b")]
    out = rrf_fuse(v, b, k=10)
    assert out[0]["_rrf"] == round(1.0 / (10 + 1), 6)


def test_tag_rrf_adds_field():
    out = _tag_rrf([_doc("a"), _doc("b")])
    assert out[0]["_rrf"] == round(1.0 / (RRF_K + 1), 6)
    assert out[1]["_rrf"] == round(1.0 / (RRF_K + 2), 6)


def test_rrf_fuse_preserves_content():
    v = [_doc("a", {"score": 0.9})]
    out = rrf_fuse(v, [])
    assert out[0]["score"] == 0.9
