"""核心算法正确性契约测试 — fusion (RRF) + structure_chunker。

动机: 这两个模块是检索质量的底座, 此前零测试防护; 一旦回归会静默破坏
检索结果而不报错。本文件锁定其行为契约 (非为覆盖率数字, 而是回归护栏)。

不依赖任何外部服务/模型, 纯算法可重复。
"""

from types import SimpleNamespace

from engines.chunking.structure_chunker import RecursiveTextSplitter, StructureChunker
from engines.retrieval.fusion import RRF_K, rrf_fuse

# ---------- fusion / RRF ----------


def _doc(cid, score=None):
    d = {"chunk_id": cid, "content": f"content-{cid}"}
    if score is not None:
        d["score"] = score
    return d


def test_rrf_fuses_two_routes_and_sorts_by_score():
    vec = [_doc("a"), _doc("b"), _doc("c")]  # 排名 a>b>c
    bm25 = [_doc("c"), _doc("a"), _doc("d")]  # 排名 c>a>d
    out = rrf_fuse(vec, bm25)
    ids = [d["chunk_id"] for d in out]
    assert ids[0] == "a"  # a 两路都排前 → 分数最高
    assert "d" in ids and "b" in ids
    assert all("_rrf" in d for d in out)


def test_rrf_vector_only_gets_tagged():
    vec = [_doc("a"), _doc("b")]
    out = rrf_fuse(vec, [])
    assert [d["chunk_id"] for d in out] == ["a", "b"]
    assert out[0]["_rrf"] >= out[1]["_rrf"]  # 排名权重递减


def test_rrf_bm25_only_gets_tagged():
    bm25 = [_doc("x"), _doc("y")]
    out = rrf_fuse([], bm25)
    assert [d["chunk_id"] for d in out] == ["x", "y"]
    assert "_rrf" in out[0]


def test_rrf_skips_docs_without_id():
    # 双路 _feed 与单路 _tag_rrf 都应跳过无 chunk_id/id 的 doc
    vec = [_doc("a"), {"content": "no-id"}]
    out = rrf_fuse(vec, [])
    assert [d["chunk_id"] for d in out] == ["a"]
    bm25 = [_doc("a"), {"content": "no-id"}]
    out2 = rrf_fuse([], bm25)
    assert [d["chunk_id"] for d in out2] == ["a"]


def test_rrf_same_id_across_routes_accumulates():
    vec = [_doc("a")]
    bm25 = [_doc("a")]  # 同 id 两路都出现
    out = rrf_fuse(vec, bm25)
    assert len(out) == 1
    assert out[0]["chunk_id"] == "a"
    # 分数应 > 单路 (1/(k+1) + 1/(k+1))
    assert out[0]["_rrf"] > 1.0 / (RRF_K + 1)


def test_rrf_empty_both_returns_empty():
    assert rrf_fuse([], []) == []


def test_rrf_negative_k_defended():
    # 负 k 不应崩溃/除零, 应回退 RRF_K
    vec = [_doc("a"), _doc("b")]
    out = rrf_fuse(vec, [], k=-1)
    assert [d["chunk_id"] for d in out] == ["a", "b"]
    assert out[0]["_rrf"] > 0


# ---------- structure_chunker ----------


def _block(btype, content, level=None, page=1):
    md = {}
    if level is not None:
        md["heading_level"] = level
    return {"type": btype, "content": content, "metadata": md, "page_num": page}


def _uir(blocks, doc_id="d1", source=None):
    return SimpleNamespace(
        doc_id=doc_id,
        source=source or {"path": "/data/doc.txt"},
        pages=[{"blocks": blocks}],
    )


def test_chunk_empty_pages_returns_empty():
    doc = SimpleNamespace(doc_id="d1", source={"path": "x"}, pages=[])
    assert StructureChunker().chunk(doc) == []


def test_chunk_pure_title_skipped():
    doc = _uir([_block("title", "标题一", 1), _block("title", "标题二", 2)])
    assert StructureChunker().chunk(doc) == []


def test_recursive_splitter_applies_overlap():
    # 直接测 RecursiveTextSplitter 的 overlap 语义 (确定性, 不受 StructureChunker.strip 干扰)
    splitter = RecursiveTextSplitter(chunk_size=40, chunk_overlap=12)
    text = "，".join(f"片段{i}内容" for i in range(15))
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    # 后块以前块尾部 overlap 字符开头
    assert chunks[1].startswith(chunks[0][-12:])


def test_chunk_long_paragraph_hard_split():
    # 无分隔符的超长文本 → 硬切为多个 <= max_chars 的块
    long_text = "x" * 2000
    doc = _uir([_block("paragraph", long_text)])
    chunks = StructureChunker(max_chars=800, overlap=0).chunk(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.content) <= 800


def test_chunk_heading_chain_tracks_levels():
    doc = _uir(
        [
            _block("title", "第一章", 1),
            _block("paragraph", "正文A"),
            _block("title", "第一节", 2),
            _block("paragraph", "正文B"),
        ]
    )
    chunks = StructureChunker().chunk(doc)
    # 正文A 的标题链应含 "第一章"; 正文B 应含 "第一章"+"第一节"
    assert any("第一章" in (c.context.get("title_chain") or []) for c in chunks)
    b_chunk = next(c for c in chunks if "正文B" in c.content)
    assert b_chunk.context["title_chain"] == ["第一章", "第一节"]


def test_chunk_estimate_level_by_font_size():
    # 无 heading_level 时按字号兜底
    big = _block("title", "大标题", level=None)
    big["metadata"]["font_size"] = 24
    small = _block("title", "小标题", level=None)
    small["metadata"]["font_size"] = 13
    assert StructureChunker._estimate_level(big) == 1
    assert StructureChunker._estimate_level(small) == 4


def test_chunk_metadata_carries_page_and_counts():
    doc = _uir(
        [
            _block("title", "标题", 1, page=1),
            _block("paragraph", "第一段", page=2),
            _block("paragraph", "第二段", page=3),
        ]
    )
    chunks = StructureChunker().chunk(doc)
    assert len(chunks) == 1
    md = chunks[0].metadata
    # segment 含标题块(page1) + 两段正文(page2,3)
    assert md["page_range"] == [1, 3]
    # segment 含 标题块 + 2 段落块, paragraph_count 记录 segment 块数
    assert md["paragraph_count"] == 3
    assert md["char_count"] > 0
