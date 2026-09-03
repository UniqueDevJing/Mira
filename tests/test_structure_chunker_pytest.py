"""StructureChunker: 标题边界 + 递归字符回退"""

from engines.chunking.structure_chunker import StructureChunker


def _doc(blocks):
    return type("Doc", (), {"doc_id": "d1", "pages": [{"page_num": 1, "blocks": blocks}]})


def _title(text, level=None):
    md = {"heading_level": level} if level is not None else {}
    return {"type": "title", "content": text, "page_num": 1, "metadata": md}


def _para(text):
    return {"type": "paragraph", "content": text, "page_num": 1, "metadata": {}}


def test_empty_doc_returns_no_chunks():
    assert StructureChunker().chunk(_doc([])) == []


def test_title_is_segment_boundary():
    blocks = [_title("第一章 概述", 1), _para("正文A"), _title("第二章", 1), _para("正文B")]
    chunks = StructureChunker().chunk(_doc(blocks))
    assert len(chunks) == 2
    assert "正文A" in chunks[0].content and "正文B" not in chunks[0].content
    assert chunks[1].context["title_chain"] == ["第二章"]


def test_nested_title_chain_keeps_parents():
    blocks = [_title("H1", 1), _para("p1"), _title("H2", 2), _para("p2")]
    chunks = StructureChunker().chunk(_doc(blocks))
    assert len(chunks) == 2
    assert chunks[1].context["title_chain"] == ["H1", "H2"]


def test_oversized_segment_recursively_split():
    long = _para("测试内容。" * 300)
    chunks = StructureChunker(max_chars=200, overlap=20).chunk(_doc([_title("T", 1), long]))
    children = [c for c in chunks if not c.metadata.get("is_parent")]
    parents = [c for c in chunks if c.metadata.get("is_parent")]
    # 长段落拆出多个 child(检索单元) + 1 个 parent(整段大上下文)
    assert len(children) >= 2
    assert len(parents) == 1
    # child 均为小段(检索单元), 不超过 chunk_size + overlap 容差
    assert all(len(c.content) <= 200 + 40 for c in children)
    # child 通过 parent_id 回指 parent, parent 内容为整段(大上下文)
    pid = parents[0].chunk_id
    assert all(c.metadata.get("parent_id") == pid for c in children)
    assert len(parents[0].content) > 200
    assert all(c.context["title_chain"] == ["T"] for c in chunks)


def test_no_title_single_segment_split():
    long = _para("无标题文本。" * 200)
    chunks = StructureChunker(max_chars=100, overlap=0).chunk(_doc([long]))
    assert len(chunks) >= 2


def test_chinese_separator_break_no_mid_sentence():
    text = "第一句。" * 4 + "第二句。" * 4 + "第三句。" * 4  # 48 字符, 超过 max_chars=12
    chunks = StructureChunker(max_chars=12, overlap=0).chunk(_doc([_para(text)]))
    # 递归切在中文标点断句, 不在字符中间腰斩
    assert len(chunks) >= 2
    assert all(c.content.endswith("。") for c in chunks)


def test_hard_split_no_double_overlap():
    # 无任何分隔符长串 (URL/长无标点): 硬切路径, overlap 只加一次, 内容不重复
    long = _para("A" * 250)
    chunks = StructureChunker(max_chars=100, overlap=20).chunk(_doc([long]))
    children = [c for c in chunks if not c.metadata.get("is_parent")]
    parents = [c for c in chunks if c.metadata.get("is_parent")]
    assert len(children) >= 2
    assert all(len(c.content) <= 100 + 20 for c in children)
    # 子块总字符数 = 原文 + 每次重叠前缀 = 250 + 2*20 = 290; 旧 bug (_hard_split 自算 overlap) 会到 330
    # (parent 为整段原文, 不计入子块重叠计算)
    assert sum(len(c.content) for c in children) == 250 + 2 * 20
    assert len(parents) == 1 and parents[0].content == "A" * 250


def test_long_segment_produces_parent_child():
    # 长段落(>max_chars)应产 child(多个) + parent, 且 parent_id 链路正确
    long = _para("章节正文内容。" * 200)
    chunks = StructureChunker().chunk(_doc([_title("章", 1), long]))
    children = [c for c in chunks if not c.metadata.get("is_parent")]
    parents = [c for c in chunks if c.metadata.get("is_parent")]
    assert len(parents) == 1
    assert len(children) >= 2
    pid = parents[0].chunk_id
    assert all(c.metadata.get("parent_id") == pid for c in children)
    # parent 自身 parent_id 为空, 内容为整段
    assert parents[0].metadata.get("parent_id") == ""


def test_short_segment_no_parent():
    # 短段落(单块即覆盖)不应产生 parent, 保持与旧版一致的扁平结构
    blocks = [_title("第一章", 1), _para("少量正文。"), _title("第二章", 1), _para("另一段。")]
    chunks = StructureChunker().chunk(_doc(blocks))
    assert len(chunks) == 2
    assert all(not c.metadata.get("is_parent") for c in chunks)
    assert all(c.metadata.get("parent_id") == "" for c in chunks)
