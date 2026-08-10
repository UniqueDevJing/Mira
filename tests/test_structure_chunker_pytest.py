"""StructureChunker: 标题边界 + 递归字符回退"""
import pytest

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
    assert len(chunks) >= 2
    assert all(len(c.content) <= 200 + 40 for c in chunks)  # chunk_size + overlap 容差
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
    assert len(chunks) >= 2
    assert all(len(c.content) <= 100 + 20 for c in chunks)
    # 总字符数 = 原文 + 每次重叠前缀 = 250 + 2*20 = 290; 旧 bug (_hard_split 自算 overlap) 会到 330
    assert sum(len(c.content) for c in chunks) == 250 + 2 * 20
