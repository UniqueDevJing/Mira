"""MarkdownParser: # 标题 → title block, 空行分段"""

import pytest

from engines.parsing.markdown_parser import MarkdownParser


@pytest.fixture
def md_path(tmp_path):
    p = tmp_path / "sample.md"
    p.write_text("# 第一章 概述\n\n这是第一段。\n\n## 1.1 背景\n\n这是背景段落。", encoding="utf-8")
    return str(p)


def test_heading_level_detected(md_path):
    uir = MarkdownParser().parse(md_path)
    titles = [b for b in uir.pages[0]["blocks"] if b["type"] == "title"]
    assert [t["content"] for t in titles] == ["第一章 概述", "1.1 背景"]
    assert [t["metadata"]["heading_level"] for t in titles] == [1, 2]


def test_paragraph_block(md_path):
    uir = MarkdownParser().parse(md_path)
    paras = [b for b in uir.pages[0]["blocks"] if b["type"] == "paragraph"]
    assert any("这是第一段" in b["content"] for b in paras)


def test_doc_id_deterministic(md_path):
    assert MarkdownParser().parse(md_path).doc_id == MarkdownParser().parse(md_path).doc_id
