"""TxtParser: 空行分段, 全为 paragraph"""

import pytest

from engines.parsing.txt_parser import TxtParser


@pytest.fixture
def txt_path(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("第一段文本。\n\n第二段文本。", encoding="utf-8")
    return str(p)


def test_paragraphs_split_by_blank_line(txt_path):
    uir = TxtParser().parse(txt_path)
    blocks = uir.pages[0]["blocks"]
    assert len(blocks) == 2
    assert all(b["type"] == "paragraph" for b in blocks)
    assert "第一段" in blocks[0]["content"]


def test_no_title_blocks(txt_path):
    uir = TxtParser().parse(txt_path)
    assert not any(b["type"] == "title" for b in uir.pages[0]["blocks"])
