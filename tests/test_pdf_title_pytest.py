"""PDF 标题检测: 字号 + 加粗 + 编号组合"""
from engines.parsing.pdf_parser import PDFParser


def _block(text, size=12, flags=0, y=100):
    return {
        "bbox": [50, y, 400, y + 30],
        "lines": [{"spans": [{"text": text, "size": size, "flags": flags}]}],
    }


def test_bold_is_title():
    assert PDFParser()._classify_block(_block("摘要", size=12, flags=16), 842) == "title"


def test_large_font_is_title():
    assert PDFParser()._classify_block(_block("章节标题", size=20, flags=0), 842) == "title"


def test_numbered_chinese_title():
    assert PDFParser()._classify_block(_block("第一章 概述", size=12, flags=0), 842) == "title"


def test_numbered_arabic_title():
    assert PDFParser()._classify_block(_block("1.1 背景", size=12, flags=0), 842) == "title"


def test_plain_paragraph_not_title():
    assert PDFParser()._classify_block(_block("普通正文文本", size=12, flags=0), 842) == "paragraph"
