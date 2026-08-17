"""PDF 解析测试 — 覆盖类型检测/行级分类/原生解析块与表格/扫描件委托 OCR/表格异常降级。"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz

from engines.parsing.models import UIRDocument
from engines.parsing.pdf_parser import PDFParser

TEXT = "Native PDF body text extraction works correctly."  # ASCII, 48 字符 > scanned 阈值 30, 且单行不溢出页面宽度


def _make_text_pdf(path: Path, pages: int = 1, text: str = TEXT) -> None:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _pdfplumber_cm(fake_pdf):
    cm = MagicMock()
    cm.__enter__.return_value = fake_pdf
    cm.__exit__.return_value = False
    return cm


def test_detect_type_native(tmp_path):
    pdf = tmp_path / "n.pdf"
    _make_text_pdf(pdf)
    assert PDFParser()._detect_type(str(pdf)) == "native"


def test_detect_type_scanned_when_low_text(tmp_path):
    pdf = tmp_path / "s.pdf"
    doc = fitz.open()
    doc.new_page()  # 空页无文本
    doc.save(str(pdf))
    doc.close()
    assert PDFParser()._detect_type(str(pdf)) == "scanned"


def test_classify_line_header():
    p = PDFParser()
    assert p._classify_line("页眉", [0, 10, 100, 30], [], 842) == "header"


def test_classify_line_footer():
    p = PDFParser()
    assert p._classify_line("页脚", [0, 800, 100, 830], [], 842) == "footer"


def test_classify_line_title_by_size():
    p = PDFParser()
    spans = [{"size": 20, "flags": 0}]
    assert p._classify_line("大标题", [0, 100, 100, 120], spans, 842) == "title"


def test_classify_line_title_by_bold():
    p = PDFParser()
    spans = [{"size": 11, "flags": 16}]
    assert p._classify_line("加粗标题", [0, 100, 100, 120], spans, 842) == "title"


def test_classify_line_title_by_number():
    p = PDFParser()
    spans = [{"size": 11, "flags": 0}]
    assert p._classify_line("1.1 概述", [0, 100, 100, 120], spans, 842) == "title"


def test_classify_line_paragraph():
    p = PDFParser()
    spans = [{"size": 11, "flags": 0}]
    assert p._classify_line("普通正文内容", [0, 200, 100, 220], spans, 842) == "paragraph"


def test_parse_native_builds_blocks_and_tables(tmp_path):
    pdf = tmp_path / "n.pdf"
    _make_text_pdf(pdf)
    fake_pdf = MagicMock()
    fake_page = MagicMock()
    fake_page.extract_tables.return_value = [[["h1", "h2"], ["a", "b"]]]  # 一个 2 行表格
    fake_page.bbox = (0, 0, 595, 842)
    fake_pdf.pages = [fake_page]
    with patch("pdfplumber.open", return_value=_pdfplumber_cm(fake_pdf)):
        doc = PDFParser().parse(str(pdf))
    assert doc.source["type"] == "pdf"
    assert doc.pages[0]["blocks"][0]["content"] == TEXT
    assert len(doc.tables) == 1
    assert doc.tables[0]["matrix"][0] == ["h1", "h2"]


def test_parse_native_table_failure_degrades(tmp_path):
    pdf = tmp_path / "n.pdf"
    _make_text_pdf(pdf)
    fake_pdf = MagicMock()
    fake_page = MagicMock()
    fake_page.extract_tables.side_effect = RuntimeError("bad table")
    fake_page.bbox = (0, 0, 595, 842)
    fake_pdf.pages = [fake_page]
    with patch("pdfplumber.open", return_value=_pdfplumber_cm(fake_pdf)):
        doc = PDFParser().parse(str(pdf))
    assert len(doc.pages) == 1  # 文本仍提取
    assert doc.tables == []  # 表格异常降级为空


def test_parse_scanned_delegates_to_ocr(tmp_path):
    pdf = tmp_path / "s.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    fake_doc = UIRDocument(doc_id="x", source={"type": "scanned_pdf", "path": str(pdf)}, pages=[], tables=[])
    with patch("engines.parsing.ocr.OCRProcessor") as MockOCR:
        MockOCR.return_value.process.return_value = fake_doc
        result = PDFParser().parse(str(pdf))
    assert result.source["type"] == "scanned_pdf"
    MockOCR.return_value.process.assert_called_once()
