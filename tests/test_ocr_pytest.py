"""OCR 处理器测试 — mock RapidOCR 避免真实模型加载, 覆盖 DPI 降级/块构建/单页降级/空结果。"""

import types
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from engines.parsing.ocr import OCRProcessor

# RapidOCR 单页返回: list[(bbox, text, confidence)], bbox 为 4 角点 [[x,y],...]
OCR_RESULT = [[[(0, 0), (100, 0), (100, 50), (0, 50)], "识别文字", 0.95]]


def _make_pdf(path: Path, pages: int = 1, text: str = "内容" * 20) -> None:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


class _FakeOCR:
    """模拟 RapidOCR: __call__(img) -> (result, None)。result=None 表示无文字。"""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        if not self._results:
            return (None, None)
        return (self._results[(self.calls - 1) % len(self._results)], None)


@pytest.fixture
def patched_ocr():
    import engines.parsing.ocr as ocr_mod

    with patch("engines.parsing.ocr.RapidOCR") as mock_cls:
        mock_cls.return_value = _FakeOCR([OCR_RESULT])
        ocr_mod._ocr_instance = None
        yield


def test_dpi_for_normal_a4():
    page = types.SimpleNamespace(rect=types.SimpleNamespace(width=595, height=842))
    assert OCRProcessor._dpi_for(page) == 300


def test_dpi_for_oversized_page_stays_under_pixmap_cap():
    # 5000x5000 远超 ~5000x5000 像素上限, 必须动态降 dpi 防 OOM
    page = types.SimpleNamespace(rect=types.SimpleNamespace(width=5000, height=5000))
    dpi = OCRProcessor._dpi_for(page)
    assert dpi < 300
    px = (5000 * dpi / 72) * (5000 * dpi / 72)
    assert px <= 25_000_000


def test_dpi_for_medium_page_downscales():
    page = types.SimpleNamespace(rect=types.SimpleNamespace(width=2000, height=2000))
    assert OCRProcessor._dpi_for(page) < 300


def test_process_builds_blocks_per_page(tmp_path, patched_ocr):
    pdf = tmp_path / "s.pdf"
    _make_pdf(pdf, pages=2)
    doc = OCRProcessor().process(str(pdf))
    assert doc.source["type"] == "scanned_pdf"
    assert len(doc.pages) == 2
    block = doc.pages[0]["blocks"][0]
    assert block["content"] == "识别文字"
    assert block["metadata"]["ocr_confidence"] == 0.95
    assert block["page_num"] == 1


def test_process_single_page_failure_degrades(tmp_path):
    import engines.parsing.ocr as ocr_mod

    class BoomOCR:
        def __init__(self):
            self.n = 0

        def __call__(self, img):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("ocr boom")
            return (OCR_RESULT, None)

    with patch("engines.parsing.ocr.RapidOCR", return_value=BoomOCR()):
        ocr_mod._ocr_instance = None
        pdf = tmp_path / "s.pdf"
        _make_pdf(pdf, pages=2)
        doc = OCRProcessor().process(str(pdf))
    assert len(doc.pages) == 1  # 失败页被丢弃, 仅成功页入库 (不 abort 整份)
    assert doc.pages[0]["blocks"][0]["content"] == "识别文字"


def test_process_empty_ocr_result_no_blocks(tmp_path, patched_ocr):
    import engines.parsing.ocr as ocr_mod

    with patch("engines.parsing.ocr.RapidOCR") as mc:
        mc.return_value = _FakeOCR([None])
        ocr_mod._ocr_instance = None
        pdf = tmp_path / "s.pdf"
        _make_pdf(pdf, pages=1, text="")
        doc = OCRProcessor().process(str(pdf))
    assert doc.pages[0]["blocks"] == []
