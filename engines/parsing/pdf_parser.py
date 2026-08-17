"""电子原生 PDF 解析，PyMuPDF + PDFPlumber 双引擎"""

import hashlib
import logging
import re
from pathlib import Path

import fitz
import pdfplumber

from engines.parsing.models import UIRDocument

logger = logging.getLogger(__name__)


class PDFParser:
    supported_types = (".pdf",)

    def parse(self, file_path: str) -> UIRDocument:
        pdf_type = self._detect_type(file_path)
        if pdf_type == "native":
            return self._parse_native(file_path)
        else:
            return self._parse_scanned(file_path)

    # 扫描件判定阈值: 平均每页文本字符数低于此值才走 OCR (类常量, 便于测试覆盖/调整)
    SCANNED_AVG_CHARS_THRESHOLD = 30

    def _detect_type(self, file_path: str) -> str:
        with fitz.open(file_path) as doc:
            if doc.page_count == 0:
                return "native"
            sample = min(doc.page_count, 3)
            total = sum(len(page.get_text()) for page in doc[:sample])
            avg = total / sample
        return "scanned" if avg < self.SCANNED_AVG_CHARS_THRESHOLD else "native"

    def _parse_native(self, file_path: str) -> UIRDocument:
        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        pages = []
        tables = []

        with fitz.open(file_path) as mupdf_doc:
            for page_num, page in enumerate(mupdf_doc, 1):
                page_height = page.rect.height  # 实际页面高度
                blocks = []
                for block in page.get_text("dict")["blocks"]:
                    if block["type"] != 0:
                        continue
                    for line in block["lines"]:
                        text = "".join(span["text"] for span in line["spans"])
                        if not text.strip():
                            continue
                        spans = line["spans"]
                        # 行级分类: 每行自含 bbox/字号/加粗, 避免多行标题被腰斩成多个 title、
                        # 或正文块首行加粗导致整块误判标题 (原实现按整块判定但按行产出, 粒度错位)
                        line_bbox = list(line.get("bbox") or block["bbox"])
                        blocks.append(
                            {
                                "type": self._classify_line(text.strip(), line_bbox, spans, page_height),
                                "bbox": line_bbox,
                                "content": text.strip(),
                                "page_num": page_num,
                                "metadata": {
                                    "font_size": spans[0]["size"] if spans else 0,
                                    "is_bold": bool(spans[0]["flags"] & 16) if spans else False,
                                },
                            }
                        )
                pages.append({"page_num": page_num, "blocks": blocks})

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # 逐页隔离: 畸形页 extract_tables 抛异常时跳过该页, 不 abort 整份 (文本已提取不受影响)
                try:
                    for t in page.extract_tables():
                        if t:
                            tables.append(
                                {
                                    "page_num": page_num,
                                    "bbox": list(page.bbox),
                                    "matrix": t,
                                    "headers": t[0] if t else [],
                                }
                            )
                except Exception as e:  # noqa: BLE001 — 单页降级边界
                    logger.warning("PDF 第 %d 页表格提取失败, 跳过: %s", page_num, str(e)[:120])

        return UIRDocument(doc_id=doc_id, source={"type": "pdf", "path": file_path}, pages=pages, tables=tables)

    def _parse_scanned(self, file_path: str) -> UIRDocument:
        from engines.parsing.ocr import OCRProcessor

        return OCRProcessor().process(file_path)

    _NUMBERED_TITLE_RE = re.compile(r"^(第[一二三四五六七八九十百千]+[章节篇]|(\d+(\.\d+)*)[、\s.])")

    def _classify_block(self, block: dict, page_height: float = 842) -> str:
        """兼容入口: 从块提取首行字号/加粗 + 整块文本, 委托行级分类 (测试与旧调用方)。"""
        spans = block["lines"][0]["spans"] if block.get("lines") else []
        text = "".join(s["text"] for line in block.get("lines", []) for s in line["spans"])
        return self._classify_line(text, block["bbox"], spans, page_height)

    def _classify_line(self, text: str, bbox: list, spans: list, page_height: float = 842) -> str:
        """按行分类页眉/页脚/标题/正文 — 行级 bbox/字号/加粗自包含。

        page_height 默认 A4 (842pt)，由调用方传入实际页面高度。
        """
        if bbox[1] < 50:
            return "header"
        elif bbox[1] > page_height - 50:
            return "footer"
        size = spans[0]["size"] if spans else 0
        bold = bool(spans[0]["flags"] & 16) if spans else False
        if size > 16 or bold or self._NUMBERED_TITLE_RE.match(text):
            return "title"
        return "paragraph"
