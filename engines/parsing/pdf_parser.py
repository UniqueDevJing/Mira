"""电子原生 PDF 解析，PyMuPDF + PDFPlumber 双引擎"""
import hashlib
import re
from pathlib import Path

import fitz
import pdfplumber

from engines.parsing.models import UIRDocument


class PDFParser:
    supported_types = (".pdf",)

    def parse(self, file_path: str) -> UIRDocument:
        pdf_type = self._detect_type(file_path)
        if pdf_type == "native":
            return self._parse_native(file_path)
        else:
            return self._parse_scanned(file_path)

    def _detect_type(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text_chars = sum(len(page.get_text()) for page in doc[:3])
        doc.close()
        return "scanned" if text_chars < 100 else "native"

    def _parse_native(self, file_path: str) -> UIRDocument:
        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        pages = []
        tables = []

        mupdf_doc = fitz.open(file_path)
        for page_num, page in enumerate(mupdf_doc, 1):
            page_height = page.rect.height  # 实际页面高度
            blocks = []
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 0:
                    for line in block["lines"]:
                        text = "".join([span["text"] for span in line["spans"]])
                        if text.strip():
                            blocks.append({
                                "type": self._classify_block(block, page_height),
                                "bbox": list(block["bbox"]),
                                "content": text.strip(),
                                "page_num": page_num,
                                "metadata": {
                                    "font_size": line["spans"][0]["size"] if line["spans"] else 0,
                                    "is_bold": bool(line["spans"][0]["flags"] & 16) if line["spans"] else False,
                                }
                            })
            pages.append({"page_num": page_num, "blocks": blocks})

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                for t in page.extract_tables():
                    if t:
                        tables.append({
                            "page_num": page_num,
                            "bbox": list(page.bbox),
                            "matrix": t,
                            "headers": t[0] if t else []
                        })

        mupdf_doc.close()
        return UIRDocument(doc_id=doc_id, source={"type": "pdf", "path": file_path},
                           pages=pages, tables=tables)

    def _parse_scanned(self, file_path: str) -> UIRDocument:
        from engines.parsing.ocr import OCRProcessor
        return OCRProcessor().process(file_path)

    _NUMBERED_TITLE_RE = re.compile(r"^(第[一二三四五六七八九十百千]+[章节篇]|(\d+(\.\d+)*)[、\s.])")

    def _classify_block(self, block: dict, page_height: float = 842) -> str:
        """根据位置/字号/加粗/编号分类文本块。

        page_height 默认 A4 (842pt)，由调用方传入实际页面高度。
        """
        bbox = block["bbox"]
        if bbox[1] < 50:
            return "header"
        elif bbox[1] > page_height - 50:
            return "footer"
        spans = block["lines"][0]["spans"] if block.get("lines") else []
        size = spans[0]["size"] if spans else 0
        bold = bool(spans[0]["flags"] & 16) if spans else False
        text = "".join(s["text"] for line in block.get("lines", []) for s in line["spans"])
        if size > 16 or bold or self._NUMBERED_TITLE_RE.match(text):
            return "title"
        return "paragraph"
