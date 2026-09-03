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
            doc = self._parse_native(file_path)
            # 原生文本疑似乱码(字体编码损坏, 非扫描件) → 转 OCR 重抽。
            # 此类 PDF 每页字符数达标被放行native路径, 但文字层是乱码字形,
            # 渲染成像素后 OCR 反而能正确识别。
            if self._is_garbled(doc):
                logger.warning("原生文本疑似乱码(字体编码损坏), 转 OCR 重抽: %s", file_path)
                try:
                    return self._parse_scanned(file_path)
                except Exception as e:  # noqa: BLE001 — OCR 失败保留原生(乱码)结果, 不中断入库
                    logger.warning("OCR 重抽失败, 保留原生(乱码)结果: %s", str(e)[:120])
                    return doc
            return doc
        else:
            return self._parse_scanned(file_path)

    # 乱码判定: 字体编码损坏时, 字符散落于非常用 Unicode 区块
    # (控制字符 / CJK 扩展A / 私有区 PUA / 替换符 U+FFFD / 扩展B+),
    # 而常用汉字块(U+4E00-9FFF)占比极低。真实中文文档反之。
    @staticmethod
    def _looks_garbled(text: str) -> bool:
        if not text or len(text) < 40:
            return False
        common = 0
        suspicious = 0
        for ch in text:
            o = ord(ch)
            if 0x4E00 <= o <= 0x9FFF:
                common += 1
            elif (0x00 <= o <= 0x08) or (0x3400 <= o <= 0x4DBF) or (0xE000 <= o <= 0xF8FF) or o == 0xFFFD or (0x20000 <= o):
                suspicious += 1
        total_cjk = common + suspicious
        if total_cjk < 20:
            return False  # 中文内容太少, 不判定(英文/数字为主文档不乱码)
        # 真实中文: 常用字块占绝对多数; 乱码: 异块字符过半
        return suspicious / total_cjk > 0.5

    def _is_garbled(self, doc: "UIRDocument") -> bool:
        text = " ".join(b.get("content", "") for p in doc.pages for b in p.get("blocks", []))
        return self._looks_garbled(text)

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
        """扫描件走 OCR。OCR 引擎不可用(如容器缺 libxcb 等系统库)时降级为原生文本提取,
        避免整份文档入库崩溃 — OCR 失败至少还能拿到文本层(若有), 优于直接抛异常中断。"""
        try:
            from engines.parsing.ocr import OCRProcessor

            return OCRProcessor().process(file_path)
        except Exception as e:  # noqa: BLE001 — OCR 不可用降级边界: 退回原生提取, 不中断入库
            logger.warning("OCR 不可用, 降级原生文本提取: %s", str(e)[:120])
            return self._parse_native(file_path)

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
