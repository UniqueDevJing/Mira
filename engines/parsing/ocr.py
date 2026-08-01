"""OCR 处理器：基于 PaddleOCR 的扫描件文本提取"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import fitz
from paddleocr import PaddleOCR


class OCRProcessor:
    def __init__(self, lang: str = "ch"):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=True, show_log=False)

    def process(self, file_path: str):
        from engines.parsing.pdf_parser import UIRDocument

        doc = fitz.open(file_path)
        pages = []
        tables = []

        for page_num, page in enumerate(doc, 1):
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img = self._preprocess(img)

            result = self.ocr.ocr(img, cls=True)
            blocks = []
            if result and result[0]:
                for line in result[0]:
                    bbox_points = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    blocks.append({
                        "type": "paragraph",
                        "bbox": [min(xs), min(ys), max(xs), max(ys)],
                        "content": text,
                        "page_num": page_num,
                        "metadata": {"ocr_confidence": confidence}
                    })
            pages.append({"page_num": page_num, "blocks": blocks})

        doc.close()
        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        return UIRDocument(doc_id=doc_id, source={"type": "scanned_pdf", "path": file_path},
                           pages=pages, tables=tables)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return cv2.medianBlur(binary, 3)
