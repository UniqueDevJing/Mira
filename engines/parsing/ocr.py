"""OCR 处理器：基于 RapidOCR (onnxruntime) 的扫描件文本提取

替代原 PaddleOCR: paddlepaddle 不支持 Python 3.14, 且 2.x→3.x API 断裂。
RapidOCR 纯 onnxruntime CPU 底座, Py 版本兼容好, 中文识别精度相当。
"""

import hashlib
import logging
import threading
from pathlib import Path

import fitz
import numpy as np
from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger(__name__)

# 模块级单例: onnxruntime 会话加载耗时数百 ms~秒级, 每文档重建浪费
# (原实现每 OCRProcessor 实例新建会话)。onnxruntime session 并发推理线程安全。
_ocr_instance = None
_ocr_lock = threading.Lock()


def _get_ocr() -> RapidOCR:
    global _ocr_instance
    if _ocr_instance is None:
        with _ocr_lock:  # 双重检查: 并发首请求不重复实例化 (会话初始化数百 ms~秒级)
            if _ocr_instance is None:
                logger.info("RapidOCR 初始化: CPU 模式 (模块级单例)")
                _ocr_instance = RapidOCR()
    return _ocr_instance


class OCRProcessor:
    def __init__(self, lang: str = "ch", use_gpu: bool = False):
        if use_gpu:
            logger.warning("RapidOCR 仅 CPU 模式, use_gpu 参数被忽略")
        # lang 仅作兼容参数保留; RapidOCR 自带中英模型, 无需指定
        self.ocr = _get_ocr()

    @staticmethod
    def _dpi_for(page) -> int:
        """大版面页动态降 dpi — 300dpi 下 A0/A2 页面渲染数百 MB pixmap, 后台线程 OOM。
        (50MB 上传限制不约束解压后页面尺寸, 需像素级保护)"""
        dpi = 300
        max_pix = 25_000_000  # ~5000x5000 上限
        while dpi > 72 and (page.rect.width * dpi / 72) * (page.rect.height * dpi / 72) > max_pix:
            dpi //= 2
        return dpi

    def process(self, file_path: str):
        from engines.parsing.models import UIRDocument

        pages = []
        tables = []

        # with 保证异常路径也释放文件句柄 (原 doc.close() 只在正常流末尾执行, 异常即泄漏)
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, 1):
                # 逐页降级: 单页 OCR 失败跳过并记日志, 不 abort 整份 (扫描件个别页损坏是常态,
                # 全部成功的页仍产出文本, 而非整份标记 failed)
                try:
                    pix = page.get_pixmap(dpi=self._dpi_for(page))
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    # 原 PaddleOCR 版有自适应二值化预处理; RapidOCR 内部检测/识别期望自然图像,
                    # 二值化会丢信息, 直接喂 RGB
                    result, _ = self.ocr(img)
                    blocks = []
                    if result:
                        for box, text, confidence in result:
                            xs = [p[0] for p in box]
                            ys = [p[1] for p in box]
                            blocks.append(
                                {
                                    "type": "paragraph",
                                    "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                                    "content": text,
                                    "page_num": page_num,
                                    "metadata": {"ocr_confidence": float(confidence)},
                                }
                            )
                    pages.append({"page_num": page_num, "blocks": blocks})
                except Exception as e:  # noqa: BLE001 — 单页降级边界
                    logger.warning("扫描件第 %d 页 OCR 失败, 跳过: %s", page_num, str(e)[:120])

        if not pages:
            logger.warning("扫描件全部页 OCR 失败, 返回空文档: %s", Path(file_path).name)

        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        return UIRDocument(doc_id=doc_id, source={"type": "scanned_pdf", "path": file_path}, pages=pages, tables=tables)
