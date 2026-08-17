"""TXT 解析: 空行分段, 全为 paragraph 块 (编码自适应, 防 GBK 中文乱码)"""

import hashlib
import re
from pathlib import Path

from engines.parsing.models import UIRDocument
from engines.parsing.text_io import read_text_auto


class TxtParser:
    supported_types = (".txt",)

    def parse(self, file_path: str) -> UIRDocument:
        text = read_text_auto(file_path)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        blocks = [{"type": "paragraph", "bbox": [], "content": p, "page_num": 1, "metadata": {}} for p in paragraphs]
        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        return UIRDocument(
            doc_id=doc_id,
            source={"type": "txt", "path": file_path},
            pages=[{"page_num": 1, "blocks": blocks}],
            tables=[],
        )
