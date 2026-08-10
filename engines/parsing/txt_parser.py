"""TXT 解析: 空行分段, 全为 paragraph 块"""
import hashlib
import re
from pathlib import Path

from engines.parsing.models import UIRDocument


class TxtParser:
    supported_types = [".txt"]

    def parse(self, file_path: str) -> UIRDocument:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        blocks = [
            {"type": "paragraph", "bbox": [], "content": p, "page_num": 1, "metadata": {}}
            for p in paragraphs
        ]
        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        return UIRDocument(doc_id=doc_id, source={"type": "txt", "path": file_path},
                           pages=[{"page_num": 1, "blocks": blocks}], tables=[])
