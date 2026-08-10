"""Markdown 解析: # 标题 → title block, 空行分段"""
import hashlib
import re
from pathlib import Path

from engines.parsing.models import UIRDocument

_TITLE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownParser:
    supported_types = [".md", ".markdown"]

    def parse(self, file_path: str) -> UIRDocument:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        raw_paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        blocks = []
        for para in raw_paras:
            cleaned = "\n".join(line.strip() for line in para.splitlines() if line.strip())
            m = _TITLE_RE.match(cleaned)
            if m:
                blocks.append({
                    "type": "title", "bbox": [], "content": m.group(2),
                    "page_num": 1, "metadata": {"heading_level": len(m.group(1))},
                })
            else:
                blocks.append({
                    "type": "paragraph", "bbox": [], "content": cleaned,
                    "page_num": 1, "metadata": {},
                })
        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        return UIRDocument(doc_id=doc_id, source={"type": "markdown", "path": file_path},
                           pages=[{"page_num": 1, "blocks": blocks}], tables=[])
