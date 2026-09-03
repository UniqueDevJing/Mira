"""HTML 解析: bs4 抽取标题/段落/表格 → 统一 UIRDocument。

依赖 beautifulsoup4 (lazy import: 未安装时仅 parse() 报错, 不影响模块导入与其他 parser)。
去噪: 跳过 script/style/head/nav/footer/noscript 等无正文区; 保留标题层级、表格结构、链接文本。
"""

import hashlib
import logging
from pathlib import Path

from engines.parsing.models import UIRDocument

logger = logging.getLogger(__name__)

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


class HTMLParser:
    supported_types = (".html", ".htm")

    def parse(self, file_path: str) -> UIRDocument:
        from bs4 import BeautifulSoup

        raw = Path(file_path).read_bytes()
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "head", "nav", "footer", "noscript", "template"]):
            tag.decompose()

        doc_id = hashlib.sha256(raw).hexdigest()[:16]
        blocks: list[dict] = []
        tables: list[dict] = []

        def _flush_para(lines: list[str]) -> None:
            text = "\n".join(lines).strip()
            if text:
                blocks.append({"type": "paragraph", "bbox": [], "content": text, "page_num": 1, "metadata": {}})

        para: list[str] = []
        for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
            if el.name in _HEADING_TAGS:
                _flush_para(para)
                para = []
                blocks.append(
                    {
                        "type": "title",
                        "bbox": [],
                        "content": el.get_text(" ", strip=True),
                        "page_num": 1,
                        "metadata": {"heading_level": _HEADING_TAGS[el.name]},
                    }
                )
            elif el.name in ("p", "li"):
                txt = el.get_text(" ", strip=True)
                if txt:
                    para.append(txt)
            elif el.name == "table":
                rows = [
                    [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                    for tr in el.find_all("tr")
                ]
                rows = [r for r in rows if r]
                if rows:
                    tables.append({"page_num": 1, "bbox": [], "matrix": rows, "headers": rows[0]})
                    # 表格同时作为简表段落, 便于被检索命中
                    preview = " | ".join(rows[0]) + " ; " + " / ".join(
                        " | ".join(r) for r in rows[1 : min(len(rows), 9)]
                    )
                    blocks.append(
                        {"type": "paragraph", "bbox": [], "content": preview, "page_num": 1, "metadata": {"is_table": True}}
                    )
        _flush_para(para)

        update_time = int(Path(file_path).stat().st_mtime)
        return UIRDocument(
            doc_id=doc_id,
            source={"type": "html", "path": file_path},
            pages=[{"page_num": 1, "blocks": blocks}],
            tables=tables,
            update_time=update_time,
        )
