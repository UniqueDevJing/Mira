"""Markdown 解析: 行首 # 标题 → title block, 空行分段, 代码围栏内不识别标题。

与 spec 一致: 标题按"行首 #{1,6} "判定 (原实现按"段落首行"匹配,
"# 标题\\n正文"(无空行) 会整段判为段落, # 字面量混入正文)。
代码围栏 (``` / ~~~) 内一律按段落处理, 不误判 # 开头的代码注释为标题。
"""

import hashlib
import re
from pathlib import Path

from engines.parsing.models import UIRDocument
from engines.parsing.text_io import read_text_auto

_TITLE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


class MarkdownParser:
    supported_types = (".md", ".markdown")

    def parse(self, file_path: str) -> UIRDocument:
        text = read_text_auto(file_path)
        blocks = []
        para_lines: list[str] = []
        in_fence = False

        def flush_para():
            nonlocal para_lines
            if para_lines:
                blocks.append(
                    {
                        "type": "paragraph",
                        "bbox": [],
                        "content": "\n".join(para_lines).strip(),
                        "page_num": 1,
                        "metadata": {},
                    }
                )
                para_lines = []

        for raw in text.splitlines():
            line = raw.strip()
            # fence 内空行不切断代码块 (原实现空行 flush 在 in_fence 判断前, 代码块被误切成多段)
            if not line and not in_fence:
                flush_para()
                continue
            if _FENCE_RE.match(line):
                flush_para()
                in_fence = not in_fence
                continue
            if not in_fence:
                m = _TITLE_RE.match(line)
                if m:
                    flush_para()
                    blocks.append(
                        {
                            "type": "title",
                            "bbox": [],
                            "content": m.group(2).strip(),
                            "page_num": 1,
                            "metadata": {"heading_level": len(m.group(1))},
                        }
                    )
                    continue
            para_lines.append(line)
        flush_para()

        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        return UIRDocument(
            doc_id=doc_id,
            source={"type": "markdown", "path": file_path},
            pages=[{"page_num": 1, "blocks": blocks}],
            tables=[],
        )
