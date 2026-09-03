"""代码整块切分器 — 父块(整段代码) + 子块(函数/类/模块)成对产出, 支持父子检索。

设计目标 (视频要求「代码按函数/类/模块整块保留」):
- 代码块内的函数/类/模块不被中段截断, 每个顶层定义独立成块(子块)。
- 同时产出一个父块(整段代码), 子块带 parent_id 指向父块, 供检索命中子块后
  回父块大上下文给 LLM (父子文档机制)。
- 非代码块(段落/标题)委托 StructureChunker, 行为与 semantic 切分一致。

代码识别: 不依赖解析器打标(避免改动 markdown_parser 引发其他策略回归),
改用内容启发式 — 含 def/class/function/const/import/=>/{}/缩进声明行比例高的块判为代码。
"""

import logging
import re

from engines.chunking.structure_chunker import StructureChunker
from engines.interfaces import Chunk
from engines.parsing.models import UIRDocument

logger = logging.getLogger(__name__)

# 单元边界: 顶层定义起点 (用于切分, 不把 import/配置语句拆成碎块)
_START_LINE = re.compile(
    r"""^\s*(def|class|async\s+def|function|public|private|protected|internal|export|
         const|let|var|func|interface|struct|enum|impl|trait)\b""",
    re.VERBOSE,
)
# 代码检测: 更宽, 含 import/from/using 等语句 (用于判断"这是不是代码块")
_CODE_LINE = re.compile(
    r"""^\s*(def|class|async\s+def|function|public|private|protected|internal|export|
         const|let|var|func|interface|struct|enum|impl|trait|import|from|using|
         package|namespace|require|include|select|create\s+table|<\?php)\b""",
    re.VERBOSE,
)
_BRACE_DECL = re.compile(r"^\s*\w[\w<>,\s*]*\s+\w+\s*\(")  # C/Java 风格函数声明
_CODE_TOKEN = re.compile(r"(=>|&&|\|\||!=|==|\bconst\b|;\s*$|^\s*\{|^\s*\})")


class CodeChunker:
    def __init__(self, max_chars: int = 800, overlap: int = 80):
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, uir_doc) -> list[Chunk]:
        doc_id = uir_doc.doc_id
        doc_title = StructureChunker._doc_title(uir_doc)
        update_time = getattr(uir_doc, "update_time", 0)
        blocks = self._flatten_blocks(uir_doc)

        noncode: list[str] = []
        chunks: list[Chunk] = []
        parent_idx = 0
        for content, page_num, is_code, lang in blocks:
            if not is_code:
                noncode.append(content)
                continue
            units = self._split_code(content, lang)
            # 整段即一个单元: 无需父子, 直接单块(避免无意义的自引用父)
            if len(units) == 1 and units[0].strip() == content.strip():
                chunks.append(self._mk(doc_id, doc_title, update_time, content, page_num, lang,
                                       chunk_id=f"{doc_id}_code_{parent_idx:03d}",
                                       parent_id="", is_parent=True))
                parent_idx += 1
                continue
            pid = f"{doc_id}_code_p{parent_idx:03d}"
            parent_idx += 1
            # 父块 = 整段代码 (检索命中子块后回此大上下文)
            chunks.append(self._mk(doc_id, doc_title, update_time, content, page_num, lang,
                                   chunk_id=pid, parent_id="", is_parent=True))
            for ci, unit in enumerate(units):
                chunks.append(self._mk(doc_id, doc_title, update_time, unit, page_num, lang,
                                       chunk_id=f"{pid}_c{ci:03d}",
                                       parent_id=pid, is_parent=False))
        # 非代码块委托 StructureChunker, 保留语义切分行为
        if noncode:
            synth = UIRDocument(
                doc_id=doc_id,
                source=dict(getattr(uir_doc, "source", {}) or {}),
                pages=[{"page_num": 1, "blocks": [
                    {"type": "paragraph", "bbox": [], "content": t, "page_num": 1, "metadata": {}}
                    for t in noncode
                ]}],
                tables=getattr(uir_doc, "tables", None) or [],
                update_time=update_time,
            )
            chunks.extend(StructureChunker(self.max_chars, self.overlap).chunk(synth))
        logger.info("[%s] 代码切分: 父块=%d 子块合计=%d", doc_id, parent_idx,
                    sum(1 for c in chunks if not c.metadata.get("is_parent")))
        return chunks

    def _mk(self, doc_id, doc_title, update_time, content, page_num, lang,
            parent_id, is_parent, chunk_id=None) -> Chunk:
        cid = chunk_id or f"{doc_id}_code_{abs(hash(content)) % 100000:05d}"
        return Chunk(
            chunk_id=cid,
            doc_id=doc_id,
            content=content.strip(),
            context={"doc_title": doc_title, "is_code": True, "language": lang},
            metadata={
                "char_count": len(content),
                "strategy": "code",
                "is_parent": is_parent,
                "parent_id": parent_id,
                "language": lang,
                "page_range": [page_num, page_num],
                "file_name": doc_title or doc_id,
                "update_time": update_time,
            },
        )

    @staticmethod
    def _flatten_blocks(uir_doc):
        out = []
        for page in getattr(uir_doc, "pages", []):
            for b in page.get("blocks", []):
                if b.get("type") not in ("paragraph", "title", "code"):
                    continue
                content = b.get("content", "") or ""
                if not content.strip():
                    continue
                lang = (b.get("metadata") or {}).get("language", "")
                out.append((content, b.get("page_num", 1), CodeChunker._looks_like_code(content), lang))
        return out

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return False
        code_lines = 0
        for ln in lines:
            if _CODE_LINE.match(ln) or _BRACE_DECL.match(ln) or _CODE_TOKEN.search(ln):
                code_lines += 1
        return code_lines >= max(2, len(lines) * 0.3)

    def _split_code(self, content: str, lang: str) -> list[str]:
        lines = content.splitlines()
        fam = self._lang_family(lang)
        if fam == "python":
            bounds = self._split_python(lines)
        elif fam == "brace":
            bounds = self._split_brace(lines)
        else:
            bounds = self._split_generic(lines)
        units = []
        for s, e in bounds:
            unit = "\n".join(lines[s:e]).strip()
            if unit:
                units.append(unit)
        return units or [content.strip()]

    @staticmethod
    def _lang_family(lang: str) -> str:
        lang = (lang or "").lower()
        if lang in ("py", "python", "py3"):
            return "python"
        if lang in ("js", "javascript", "ts", "typescript", "java", "c", "cpp", "c++",
                    "go", "golang", "rs", "rust", "cs", "csharp", "php", "swift", "kt", "kotlin"):
            return "brace"
        return "auto"

    @staticmethod
    def _split_python(lines):
        starts = [i for i, ln in enumerate(lines)
                  if (ln[:1] not in (" ", "\t")) and re.match(r"^(async\s+def|def|class)\b", ln)]
        return CodeChunker._bounds_from_starts(lines, starts)

    @staticmethod
    def _split_brace(lines):
        starts = []
        for i, ln in enumerate(lines):
            if ln[:1] in (" ", "\t"):
                continue
            if _START_LINE.match(ln) or _BRACE_DECL.match(ln):
                starts.append(i)
        return CodeChunker._bounds_from_starts(lines, starts)

    @staticmethod
    def _split_generic(lines):
        starts = [0]
        for i, ln in enumerate(lines):
            if i > 0 and not ln.strip() and lines[i - 1].strip():
                starts.append(i)
        return CodeChunker._bounds_from_starts(lines, starts)

    @staticmethod
    def _bounds_from_starts(lines, starts):
        starts = sorted(set(starts))
        # 覆盖文件头(import/模块文档)到首个定义之间的内容, 避免丢弃
        if starts and starts[0] > 0:
            starts = [0] + starts
        if not starts:
            return [(0, len(lines))]
        bounds = []
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else len(lines)
            bounds.append((s, e))
        return bounds
