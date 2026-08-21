"""基于文档结构的统一分块: 标题边界 + 递归字符回退"""

import logging
import re

from engines.interfaces import Chunk

logger = logging.getLogger(__name__)

_CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class RecursiveTextSplitter:
    """递归字符切分: 按分隔符优先级逐级切，块超长则用下一级分隔符递归。

    自写而非 langchain — 项目 design-decisions 拒绝 LangChain 抽象层，且无此依赖。
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 128, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or _CHINESE_SEPARATORS

    def split_text(self, text: str) -> list[str]:
        # overlap 只在此处应用一次, 避免递归层内复合
        return self._apply_overlap(self._split(text, self.separators))

    def _split(self, text: str, seps: list[str]) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        if not seps or seps[0] == "":
            return self._hard_split(text)
        sep = seps[0]
        chunks, current = [], ""
        for piece in self._split_keepends(text, sep):
            if len(piece) > self.chunk_size:
                # 超长 piece: 先 flush 当前块, 再用下一级分隔符递归
                current = self._flush(chunks, current)
                current = self._merge_split(current, self._split(piece, seps[1:]), chunks)
            else:
                current = self._push(current, piece, chunks)
        self._flush(chunks, current)
        return chunks

    def _push(self, current: str, piece: str, chunks: list[str]) -> str:
        """piece 放得下并入当前块, 放不下先 flush 当前块再新开。"""
        if current and len(current) + len(piece) <= self.chunk_size:
            return current + piece
        self._flush(chunks, current)
        return piece

    def _flush(self, chunks: list[str], buf: str) -> str:
        """把缓冲块 push 进结果, 返回空缓冲。"""
        if buf:
            chunks.append(buf)
        return ""

    def _merge_split(self, current: str, subs: list[str], chunks: list[str]) -> str:
        """把递归子切结果逐个并入缓冲 (等价于逐 piece 走 _push)。"""
        for sub in subs:
            current = self._push(current, sub, chunks)
        return current

    @staticmethod
    def _split_keepends(text: str, sep: str) -> list[str]:
        parts = re.split(f"({re.escape(sep)})", text)
        pieces = []
        for i in range(0, len(parts) - 1, 2):
            pieces.append(parts[i] + parts[i + 1])
        if len(parts) % 2 == 1 and parts[-1]:
            pieces.append(parts[-1])
        return [p for p in pieces if p]

    def _hard_split(self, text: str) -> list[str]:
        """无分隔符可用: 按 chunk_size 硬切, 不含 overlap — 由顶层 _apply_overlap 统一加一次"""
        step = max(self.chunk_size, 1)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        out = [chunks[0]]
        for c in chunks[1:]:
            out.append(out[-1][-self.chunk_overlap :] + c)
        return out


class StructureChunker:
    def __init__(self, max_chars: int = 800, overlap: int = 128):
        self.max_chars = max_chars
        self.overlap = overlap
        self._splitter = RecursiveTextSplitter(
            chunk_size=max_chars, chunk_overlap=overlap, separators=_CHINESE_SEPARATORS
        )

    def chunk(self, uir_doc) -> list[Chunk]:
        blocks = self._flatten_blocks(uir_doc)
        if not blocks:
            return []
        chains = self._heading_chains(blocks)
        segments = self._split_segments(blocks)
        doc_title = self._doc_title(uir_doc)
        chunks = []
        for start_idx, seg in segments:
            title_chain = chains[start_idx]
            # 纯标题段 (标题后紧跟标题/文档尾): 不产"仅标题"空 chunk, 标题由下一段承接
            if not any(b["type"] == "paragraph" for b in seg):
                continue
            content = "\n\n".join(b["content"] for b in seg)
            if len(content) <= self.max_chars:
                chunks.append(self._make_chunk(uir_doc.doc_id, content, seg, title_chain, doc_title, len(chunks)))
            else:
                for part in self._splitter.split_text(content):
                    chunks.append(self._make_chunk(uir_doc.doc_id, part, seg, title_chain, doc_title, len(chunks)))
        return chunks

    @staticmethod
    def _doc_title(uir_doc) -> str:
        """从 source.path 取文件名作文档标题 (原实现误存 doc_id)。"""
        src = getattr(uir_doc, "source", None) or {}
        path = src.get("path", "") or ""
        if path:
            return path.replace("\\", "/").rsplit("/", 1)[-1]
        return ""

    @staticmethod
    def _flatten_text(uir_doc) -> str:
        """拼接所有段落/标题文本 (条款/问答切分复用)。"""
        return "\n".join(
            b["content"]
            for page in uir_doc.pages
            for b in page["blocks"]
            if b["type"] in ("paragraph", "title")
        )

    @staticmethod
    def _flatten_blocks(uir_doc) -> list[dict]:
        blocks = []
        for page in uir_doc.pages:
            for block in page["blocks"]:
                if block["type"] in ("paragraph", "title"):
                    blocks.append(block)
        return blocks

    @staticmethod
    def _heading_chains(blocks: list[dict]) -> list[list[str]]:
        stack = []  # [(level, text)]
        chains = []
        for block in blocks:
            if block["type"] == "title":
                level = block.get("metadata", {}).get("heading_level")
                if level is None:
                    level = StructureChunker._estimate_level(block)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, block["content"]))
            chains.append([t for _, t in stack])
        return chains

    @staticmethod
    def _split_segments(blocks: list[dict]) -> list[tuple[int, list[dict]]]:
        segments = []
        start = 0
        for i, block in enumerate(blocks):
            if i > 0 and block["type"] == "title":
                segments.append((start, blocks[start:i]))
                start = i
        segments.append((start, blocks[start:]))
        return segments

    @staticmethod
    def _estimate_level(block: dict) -> int:
        """PDF 无 heading_level 时的兜底: font_size + 加粗 + 编号正则"""
        md = block.get("metadata", {})
        size = md.get("font_size", 12)
        if md.get("is_bold") or re.match(r"^第[一二三四五六七八九十百千]+[章节篇]", block["content"]):
            size = max(size, 16)
        if size >= 22:
            return 1
        elif size >= 18:
            return 2
        elif size >= 14:
            return 3
        return 4

    @staticmethod
    def _make_chunk(
        doc_id: str, content: str, segment: list[dict], title_chain: list[str], doc_title: str, index: int
    ) -> Chunk:
        first, last = segment[0], segment[-1]
        return Chunk(
            chunk_id=f"{doc_id}_chunk_{index:04d}",
            doc_id=doc_id,
            content=content.strip(),
            context={"title_chain": title_chain, "doc_title": doc_title or doc_id},
            metadata={
                "page_range": [first.get("page_num", 1), last.get("page_num", 1)],
                "char_count": len(content),
                "paragraph_count": len(segment),
            },
        )
