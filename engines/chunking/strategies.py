"""按文档类型选择切分策略 — 类型化工作流的核心环节。

策略:
  semantic : 结构感知语义分块 (StructureChunker, 默认, 适用于绝大多数文档)
  faq      : 问答对切分 (客服/FAQ — 保留 问/答 结构, 每对独立成块)
  clause   : 条款级切分 (合同/法务 — 按 第X条/第X章/数字. 边界切分)
  table    : 表格感知 (把解析出的表格单独成块, 可叠加于 semantic/clause)

工厂 get_chunker(spec, settings) 根据 DocTypeSpec 返回实现了 chunk(uir)->list[Chunk] 的对象。
所有策略产出统一的 Chunk 结构, 与下游检索/入库契约兼容。
"""

import logging
import re

from engines.interfaces import Chunk
from engines.chunking.structure_chunker import StructureChunker

logger = logging.getLogger(__name__)


class FaqChunker:
    """FAQ/话术切分: 识别 问/答 对, 每对独立成块, 保留问答结构。

    识别模式: "问：/答：", "Q:/A:", "问题：/回答：", "用户：/客服：" 等。
    无明确问答结构时退化语义分块, 保证不丢内容。
    """

    _QA_PAIRS = re.compile(
        r"(?:^|\n)\s*(?:问|Q|问题|用户|客户)[\s:：]+([\s\S]*?)\s*"
        r"(?:答|A|回答|话术|回复|客服)[\s:：]+([\s\S]*?)(?=(?:\n\s*(?:问|Q|问题|用户|客户)[\s:：])|\Z)",
        re.IGNORECASE,
    )

    def __init__(self, max_chars: int = 600, overlap: int = 80):
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, uir_doc) -> list[Chunk]:
        doc_id = uir_doc.doc_id
        text = StructureChunker._flatten_text(uir_doc)
        pairs = [(m.group(1).strip(), m.group(2).strip()) for m in self._QA_PAIRS.finditer(text)]
        if not pairs:
            # 无明确问答结构 → 退化语义分块
            return StructureChunker(self.max_chars, self.overlap).chunk(uir_doc)
        chunks = []
        for i, (q, a) in enumerate(pairs):
            content = f"问：{q}\n答：{a}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_faq_{i:04d}",
                    doc_id=doc_id,
                    content=content,
                    context={"doc_title": StructureChunker._doc_title(uir_doc)},
                    metadata={"char_count": len(content), "strategy": "faq"},
                )
            )
        logger.info("[%s] FAQ 切分: %d 问答对", doc_id, len(chunks))
        return chunks


class ClauseChunker:
    """条款级切分: 按 第X条/第X章/数字. 边界切分, 保留条款编号与上下文。

    适用于合同/协议/规章制度等强条款结构文档。
    """

    # 捕获组: re.split 保留条款头, 拼回对应段落 (避免丢失 "第X条" 标识)
    _CLAUSE_SPLIT = re.compile(
        r"(\n\s*(?:第[一二三四五六七八九十百千0-9]+[条章节目篇]\s|(?:[0-9]+(?:\.[0-9]+)+)[\s、.、]))"
    )

    def __init__(self, max_chars: int = 1200, overlap: int = 200):
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, uir_doc) -> list[Chunk]:
        doc_id = uir_doc.doc_id
        text = StructureChunker._flatten_text(uir_doc)
        segments = self._split_clauses(text)
        chunks = []
        for i, seg in enumerate(segments):
            seg = seg.strip()
            if not seg:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_clause_{i:04d}",
                    doc_id=doc_id,
                    content=seg,
                    context={"doc_title": StructureChunker._doc_title(uir_doc)},
                    metadata={"char_count": len(seg), "strategy": "clause"},
                )
            )
        logger.info("[%s] 条款切分: %d 段", doc_id, len(chunks))
        return chunks

    def _split_clauses(self, text: str) -> list[str]:
        """按条款头切分并保留条款头: re.split 捕获组 → [前导, 头, 体, 头, 体, ...]。"""
        parts = self._CLAUSE_SPLIT.split(text)
        out: list[str] = []
        preamble = (parts[0] or "").strip()
        if preamble:
            out.append(preamble)
        i = 1
        n = len(parts)
        while i < n:
            head = (parts[i] or "").strip()
            body = (parts[i + 1] if i + 1 < n else "").strip()
            if head or body:
                out.append(head + body)
            i += 2
        # 超长段二次硬切 (条款正文过长时)
        merged: list[str] = []
        for seg in out:
            if len(seg) <= self.max_chars:
                merged.append(seg)
            else:
                merged.extend(self._hard_split(seg))
        return merged

    def _hard_split(self, text: str) -> list[str]:
        step = max(self.max_chars, 1)
        return [text[i : i + self.max_chars] for i in range(0, len(text), step)]


class _TableAware:
    """包装器: 在基础切分结果上追加表格块 (解析出的表格单独成块, 不再丢弃)。"""

    def __init__(self, base):
        self.base = base

    def chunk(self, uir_doc) -> list[Chunk]:
        chunks = self.base.chunk(uir_doc)
        tables = getattr(uir_doc, "tables", None) or []
        for i, t in enumerate(tables):
            md = t.get("markdown") or t.get("text") or self._cells_to_text(t)
            if not md:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{uir_doc.doc_id}_tbl_{i:04d}",
                    doc_id=uir_doc.doc_id,
                    content=md,
                    context={"doc_title": StructureChunker._doc_title(uir_doc), "is_table": True},
                    metadata={"char_count": len(md), "strategy": "table"},
                )
            )
        if tables:
            logger.info("[%s] 表格感知: 追加 %d 个表格块", uir_doc.doc_id, len(tables))
        return chunks

    @staticmethod
    def _cells_to_text(t: dict) -> str:
        rows = t.get("rows") or t.get("cells") or []
        if not rows:
            return ""
        return "\n".join(" | ".join(str(c) for c in row) for row in rows)


def get_chunker(spec, settings) -> object:
    """根据 DocTypeSpec.chunker 配置返回切分器。

    spec: DocTypeSpec; settings: 全局配置 (chunk_max_chars/chunk_overlap 兜底)。
    支持 strategy: semantic|faq|clause, 以及 parse.table 叠加表格块。
    """
    cfg = spec.chunker or {}
    strategy = cfg.get("strategy", "semantic")
    max_chars = cfg.get("max_chars") or getattr(settings, "chunk_max_chars", 800)
    overlap = cfg.get("overlap") or getattr(settings, "chunk_overlap", 128)
    table_aware = bool((spec.parse or {}).get("table"))

    if strategy == "faq":
        base = FaqChunker(max_chars, overlap)
    elif strategy == "clause":
        base = ClauseChunker(max_chars, overlap)
    else:
        base = StructureChunker(max_chars, overlap)

    return _TableAware(base) if table_aware else base
