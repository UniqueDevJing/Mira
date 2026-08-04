"""基于文档结构的动态语义分块"""
from typing import List
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from engines.interfaces import Chunk


class SemanticChunker:
    def __init__(self, min_tokens: int = 100, max_tokens: int = 800):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.similarity_threshold = 0.65

    @property
    def model(self):
        """复用 EmbeddingService 的全局单例模型，避免重复加载"""
        from engines.embedding.embedder import _get_model
        return _get_model()

    def chunk(self, uir_doc) -> List[Chunk]:
        tree = self._build_title_tree(uir_doc)
        paragraphs = self._flatten_paragraphs(uir_doc)
        cut_indices = self._detect_boundaries(paragraphs)
        return self._generate_chunks(uir_doc.doc_id, paragraphs, cut_indices, tree)

    def _build_title_tree(self, uir_doc) -> dict:
        tree = {}
        title_stack = []
        for page in uir_doc.pages:
            for block in page["blocks"]:
                if block["type"] == "title":
                    font_size = block.get("metadata", {}).get("font_size", 12)
                    level = self._estimate_heading_level(font_size, len(title_stack))
                    title_stack = title_stack[:level - 1]
                    title_stack.append(block["content"])
                    tree[block["content"]] = {"level": level, "path": list(title_stack), "children": []}
        return tree

    def _estimate_heading_level(self, font_size: float, current_level: int) -> int:
        if font_size >= 22:
            return 1
        elif font_size >= 18:
            return 2
        elif font_size >= 14:
            return 3
        return min(current_level + 1, 4)

    def _flatten_paragraphs(self, uir_doc) -> List[dict]:
        paragraphs = []
        for page in uir_doc.pages:
            for block in page["blocks"]:
                if block["type"] in ("paragraph", "title"):
                    paragraphs.append({
                        "content": block["content"],
                        "type": block["type"],
                        "page_num": block["page_num"]
                    })
        return paragraphs

    def _detect_boundaries(self, paragraphs: List[dict]) -> List[int]:
        if len(paragraphs) <= 1:
            return [0]
        texts = [p["content"][:512] for p in paragraphs]
        embeddings = self.model.encode(texts)
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        if not similarities:
            return [0]
        threshold = np.mean(similarities) - 0.5 * np.std(similarities)
        cut_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold or paragraphs[i + 1]["type"] == "title":
                cut_indices.append(i + 1)
        cut_indices.append(len(paragraphs))
        return cut_indices

    def _generate_chunks(self, doc_id: str, paragraphs: List[dict],
                         cut_indices: List[int], tree: dict) -> List[Chunk]:
        chunks = []
        max_chars = self.max_tokens * 4
        for i in range(len(cut_indices) - 1):
            start = cut_indices[i]
            end = cut_indices[i + 1]
            segment_paras = paragraphs[start:end]
            content = "\n\n".join(p["content"] for p in segment_paras)

            if len(content) > max_chars:
                for j in range(0, len(content), max_chars):
                    sub = content[j:j + max_chars]
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_chunk_{len(chunks):04d}",
                        doc_id=doc_id, content=sub.strip(),
                        context={}, metadata={"char_count": len(sub)}
                    ))
                continue

            title_chain = self._find_title_chain(paragraphs[start], tree) if tree else []
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{len(chunks):04d}",
                doc_id=doc_id, content=content.strip(),
                context={"title_chain": title_chain, "doc_title": doc_id},
                metadata={
                    "page_range": [paragraphs[start].get("page_num", 1),
                                   paragraphs[end - 1].get("page_num", 1)],
                    "char_count": len(content),
                    "paragraph_count": len(segment_paras),
                }
            ))
        return chunks

    def _find_title_chain(self, paragraph: dict, tree: dict) -> List[str]:
        """根据段落位置和标题树，找到最近的标题层级链。

        从标题树中找到该段落之前最近的标题，返回其 path（层级链）。
        """
        if not tree:
            return []

        # 按出现顺序收集所有标题
        titles_in_order = list(tree.keys())
        if not titles_in_order:
            return []

        # 找到段落之前的最近标题（简单策略：最后一个标题）
        # 更精确的实现需要记录标题在文档中的位置
        last_title = titles_in_order[-1]
        node = tree.get(last_title, {})
        return node.get("path", [last_title])
