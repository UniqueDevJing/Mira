"""向量 + 图谱混合检索器"""
from typing import List
import logging

from engines.interfaces import RetrieverInterface

logger = logging.getLogger(__name__)


class HybridRetriever(RetrieverInterface):
    def __init__(self, vector_store, graph_retriever=None, embedder=None, reranker=None):
        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 20) -> dict:
        docs = self._vector_retrieve(query, top_k * 2)
        graph_context = None

        # 图谱检索
        if self.graph_retriever:
            graph_context = self.graph_retriever.retrieve(query, top_k)
            if graph_context and graph_context.get("source_chunks"):
                # 从向量库回查 source_chunks 的实际内容
                chunk_ids = [cid for cid in graph_context["source_chunks"]
                             if isinstance(cid, str)]
                if chunk_ids:
                    existing_ids = {d.get("chunk_id") or d.get("id") for d in docs}
                    for cid in chunk_ids:
                        if cid not in existing_ids:
                            # 尝试从向量库查询实际内容
                            content = self._lookup_chunk_content(cid)
                            docs.append({
                                "id": cid, "chunk_id": cid,
                                "content": content or f"[图谱关联片段 {cid}]",
                                "source": "graph",
                            })
                # dict 类型的 chunk 直接添加
                for cid in graph_context["source_chunks"]:
                    if isinstance(cid, dict):
                        docs.append(cid)

        seen = set()
        deduped = []
        for d in docs:
            key = d.get("chunk_id") or d.get("id", "")
            if key and key not in seen:
                deduped.append(d)
                seen.add(key)

        # Reranker 精排
        if self.reranker and deduped:
            deduped = self.reranker.rerank(query, deduped, top_k)

        return {"documents": deduped[:top_k], "graph_context": graph_context}

    def _vector_retrieve(self, query: str, top_k: int) -> List[dict]:
        if self.embedder and self.vector_store:
            query_emb = self.embedder.embed_query(query)
            return self.vector_store.search(query_emb, top_k=top_k)
        return []

    def _lookup_chunk_content(self, chunk_id: str) -> str:
        """从向量库中查询 chunk 的实际内容"""
        if not self.vector_store:
            return ""
        try:
            results = self.vector_store.search(
                [0.0] * 512,  # dummy embedding，仅用于过滤
                top_k=1,
                filter_expr=f'id = "{chunk_id}"',
            )
            if results:
                return results[0].get("content", "")
        except Exception as e:
            logger.debug("查询 chunk %s 内容失败: %s", chunk_id, str(e)[:100])
        return ""
