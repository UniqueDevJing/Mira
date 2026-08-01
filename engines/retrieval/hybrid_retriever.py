"""向量 + 图谱混合检索器"""
from typing import List


class HybridRetriever:
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
                # source_chunks contains chunk_id strings, wrap as dicts
                for cid in graph_context["source_chunks"]:
                    if isinstance(cid, str):
                        docs.append({"id": cid, "chunk_id": cid, "content": cid})
                    elif isinstance(cid, dict):
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
        if self.embedder:
            query_emb = self.embedder.embed_query(query)
            return self.vector_store.search(query_emb, top_k=top_k)
        return []
