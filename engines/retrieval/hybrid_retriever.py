"""向量 + 图谱混合检索器"""

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

        if self.graph_retriever:
            graph_context = self.graph_retriever.retrieve(query, top_k)
            if graph_context:
                docs = self._merge_graph_chunks(docs, graph_context.get("source_chunks") or [])

        docs = self._dedup(docs)

        # Reranker 精排
        if self.reranker and docs:
            docs = self.reranker.rerank(query, docs, top_k)

        docs = self._expand_parents(docs[:top_k])
        result = {"documents": docs, "graph_context": graph_context}
        # 流式 defer 路径: fused 也会进 _build_context, 一并展开父块
        if "fused" in result:
            result["fused"] = self._expand_parents(result["fused"])
        return result

    def _expand_parents(self, docs: list[dict]) -> list[dict]:
        """父子文档机制: 检索命中的是子块(小、精确), 但回给 LLM 的是父块(大上下文)。

        子块 metadata.parent_id 指向父块; 此处按 parent_id 批量取父块内容,
        写入 doc["parent_content"] (与子块 content 不同时才写, 避免冗余)。
        orchestrator._build_context 会优先用 parent_content 组装 LLM 上下文,
        来源面板仍显示子块片段(精确引用)。无 vector_store/无父块时原样返回。
        """
        if not self.vector_store:
            return docs
        parent_ids = [d.get("parent_id") for d in docs if d.get("parent_id")]
        if not parent_ids:
            return docs
        try:
            rows = self.vector_store.get_by_ids(parent_ids)
        except Exception as e:  # noqa: BLE001 — 父块查询失败不阻断检索
            logger.debug("展开父块失败(降级为子块内容): %s", str(e)[:120])
            return docs
        by_id = {r.get("id"): r for r in rows if r.get("id")}
        for d in docs:
            pid = d.get("parent_id")
            if not pid or pid not in by_id:
                continue
            pc = by_id[pid].get("content", "") or ""
            child = d.get("content", "")
            if pc and pc != child:
                d["parent_content"] = pc
        return docs

    def _vector_retrieve(self, query: str, top_k: int) -> list[dict]:
        if self.embedder and self.vector_store:
            query_emb = self.embedder.embed_query(query)
            return self.vector_store.search(query_emb, top_k=top_k)
        return []

    def _merge_graph_chunks(self, docs: list[dict], source_chunks: list) -> list[dict]:
        """图谱关联片段回填: str 从向量库查内容, dict 直接并入。

        existing_ids 随 append 同步更新 — 原实现只在循环前算一次, 重复 source_chunks
        (或重复 dict) 会重复入列。graph dict 补 score 与检索文档同构。
        """
        existing_ids = {d.get("chunk_id") or d.get("id") for d in docs}
        for cid in source_chunks:
            if isinstance(cid, dict):
                key = cid.get("chunk_id") or cid.get("id")
                if key and key in existing_ids:
                    continue
                docs.append(cid)
                if key:
                    existing_ids.add(key)
            elif cid not in existing_ids:
                docs.append(
                    {
                        "id": cid,
                        "chunk_id": cid,
                        "content": self._lookup_chunk_content(cid) or f"[图谱关联片段 {cid}]",
                        "source": "graph",
                        "score": 0.0,
                    }
                )
                existing_ids.add(cid)
        return docs

    @staticmethod
    def _dedup(docs: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for d in docs:
            key = d.get("chunk_id") or d.get("id", "")
            if key and key not in seen:
                out.append(d)
                seen.add(key)
        return out

    def _lookup_chunk_content(self, chunk_id: str) -> str:
        """从向量库按 id 直接取 chunk 内容。

        原实现用 [0.0]*512 零向量 + filter 表达式 hack, 换 embedding 模型(维度≠512)即报错;
        改走 VectorStore.get_by_ids (按主键过滤, 与向量维度无关)。
        """
        if not self.vector_store:
            return ""
        try:
            rows = self.vector_store.get_by_ids([chunk_id])
            return rows[0].get("content", "") if rows else ""
        except Exception as e:  # noqa: BLE001 — chunk 内容查询失败返回空串
            logger.debug("查询 chunk %s 内容失败: %s", chunk_id, str(e)[:100])
        return ""
