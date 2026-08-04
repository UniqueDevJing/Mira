"""重排序引擎 — Bi-Encoder 嵌入相似度精排。

当前实现使用 embedding 余弦相似度（Bi-Encoder），适用于无 GPU 环境。
若需 Cross-Encoder 重排序（更高精度），可传入 ce_model_name 加载。
"""
from typing import List
import logging

from engines.interfaces import RerankerInterface

logger = logging.getLogger(__name__)


class Reranker(RerankerInterface):
    def __init__(self, embedder=None, ce_model_name: str = None):
        self.embedder = embedder
        self._ce_model = None
        self._ce_model_name = ce_model_name

    def _get_ce_model(self):
        """延迟加载 Cross-Encoder 模型"""
        if self._ce_model is None and self._ce_model_name:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("加载 Cross-Encoder 模型: %s", self._ce_model_name)
                self._ce_model = CrossEncoder(self._ce_model_name)
            except Exception as e:
                logger.warning("Cross-Encoder 加载失败，降级到 Bi-Encoder: %s", str(e)[:200])
        return self._ce_model

    def rerank(self, query: str, documents: List[dict], top_k: int = 10) -> List[dict]:
        """对检索结果重排序。优先使用 Cross-Encoder，降级到 Bi-Encoder。"""
        if not documents:
            return []

        # 尝试 Cross-Encoder
        ce_model = self._get_ce_model()
        if ce_model:
            return self._rerank_with_ce(query, documents, top_k, ce_model)

        # 降级到 Bi-Encoder
        if not self.embedder:
            return documents[:top_k]
        return self._rerank_with_bi_encoder(query, documents, top_k)

    def _rerank_with_ce(self, query: str, documents: List[dict],
                        top_k: int, ce_model) -> List[dict]:
        """Cross-Encoder 重排序（精度更高）"""
        pairs = [(query, d.get("content", "")[:512]) for d in documents]
        scores = ce_model.predict(pairs)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in scored[:top_k]:
            doc["score"] = round(float(score), 4)
            result.append(doc)
        return result

    def _rerank_with_bi_encoder(self, query: str, documents: List[dict],
                                top_k: int) -> List[dict]:
        """Bi-Encoder 余弦相似度重排序（默认降级方案）"""
        import numpy as np
        query_emb = self.embedder.embed_query(query)

        scored = []
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                scored.append((doc, 0.0))
                continue

            if "embedding" in doc and doc["embedding"]:
                score = float(np.dot(query_emb, doc["embedding"]))
            else:
                doc_emb = self.embedder.embed_query(content[:512])
                score = float(np.dot(query_emb, doc_emb))

            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in scored[:top_k]:
            doc["score"] = round(score, 4)
            result.append(doc)
        return result
