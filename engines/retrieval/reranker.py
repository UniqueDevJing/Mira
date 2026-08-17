"""重排序引擎 — Bi-Encoder 嵌入相似度精排。

当前实现使用 embedding 余弦相似度（Bi-Encoder），适用于无 GPU 环境。
若需 Cross-Encoder 重排序（更高精度），可传入 ce_model_name 加载。
"""

import logging
import threading

from engines.interfaces import RerankerInterface

logger = logging.getLogger(__name__)


class Reranker(RerankerInterface):
    def __init__(self, embedder=None, ce_model_name: str | None = None):
        self.embedder = embedder
        self._ce_model = None
        self._ce_model_name = ce_model_name
        self._ce_lock = threading.Lock()

    def _get_ce_model(self):
        """延迟加载 Cross-Encoder 模型 (双重检查锁: 并发首请求不重复加载, CE 模型内存占用大)"""
        if self._ce_model is None and self._ce_model_name:
            with self._ce_lock:
                if self._ce_model is None and self._ce_model_name:
                    try:
                        from sentence_transformers import CrossEncoder

                        logger.info("加载 Cross-Encoder 模型: %s", self._ce_model_name)
                        self._ce_model = CrossEncoder(self._ce_model_name)
                    except Exception as e:  # noqa: BLE001 — 降级边界: CE 加载失败用 Bi-Encoder
                        logger.warning("Cross-Encoder 加载失败，降级到 Bi-Encoder: %s", str(e)[:200])
        return self._ce_model

    def rerank(self, query: str, documents: list[dict], top_k: int = 10) -> list[dict]:
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

    def _rerank_with_ce(self, query: str, documents: list[dict], top_k: int, ce_model) -> list[dict]:
        """Cross-Encoder 重排序（精度更高）"""
        pairs = [(query, d.get("content", "")[:512]) for d in documents]
        try:
            scores = ce_model.predict(pairs)
        except Exception as e:  # noqa: BLE001 — 降级边界: CE 推理异常回退 Bi-Encoder, 不中断检索
            logger.warning("Cross-Encoder 推理失败, 降级 Bi-Encoder: %s", str(e)[:120])
            return self._rerank_with_bi_encoder(query, documents, top_k)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in scored[:top_k]:
            new_doc = dict(doc)  # 不改输入: 调用方可能复用原对象 (如 RRF 融合分数)
            new_doc["score"] = round(float(score), 4)
            result.append(new_doc)
        return result

    def _rerank_with_bi_encoder(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        """Bi-Encoder 余弦相似度重排序（默认降级方案）"""
        import numpy as np

        query_emb = self.embedder.embed_query(query)

        # 缺失存储向量的文档: 一次性批量嵌入 (embed_batch 带 passage: 前缀, 与库内一致;
        # 原实现逐条 embed_query 走 query: 前缀, 相似度被压低且 N 次调用)
        missing = [(i, d.get("content", "")) for i, d in enumerate(documents) if not d.get("embedding")]
        emb_by_idx = {}
        if missing:
            batch_embs = self.embedder.embed_batch([c[:512] for _, c in missing if c])
            j = 0
            for i, c in missing:
                if c:
                    emb_by_idx[i] = batch_embs[j]
                    j += 1

        scored = []
        for i, doc in enumerate(documents):
            if doc.get("embedding"):
                score = float(np.dot(query_emb, doc["embedding"]))
            elif i in emb_by_idx:
                score = float(np.dot(query_emb, emb_by_idx[i]))
            else:
                scored.append((doc, 0.0))
                continue

            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in scored[:top_k]:
            new_doc = dict(doc)  # 不改输入: 调用方可能复用原对象 (如 RRF 融合分数)
            new_doc["score"] = round(score, 4)
            result.append(new_doc)
        return result
