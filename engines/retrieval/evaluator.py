"""检索质量评估器"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalResult:
    relevance_score: float
    coverage_score: float
    confidence_score: float
    need_rewrite: bool
    reason: str = ""


class RetrievalEvaluator:
    def __init__(self, embedder, relevance_threshold: float = 0.70):
        self.embedder = embedder
        self.threshold = relevance_threshold

    def evaluate(self, query: str, results: list[dict]) -> EvalResult:
        if not results:
            return EvalResult(0, 0, 0, True, "无检索结果")
        relevance = self._calc_relevance(query, results)
        coverage = self._calc_coverage(query, results)
        confidence = 0.5 * relevance + 0.3 * coverage + 0.2 * min(len(results) / 10, 1.0)
        need_rewrite = (
            relevance < self.threshold or coverage < 0.5 or len([r for r in results if r.get("score", 0) > 0.7]) < 3
        )
        reason = ""
        if need_rewrite:
            parts = []
            if relevance < self.threshold:
                parts.append(f"相关性不足({relevance:.2f})")
            if coverage < 0.5:
                parts.append(f"覆盖度不足({coverage:.2f})")
            reason = "; ".join(parts)
        return EvalResult(relevance, coverage, confidence, need_rewrite, reason)

    def _calc_relevance(self, query: str, results: list[dict]) -> float:
        try:
            emb = self.embedder.embed_query(query)
        except Exception:  # noqa: BLE001 — 嵌入失败返回 0, 不使用随机向量
            return 0.0
        contents = [r.get("content", "")[:512] for r in results[:10] if r.get("content")]
        if not contents:
            return 0.0
        try:
            content_embs = self.embedder.embed_batch(contents)
        except Exception:  # noqa: BLE001 — 嵌入失败返回 0, 不使用随机向量
            return 0.0
        scores = [float(np.dot(emb, e)) for e in content_embs]
        return float(np.median(scores)) if scores else 0.0

    def _calc_coverage(self, query: str, results: list[dict]) -> float:
        import jieba

        query_words = set(jieba.cut(query))
        result_text = " ".join(r.get("content", "") for r in results[:10])
        covered = sum(1 for w in query_words if len(w) >= 2 and w in result_text)
        total = len([w for w in query_words if len(w) >= 2])
        return covered / total if total else 0.5
