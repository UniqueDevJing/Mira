"""重排序引擎 — 嵌入相似度 + 可选 Cross-Encoder"""
from typing import List


class Reranker:
    def __init__(self, embedder=None):
        self.embedder = embedder
        self._ce_model = None

    def rerank(self, query: str, documents: List[dict], top_k: int = 10) -> List[dict]:
        """对检索结果重排序"""
        if not documents or not self.embedder:
            return documents[:top_k]

        query_emb = self.embedder.embed_query(query)

        # 计算每个文档与查询的余弦相似度
        scored = []
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                scored.append((doc, 0))
                continue

            # 使用已有 embedding 或重新计算
            if "embedding" in doc and doc["embedding"]:
                import numpy as np
                score = float(np.dot(query_emb, doc["embedding"]))
            else:
                doc_emb = self.embedder.embed_query(content[:512])
                import numpy as np
                score = float(np.dot(query_emb, doc_emb))

            scored.append((doc, score))

        # 按分数降序排列
        scored.sort(key=lambda x: x[1], reverse=True)

        # 更新分数
        result = []
        for doc, score in scored[:top_k]:
            doc["score"] = round(score, 4)
            result.append(doc)

        return result
