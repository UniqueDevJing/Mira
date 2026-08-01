"""文本嵌入服务 — 全局单例模式，避免事件循环内加载模型"""
from typing import List

# 模块级预加载（在 asyncio 事件循环之外）
_model = None
_model_name = None


def _get_model(model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name, device=device)
        _model_name = model_name
    return _model


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.batch_size = 32

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        processed = [f"passage: {t}" for t in texts]
        model = _get_model(self.model_name, self.device)
        embeddings = model.encode(
            processed, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        model = _get_model(self.model_name, self.device)
        embeddings = model.encode(
            [f"query: {query}"], normalize_embeddings=True
        )
        return embeddings[0].tolist()
