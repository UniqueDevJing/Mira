"""文本嵌入服务 — 全局单例模式，避免事件循环内加载模型"""

import hashlib
import logging
import os
import time
from threading import Lock

from engines.interfaces import EmbedderInterface

logger = logging.getLogger(__name__)

# 模块级预加载（在 asyncio 事件循环之外）
_model = None
_model_name = None
_model_lock = Lock()

# Query Embedding 缓存 — key=hash(normalize(query)), TTL 短, 上限防膨胀
_query_cache: dict = {}
_query_cache_lock = Lock()
_query_cache_stats = {"hits": 0, "misses": 0}
_query_cache_max = 4096


def _query_cache_ttl() -> int:
    return int(os.environ.get("RAG_EMBED_CACHE_TTL_S", "600"))


def _query_cache_key(query: str, model_name: str) -> str:
    # key 含模型名: 切换 embedding 模型后旧向量不应命中 (C3)
    # 仍对原 query.strip() 计算 (大小写敏感, 与 "query:" 前缀嵌入输入一致)
    return hashlib.md5(f"{model_name}|{query.strip()}".encode()).hexdigest()


def _get_model(model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        with _model_lock:
            if _model is None or _model_name != model_name:
                from sentence_transformers import SentenceTransformer

                logger.info("加载 Embedding 模型: %s", model_name)
                _model = SentenceTransformer(model_name, device=device)
                _model_name = model_name
    return _model


class EmbeddingService(EmbedderInterface):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.batch_size = 32

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        processed = [f"passage: {t}" for t in texts]
        model = _get_model(self.model_name, self.device)
        embeddings = model.encode(
            processed, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """查询嵌入，带 TTL 缓存（相同问题重复查询不重算）。"""
        key = _query_cache_key(query, self.model_name)
        ttl = _query_cache_ttl()
        with _query_cache_lock:
            hit = _query_cache.get(key)
            if hit and (time.time() - hit[1]) < ttl:
                _query_cache_stats["hits"] += 1
                return hit[0]

        model = _get_model(self.model_name, self.device)
        embeddings = model.encode([f"query: {query}"], normalize_embeddings=True)
        result = embeddings[0].tolist()

        with _query_cache_lock:
            _query_cache_stats["misses"] += 1
            _query_cache[key] = (result, time.time())
            # 防膨胀: 逐出最旧一条, 替代原整表 clear (突发大量不同 query 会反复全清, 命中率归零)
            if len(_query_cache) > _query_cache_max:
                oldest = min(_query_cache, key=lambda k: _query_cache[k][1])
                del _query_cache[oldest]
        return result

    @staticmethod
    def embed_cache_stats() -> dict:
        """缓存命中统计（用于 Prometheus 上报）。"""
        with _query_cache_lock:
            return {
                "hits": _query_cache_stats["hits"],
                "misses": _query_cache_stats["misses"],
                "size": len(_query_cache),
            }
