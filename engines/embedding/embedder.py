"""文本嵌入服务 — 全局单例模式，避免事件循环内加载模型"""

import hashlib
import logging
import os
import time
from threading import Lock

import numpy as np

from engines.interfaces import EmbedderInterface


def _extract_dense(out):
    """兼容 bge-m3 的多输出与常规模型的单输出。

    bge-m3 的 SentenceTransformer.encode() 返回 (dense, sparse, colbert) 三元组，
    其它模型返回 (n, dim) ndarray。统一抽出 dense 向量。"""
    if isinstance(out, (tuple, list)) and len(out) == 3 and hasattr(out[0], "shape"):
        return np.asarray(out[0])
    return np.asarray(out)

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
    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        backend: str = "local",
        api_base: str = "",
        api_key: str = "",
        api_model: str = "",
        api_dims: int = 0,
        api_timeout_s: float = 10.0,
    ):
        # model_name 缺省时从全局配置读取 (2026-09-06: 支持选型切换, 默认指向 models/bge-base-zh-v1.5)
        if not model_name:
            from api.config import settings

            model_name = settings.embedding_model
        self.backend = (backend or "local").lower()
        self.device = device
        self.batch_size = 32
        self.api_base = (api_base or "").rstrip("/")
        self.api_key = api_key or ""
        self.api_model = api_model or model_name
        self.api_dims = api_dims or 0
        self.api_timeout_s = api_timeout_s
        self._api_client = None
        # 缓存键用的模型标识: API 模式用 api_model, local 用 model_name
        self.model_name = self.api_model if self.backend == "api" else model_name

    def embed_batch(self, texts: list[str], max_length: int = 512) -> list[list[float]]:
        if self.backend == "api":
            return self._api_embed(texts, is_query=False, max_length=max_length)
        if not texts:
            return []
        processed = [f"passage: {t}" for t in texts]
        model = _get_model(self.model_name, self.device)
        # Pre-truncate to max_length tokens (sentence-transformers v5.x encode() does not accept truncation kwargs).
        # 默认 512 与 bge-small 训练窗口对齐; 设为 0 表示不截断(用于诊断截断损失)。
        if max_length and max_length > 0:
            processed = [
                model.tokenizer.decode(
                    model.tokenizer(text, max_length=max_length, truncation=True)["input_ids"]
                )
                for text in processed
            ]
        embeddings = model.encode(
            processed, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return _extract_dense(embeddings).tolist()

    def embed_query(self, query: str, max_length: int = 512) -> list[float]:
        """查询嵌入，带 TTL 缓存（相同问题重复查询不重算）。"""
        if self.backend == "api":
            key = _query_cache_key(query, self.model_name)
            ttl = _query_cache_ttl()
            with _query_cache_lock:
                hit = _query_cache.get(key)
                if hit and (time.time() - hit[1]) < ttl:
                    _query_cache_stats["hits"] += 1
                    return hit[0]
            vec = self._api_embed([query], is_query=True, max_length=max_length)[0]
            with _query_cache_lock:
                _query_cache_stats["misses"] += 1
                _query_cache[key] = (vec, time.time())
                if len(_query_cache) > _query_cache_max:
                    oldest = min(_query_cache, key=lambda k: _query_cache[k][1])
                    del _query_cache[oldest]
            return vec
        key = _query_cache_key(query, self.model_name)
        ttl = _query_cache_ttl()
        with _query_cache_lock:
            hit = _query_cache.get(key)
            if hit and (time.time() - hit[1]) < ttl:
                _query_cache_stats["hits"] += 1
                return hit[0]

        model = _get_model(self.model_name, self.device)
        qtext = f"query: {query}"
        if max_length and max_length > 0:
            qtext = model.tokenizer.decode(
                model.tokenizer(qtext, max_length=max_length, truncation=True)["input_ids"]
            )
        embeddings = model.encode([qtext], normalize_embeddings=True)
        result = _extract_dense(embeddings)[0].tolist()

        with _query_cache_lock:
            _query_cache_stats["misses"] += 1
            _query_cache[key] = (result, time.time())
            # 防膨胀: 逐出最旧一条, 替代原整表 clear (突发大量不同 query 会反复全清, 命中率归零)
            if len(_query_cache) > _query_cache_max:
                oldest = min(_query_cache, key=lambda k: _query_cache[k][1])
                del _query_cache[oldest]
        return result

    def _api_embed(self, texts: list[str], is_query: bool = False, max_length: int = 512) -> list[list[float]]:
        """商用兼容 OpenAI 的 Embedding API(DashScope/OpenAI 等)。"""
        import httpx

        if not self.api_base:
            raise ValueError("embedding_backend=api 但未配置 embedding_api_base")
        payload: dict = {"model": self.api_model, "input": [str(t) for t in texts]}
        if self.api_dims and self.api_dims > 0:
            payload["dimensions"] = self.api_dims
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self._api_client is None:
            self._api_client = httpx.Client(timeout=self.api_timeout_s)
        resp = self._api_client.post(f"{self.api_base}/embeddings", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        # 按 index 排序, 保证与输入顺序一致(部分 API 乱序返回)
        ordered = sorted(data, key=lambda d: d.get("index", 0))
        return [self._normalize(d["embedding"]) for d in ordered]

    @staticmethod
    def _normalize(vec) -> list[float]:
        """L2 归一化, 与本地 normalize_embeddings=True 对齐。"""
        import numpy as np

        a = np.asarray(vec, dtype=float)
        n = float(np.linalg.norm(a))
        return (a / n).tolist() if n > 0 else a.tolist()

    @staticmethod
    def embed_cache_stats() -> dict:
        """缓存命中统计（用于 Prometheus 上报）。"""
        with _query_cache_lock:
            return {
                "hits": _query_cache_stats["hits"],
                "misses": _query_cache_stats["misses"],
                "size": len(_query_cache),
            }
