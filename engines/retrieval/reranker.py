"""重排序引擎 — Bi-Encoder 嵌入相似度精排。

当前实现使用 embedding 余弦相似度（Bi-Encoder），适用于无 GPU 环境。
若需 Cross-Encoder 重排序（更高精度），可传入 ce_model_name 加载。
"""

import logging
import os
import threading

from engines.interfaces import RerankerInterface

logger = logging.getLogger(__name__)


class Reranker(RerankerInterface):
    def __init__(self, embedder=None, ce_model_name: str | None = None, max_length: int | None = None,
                 backend: str = "torch"):
        """max_length: CE 输入的最大 token 数。None = 模型默认 (通常 512)。

        这是 CPU 上 rerank 延迟的**最大杠杆**: 批内 padding 到最长序列, 注意力成本 O(seq^2)。
        本机实测 (bge-reranker-base, 10 候选, 24 线程 CPU):
            512 -> 1112ms    384 -> 964ms    256 -> 595ms    192 -> 436ms    128 -> 291ms
        调小前务必用 scripts/bench_rerank_backends.py 在自己的语料上验证排序一致性。

        backend: torch(默认) / onnx / auto(优先 ONNX, 失败回退 PyTorch)。
        默认 torch: 本机实测 ONNX Runtime 反而更慢 (fp32 0.76×, int8 1.21× 但排序一致率仅 84%)。
        ONNX 路径保留, 换硬件后可用 scripts/bench_rerank_backends.py 复测再切。
        """
        self.embedder = embedder
        self._ce_model = None
        self._ce_model_name = ce_model_name
        self._ce_lock = threading.Lock()
        self.max_length = max_length
        self.backend = backend

    def _get_ce_model(self):
        """延迟加载 Cross-Encoder 模型 (双重检查锁: 并发首请求不重复加载, CE 模型内存占用大)"""
        if self._ce_model is None and self._ce_model_name:
            with self._ce_lock:
                if self._ce_model is None and self._ce_model_name:
                    try:
                        self._ce_model = self._load_ce()
                    except Exception as e:  # noqa: BLE001 — 降级边界: CE 加载失败用 Bi-Encoder
                        logger.warning("Cross-Encoder 加载失败，降级到 Bi-Encoder: %s", str(e)[:200])
        return self._ce_model

    def _load_ce(self):
        """按 backend 选择 PyTorch / ONNX 推理后端, 任一失败自动回退另一个。"""
        if self.backend in ("auto", "onnx"):
            onnx_dir = self._ce_model_name.rstrip("/\\") + "-onnx"
            if os.path.isdir(onnx_dir):
                try:
                    from engines.retrieval.onnx_scorer import OnnxCrossEncoder

                    logger.info("加载 ONNX Cross-Encoder: %s", onnx_dir)
                    return OnnxCrossEncoder(onnx_dir, prefer_int8=(self.backend == "onnx"),
                                            max_length=self.max_length)
                except Exception as e:
                    if self.backend == "onnx":
                        raise
                    logger.warning("ONNX 后端不可用, 回退 PyTorch: %s", str(e)[:200])
            elif self.backend == "onnx":
                logger.warning("backend=onnx 但未找到 ONNX 目录 %s, 回退 PyTorch", onnx_dir)

        from sentence_transformers import CrossEncoder

        logger.info("加载 PyTorch Cross-Encoder: %s (max_length=%s)", self._ce_model_name, self.max_length)
        return CrossEncoder(self._ce_model_name, max_length=self.max_length)

    def warmup(self) -> bool:
        """预热 Cross-Encoder 模型 — 供启动时后台线程调用, 返回是否就绪。

        CE 模型约 600MB, 且默认在首次 rerank 时才下载。不预热的话首个查询会卡在
        下载上, 被 rerank_timeout_s 判为超时并降级; 并发下多个请求还会阻塞在加载锁上。
        """
        return self._get_ce_model() is not None

    def rerank(self, query: str, documents: list[dict], top_k: int = 10) -> list[dict]:
        """对检索结果重排序。优先使用 Cross-Encoder，降级到 Bi-Encoder。"""
        if not documents:
            return []

        # 尝试 Cross-Encoder
        ce_model = self._get_ce_model()
        if ce_model:
            return self._rerank_with_ce(query, documents, top_k, ce_model)

        # 无 Cross-Encoder 时不重排 (bi-encoder 重排与向量检索信号重复, 零提升浪费 0.5s)
        return documents[:top_k]

    def rerank_fused(self, query: str, documents: list[dict], rrf_scores: dict[str, float],
                     top_k: int = 10, alpha: float = 0.7) -> list[dict]:
        """Cross-Encoder 分与检索(RRF)分融合重排。

        纯 Cross-Encoder 在密集近重复语料下会被"实体替换"型近重复干扰项迷惑,
        把 golden 误排到其后, Recall@3 下滑。融合检索分可托住 golden
        (其 RRF 排名天然靠前), 在保留 rerank 精度的同时避免召回下滑。
        """
        if not documents:
            return []
        ce_model = self._get_ce_model()
        if not ce_model:
            return documents[:top_k]
        pairs = [(query, d.get("content", "")[:512]) for d in documents]
        try:
            ce_scores = ce_model.predict(pairs)
        except Exception as e:  # noqa: BLE001 — 降级边界: CE 推理异常回退原序, 不中断检索
            logger.warning("Cross-Encoder 推理失败, 降级原序: %s", str(e)[:120])
            return documents[:top_k]

        import numpy as np

        ce_arr = np.array([float(s) for s in ce_scores], dtype=np.float64)
        cmin, cmax = float(ce_arr.min()), float(ce_arr.max())
        ce_norm = (ce_arr - cmin) / (cmax - cmin + 1e-9)
        rrf_arr = np.array(
            [float(rrf_scores.get(d.get("chunk_id") or d.get("id"), 0.0)) for d in documents],
            dtype=np.float64,
        )
        rmax = float(rrf_arr.max()) if rrf_arr.size else 1.0
        rrf_norm = rrf_arr / (rmax + 1e-9)
        final = (1.0 - alpha) * rrf_norm + alpha * ce_norm
        # stable: 分数并列时保持原池顺序 (即 RRF 检索序), 而非快排的任意顺序。
        # 既保证结果可复现, 又让"打平"时自然回退到检索系统的判断。
        order = np.argsort(-final, kind="stable")[:top_k]
        out = []
        for i in order:
            nd = dict(documents[int(i)])
            nd["score"] = round(float(final[int(i)]), 4)
            out.append(nd)
        return out

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
