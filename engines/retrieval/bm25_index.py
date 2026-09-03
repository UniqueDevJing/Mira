"""BM25 稀疏检索 — 与向量检索并行，RRF 融合。

自实现（jieba 分词，标准 BM25 公式），避免引入额外依赖。
每知识库一个独立实例，内存增量维护。查询结果与向量检索同结构（dict），便于 RRF 融合。
"""

import json
import logging
import math
import os
import threading
from collections import Counter, defaultdict

import jieba

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


class Bm25Index:
    """BM25 索引。add_documents 增量追加，查询实时计分。

    线程安全: 后台上传线程 add_documents/remove_doc 与请求线程 search 并发,
    用 RLock 保护共享状态 (原实现无锁, 并发读写可致列表错位/统计中间态)。
    持久化: persist_path 指定时变更即落盘 (进程重启索引不丢, 与向量库持久化对称)。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, persist_path: str | None = None):
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()
        self._docs: list[dict] = []
        self._token_counts: list[Counter] = []
        self._df: dict[str, int] = defaultdict(int)  # token -> 出现文档数
        self._total_tokens = 0  # 运行总和, 增量维护 (原 add 全量重算 sum O(N), 大量增量上传累计 O(N²))
        self._avgdl = 0.0
        self._persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            data = {
                "docs": self._docs,
                "df": dict(self._df),
                "total_tokens": self._total_tokens,
                "avgdl": self._avgdl,
            }
            tmp_path = self._persist_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, self._persist_path)
        except OSError as e:
            logger.error("BM25 持久化写入失败: %s", str(e)[:120])

    @staticmethod
    def _read_persisted(path: str):
        """读取持久化索引, 返回 dict; 不可恢复时返回 None。

        现行实现统一用 JSON 落盘。历史 pickle 兼容分支已删除(仓库内无残留 .pkl,
        且从向量库自愈重建机制可在 JSON 损坏/缺失时回退完整索引)。读取失败一律回退
        None, 由 `_load` → `_check_bm25_consistency` 触发从向量库重建, 避免静默退化成纯向量检索。
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            logger.warning("BM25 持久化读取失败(将触发从向量库重建): %s", str(e)[:120])
            return None
        except OSError as e:
            logger.warning("BM25 持久化读取失败: %s", str(e)[:120])
            return None

    def _load(self, path: str) -> None:
        data = self._read_persisted(path)
        if data is None:
            return
        try:
            self._docs = data["docs"]
            self._token_counts = []
            for doc in self._docs:
                content = doc.get("content", "") or ""
                self._token_counts.append(Counter(_tokenize(content)))
            self._df = defaultdict(int, data["df"])
            self._total_tokens = data["total_tokens"]
            self._avgdl = data["avgdl"]
            logger.info("BM25 索引从 %s 恢复: %d 文档", path, len(self._docs))
        except Exception as e:  # noqa: BLE001
            logger.warning("BM25 持久化数据损坏, 回退空索引: %s", str(e)[:120])

    def add_documents(self, docs: list[dict]) -> None:
        """追加文档，维护 df / avgdl 增量统计。"""
        if not docs:
            return
        with self._lock:
            added_tokens = 0
            for doc in docs:
                content = doc.get("content", "") or ""
                counts = Counter(_tokenize(content))
                self._token_counts.append(counts)
                added_tokens += counts.total()
                for token in counts:
                    self._df[token] += 1
                self._docs.append(doc)
            self._total_tokens += added_tokens
            self._avgdl = self._total_tokens / len(self._docs) if self._docs else 0.0
            self._save()
            logger.debug("BM25 索引更新: %d 文档, avgdl=%.1f", len(self._docs), self._avgdl)

    def rebuild(self, docs: list[dict]) -> None:
        """全量重建索引(先清空再添加), 用于从权威数据源(向量库)恢复。

        与 add_documents 的区别: add_documents 是**追加**, 若在已加载旧内容的实例上
        调用会把恢复数据叠加成重复。重建/自愈场景必须用本方法。
        结束后照常落盘, 调用方持有的实例引用无需替换。
        """
        with self._lock:
            self._docs = []
            self._token_counts = []
            self._df = defaultdict(int)
            self._total_tokens = 0
            self._avgdl = 0.0
        self.add_documents(docs)
        logger.info("BM25 索引已全量重建: %d 文档", len(self._docs))

    def __len__(self) -> int:
        with self._lock:
            return len(self._docs)

    def remove_doc(self, doc_id: str) -> int:
        """按 doc_id 删除文档索引 (向量删除时同步调用, 防检索残留)。

        删除后重建 df / avgdl, 保持后续检索统计准确。
        """
        with self._lock:
            keep_docs, keep_counts = [], []
            removed_tokens = 0
            removed_df: Counter = Counter()
            removed = 0
            for doc, counts in zip(self._docs, self._token_counts):
                if doc.get("doc_id") == doc_id:
                    removed += 1
                    removed_tokens += counts.total()
                    for token in counts:
                        removed_df[token] += 1
                    continue
                keep_docs.append(doc)
                keep_counts.append(counts)
            if not removed:
                return 0
            self._docs, self._token_counts = keep_docs, keep_counts
            for token, n in removed_df.items():
                self._df[token] -= n
                if self._df[token] <= 0:
                    del self._df[token]
            self._total_tokens -= removed_tokens
            self._avgdl = self._total_tokens / len(self._docs) if self._docs else 0.0
            self._save()
            logger.info("BM25 删除文档 %s: %d 条索引", doc_id, removed)
            return removed

    def _idf(self, token: str) -> float:
        n = len(self._docs)
        df = self._df.get(token, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """返回 [{id, chunk_id, doc_id, content, score}], score 归一化到 0-1。"""
        with self._lock:
            if not self._docs:
                return []
            tokens = _tokenize(query)
            if not tokens:
                return []

            scores = []
            for i, counts in enumerate(self._token_counts):
                doc_len = counts.total()
                score = 0.0
                for token in set(tokens):
                    tf = counts.get(token, 0)
                    if tf == 0:
                        continue
                    denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self._avgdl) if self._avgdl else 1.0
                    score += self._idf(token) * (tf * (self.k1 + 1.0)) / denom
                if score > 0:
                    scores.append((i, score))

            if not scores:
                return []

            # 按分数排序取 top_k
            scores.sort(key=lambda x: x[1], reverse=True)
            max_score = scores[0][1] if scores else 1.0

            results = []
            for idx, raw in scores[:top_k]:
                doc = self._docs[idx]
                results.append(
                    {
                        "id": doc.get("id", ""),
                        "chunk_id": doc.get("chunk_id", doc.get("id", "")),
                        "doc_id": doc.get("doc_id", ""),
                        "content": doc.get("content", ""),
                        "score": round(raw / max_score, 4) if max_score else 0.0,
                        "_bm25": round(raw, 4),
                    }
                )
            return results
