"""向量存储 — LanceDB 嵌入式（零锁冲突，持久化，支持并发）"""
import time
import logging
from typing import List, Optional
import lancedb

from engines.interfaces import VectorStoreInterface

logger = logging.getLogger(__name__)


class VectorStore(VectorStoreInterface):
    def __init__(self, uri: str = "./lancedb_data", dim: int = 512):
        self.uri = uri
        self.dim = dim
        self.table_name = "documents"
        self.db = lancedb.connect(uri)
        self._table = self._ensure_table()

    @property
    def table(self):
        """延迟刷新表引用，支持多进程写入后读取"""
        return self.db.open_table(self.table_name)

    def _ensure_table(self):
        try:
            return self.db.open_table(self.table_name)
        except Exception as e:
            logger.info("表 %s 不存在，创建新表: %s", self.table_name, str(e)[:100])
            import numpy as np
            import pyarrow as pa
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), list_size=self.dim)),
                pa.field("created_at", pa.int64()),
            ])
            tbl = self.db.create_table(self.table_name, schema=schema)
            return tbl

    def insert(self, chunks: list, dedup: bool = True):
        """插入 chunks 到向量库。dedup=True 时先删除同 doc_id 的旧数据。"""
        if not chunks:
            return
        if dedup:
            doc_ids = {c.doc_id for c in chunks}
            for doc_id in doc_ids:
                self.delete_by_doc_id(doc_id)

        rows = [{
            "id": c.chunk_id,
            "doc_id": c.doc_id,
            "content": c.content[:65535],
            "embedding": [float(x) for x in c.embedding],
            "created_at": int(time.time()),
        } for c in chunks]
        self.table.add(rows)

    def search(self, query_embedding: List[float], top_k: int = 20,
               filter_expr: Optional[str] = None) -> List[dict]:
        try:
            q = self.table.search([float(x) for x in query_embedding]) \
                .metric("cosine") \
                .limit(top_k)
            if filter_expr:
                q = q.where(filter_expr)
            results = q.to_list()

            docs = []
            for r in results:
                docs.append({
                    "id": r.get("id", ""),
                    "chunk_id": r.get("id", ""),
                    "doc_id": r.get("doc_id", ""),
                    "content": r.get("content", ""),
                    "score": round(1.0 - float(r.get("_distance", 0)), 4),
                })
            return docs
        except Exception as e:
            logger.error("向量检索失败: %s", str(e)[:200])
            return []

    def delete_by_doc_id(self, doc_id: str) -> None:
        """按文档 ID 删除向量"""
        try:
            self.table.delete(f'doc_id = "{doc_id}"')
            logger.info("已删除文档 %s 的向量", doc_id)
        except Exception as e:
            logger.error("删除文档 %s 向量失败: %s", doc_id, str(e)[:200])
            raise
