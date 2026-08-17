"""向量存储 — LanceDB 嵌入式（零锁冲突，持久化，支持并发）"""

import logging
import time

import lancedb

from engines.interfaces import VectorStoreInterface

logger = logging.getLogger(__name__)


class VectorStore(VectorStoreInterface):
    def __init__(self, uri: str = "./lancedb_data", dim: int = 512, table_name: str = "documents"):
        self.uri = uri
        self.dim = dim
        self.table_name = table_name
        self.db = lancedb.connect(uri)
        self._table = self._ensure_table()
        # 表实际列 (旧表无 title_chain 等列, insert 时按列过滤, 新表全量)
        self._cols = set(self._table.schema.names)

    @property
    def table(self):
        """延迟刷新表引用，支持多进程写入后读取"""
        return self.db.open_table(self.table_name)

    def _ensure_table(self):
        try:
            return self.db.open_table(self.table_name)
        except Exception as e:  # noqa: BLE001 — 表不存在是首次启动的正常流程
            logger.info("表 %s 不存在，创建新表: %s", self.table_name, str(e)[:100])
            import pyarrow as pa

            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("doc_id", pa.string()),
                    pa.field("content", pa.string()),
                    pa.field("embedding", pa.list_(pa.float32(), list_size=self.dim)),
                    pa.field("created_at", pa.int64()),
                    # 结构元数据: 标题链/文档标题/页码区间 — spec 承诺"标题上下文保留"
                    pa.field("title_chain", pa.list_(pa.string())),
                    pa.field("doc_title", pa.string()),
                    pa.field("page_range", pa.list_(pa.int32())),
                ]
            )
            tbl = self.db.create_table(self.table_name, schema=schema)
            return tbl

    def insert(self, chunks: list, dedup: bool = True):
        """插入 chunks 到向量库。dedup=True 时先删除同 doc_id 的旧数据。

        校验先行: 任何 chunk 缺 embedding 或维度不符, 在删旧数据前报错,
        防"旧已删新未写"导致该文档检索数据静默丢失 (换模型后旧表未重建是典型触发场景)。
        """
        if not chunks:
            return

        rows = []
        for c in chunks:
            if not c.embedding:
                raise ValueError(f"chunk {c.chunk_id} 缺少 embedding, 无法入库")
            if len(c.embedding) != self.dim:
                raise ValueError(
                    f"chunk {c.chunk_id} embedding 维度 {len(c.embedding)} != 表维度 {self.dim}, "
                    "可能换 embedding 模型后旧表未重建"
                )
            row = {
                "id": c.chunk_id,
                "doc_id": c.doc_id,
                "content": c.content[:65535],
                "embedding": [float(x) for x in c.embedding],
                "created_at": int(time.time()),
            }
            ctx = c.context or {}
            md = c.metadata or {}
            row["title_chain"] = ctx.get("title_chain") or []
            row["doc_title"] = ctx.get("doc_title", "")
            pr = md.get("page_range")
            row["page_range"] = [int(pr[0]), int(pr[1])] if pr and len(pr) >= 2 else []
            # 兼容旧表: 只写表实际存在的列 (旧表无新列时静默降级, 新表全量)
            rows.append({k: v for k, v in row.items() if k in self._cols})

        if dedup:
            doc_ids = {c.doc_id for c in chunks}
            for doc_id in doc_ids:
                self.delete_by_doc_id(doc_id)
        self.table.add(rows)

    def search(self, query_embedding: list[float], top_k: int = 20, filter_expr: str | None = None) -> list[dict]:
        try:
            q = self.table.search([float(x) for x in query_embedding]).metric("cosine").limit(top_k)
            if filter_expr:
                q = q.where(filter_expr)
            results = q.to_list()

            docs = []
            for r in results:
                # 余弦相似度可能为负: 夹取到 [0,1], 避免负分触发无谓跨库兜底
                doc = {
                    "id": r.get("id", ""),
                    "chunk_id": r.get("id", ""),
                    "doc_id": r.get("doc_id", ""),
                    "content": r.get("content", ""),
                    "score": round(max(0.0, 1.0 - float(r.get("_distance", 0))), 4),
                    # 携带存储向量: rerank 走 dot-product 快路径, 免每请求重复 embed
                    "embedding": r.get("embedding", []),
                }
                if "title_chain" in self._cols:
                    doc["title_chain"] = r.get("title_chain") or []
                if "doc_title" in self._cols:
                    doc["doc_title"] = r.get("doc_title") or ""
                if "page_range" in self._cols:
                    doc["page_range"] = r.get("page_range") or []
                docs.append(doc)
            return docs
        except Exception as e:  # noqa: BLE001 — 检索降级边界: 失败返回空
            logger.error("向量检索失败: %s", str(e)[:200])
            return []

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        """按 chunk id 直接取回行 (无需向量/维度, 供图谱关联回填等场景)。

        过滤下推: where IN + prefilter, 只取目标子集, 替代原全表 to_arrow 载入 O(N) 扫描
        (图谱回填每请求调用, 语料增长后全表载入是瓶颈)。
        """
        if not ids:
            return []
        try:
            # 单引号转义 (SQL 字符串字面量), chunk_id 含引号时防过滤表达式破坏
            id_lit = ", ".join("'" + str(i).replace("'", "''") + "'" for i in ids)
            # 零向量维度取表实际 schema (旧模型建的 768d 表不受 config dim=512 影响), 保持"换模型不炸"
            q_dim = self.dim
            try:
                fld = self._table.schema.field("embedding").type
                if hasattr(fld, "list_size") and fld.list_size > 0:
                    q_dim = fld.list_size
            except Exception as e:  # noqa: BLE001 — schema 读取失败退回 config dim
                logger.debug("表 schema 维度读取失败, 用 config dim=%d: %s", self.dim, str(e)[:80])
            rows = self.table.search([0.0] * q_dim).where(f"id IN ({id_lit})", prefilter=True).limit(len(ids)).to_list()
            out = []
            for r in rows:
                item = {
                    "id": r.get("id", ""),
                    "chunk_id": r.get("id", ""),
                    "doc_id": r.get("doc_id", ""),
                    "content": r.get("content", ""),
                }
                if "title_chain" in self._cols:
                    item["title_chain"] = r.get("title_chain") or []
                if "doc_title" in self._cols:
                    item["doc_title"] = r.get("doc_title") or ""
                out.append(item)
            return out
        except Exception as e:  # noqa: BLE001 — 查询失败返回空, 调用方降级
            logger.error("按 id 查询失败: %s", str(e)[:200])
            return []

    def delete_by_doc_id(self, doc_id: str) -> None:
        """按文档 ID 删除向量"""
        try:
            # 双引号转义: doc_id 含 " 会破坏过滤表达式 (与 get_by_ids 单引号转义对称)
            escaped = doc_id.replace('"', '""')
            self.table.delete(f'doc_id = "{escaped}"')
            logger.info("已删除文档 %s 的向量", doc_id)
        except Exception as e:
            logger.error("删除文档 %s 向量失败: %s", doc_id, str(e)[:200])
            raise
