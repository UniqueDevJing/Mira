"""向量存储 — LanceDB 嵌入式（零锁冲突，持久化，支持并发）"""

import logging
import time

import lancedb
from lancedb.index import BTree, IvfHnswFlat

from engines.interfaces import VectorStoreInterface

logger = logging.getLogger(__name__)

# 检索失败可观测性: search() 降级返 [] 时, 区分"无结果"与"真出错"。
# 计数器供健康检查/指标读取, 避免静默吞异常导致线上检索故障不可见 (P-观测性)。
_VECTOR_SEARCH_ERRORS = 0


def get_vector_search_error_count() -> int:
    """本进程 VectorStore.search 累计失败次数 (降级返空但确有异常)。"""
    return _VECTOR_SEARCH_ERRORS


class VectorStore(VectorStoreInterface):
    def __init__(self, uri: str = "./lancedb_data", dim: int = 512, table_name: str = "documents"):
        self.uri = uri
        self.dim = dim
        self.table_name = table_name
        self.db = lancedb.connect(uri)
        self._table = self._ensure_table()
        # 表实际列 (旧表无 title_chain 等列, insert 时按列过滤, 新表全量)
        self._cols = set(self._table.schema.names)
        # 已有大表在初始化时即补齐索引 (生产环境重启/扩容后无需等下次写入)
        self._ensure_indices()

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
                    # 视频要求: chunk 级元数据 file_name / update_time (检索结果可展示来源文件与新鲜度)
                    pa.field("file_name", pa.string()),
                    pa.field("update_time", pa.int64()),
                    # 父子文档机制: 子块指向父块, 检索命中子块后回父块大上下文 (旧表无此列时按 _cols 守卫跳过)
                    pa.field("parent_id", pa.string()),
                ]
            )
            tbl = self.db.create_table(self.table_name, schema=schema)
            return tbl

    def _index_exists(self, column: str) -> bool:
        """某列是否已建索引。

        lancedb 0.36 的 index_stats 对向量/标量索引均返回 None (不可靠), 故改用
        list_indices 按列名判定, 跨进程可靠。
        """
        try:
            idxs = self.table.list_indices()
        except Exception:  # noqa: BLE001 — 无法判定时按"无索引"处理, 由 create 兜底
            return False
        return any(column in (getattr(ic, "columns", None) or []) for ic in idxs)

    def _ensure_indices(self) -> None:
        """滞后建索引: 仅在行数达阈值且尚未建索引时创建, 避免空表/小表建索引无效或开销。

        - 向量 ANN 索引 (IvfHnswFlat, cosine): 大语料检索从暴力 O(N) 降到近似对数级, 延迟显著下降;
          采用 HNSW(扁平, 无 PQ 量化)保证召回率, 且小表即可建、随数据增量更新。
        - 标量索引 (doc_id / parent_id, BTree): 加速 delete_by_doc_id 与 get_by_* 的 where 过滤下推。
        全部包裹在 try/except 中: 建索引失败绝不影响检索 (退化为暴力/全表扫描, 行为不变)。
        """
        try:
            count = self.table.count_rows()
        except Exception:  # noqa: BLE001 — 无法读取则跳过, 不影响主流程
            return
        # 向量索引策略 (LanceDB 0.36.0 实测):
        #  - 无索引时 search(vector) 对近似查询返回 0 行(仅精确命中偶发返回) → 小表完全不可检索;
        #    旧 `count < 256 不建索引` 依赖"暴力检索足够"的假设在本版本不成立 (已踩坑)。
        #  - 但 HNSW 在极小表(<~100 行)上图结构退化, 建索引的连接上检索反而返回 0(坏索引);
        #    故仅在 count >= 100 时建 IvfHnswFlat —— 覆盖中等/大表, 规避极小表退化。
        #  - 空表(<1)无法建索引, 跳过; <100 行的极小表属数据量限制, 需补足语料方可可靠检索。
        if count < 100:
            return
        # 向量 ANN 索引 (幂等: 已存在则跳过, 避免重复建索引)
        if not self._index_exists("embedding"):
            try:
                self.table.create_index("embedding", config=IvfHnswFlat(distance_type="cosine"))
                logger.info("已为表 %s 创建向量 ANN 索引 (IvfHnswFlat, 行数=%d)", self.table_name, count)
            except Exception as e:  # noqa: BLE001 — 建索引失败退化为暴力, 不阻断
                logger.warning("向量索引创建失败, 退化为暴力检索: %s", str(e)[:150])
        # 标量索引: 加速 doc_id / parent_id 过滤 (同样幂等 + 容错)
        for col in ("doc_id", "parent_id"):
            if col in self._cols and not self._index_exists(col):
                try:
                    self.table.create_index(col, config=BTree())
                except Exception as e:  # noqa: BLE001 — 标量索引失败可忽略
                    logger.debug("标量索引(%s)创建失败(可忽略): %s", col, str(e)[:100])

    def insert(self, chunks: list, dedup: bool = True):
        """插入 chunks 到向量库。两阶段提交: 先验证写入成功, 再删除旧数据, 防崩溃丢数据。"""
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
            row["file_name"] = (md.get("file_name") or "")[:512]
            row["update_time"] = int(md.get("update_time") or 0)
            row["parent_id"] = (md.get("parent_id") or "")[:128]
            rows.append({k: v for k, v in row.items() if k in self._cols})

        if not rows:
            return

        if dedup:
            tmp_rows = []
            for r in rows:
                tmp = dict(r)
                tmp["id"] = tmp["id"] + "_tmp"
                tmp_rows.append(tmp)
            try:
                self.table.add(tmp_rows)
            except Exception as e:
                raise RuntimeError(f"向量写入失败 (阶段1): {e}") from e
            doc_ids = {c.doc_id for c in chunks}
            for doc_id in doc_ids:
                self.delete_by_doc_id(doc_id)
            self.table.add(rows)
        else:
            self.table.add(rows)
        # 数据增长跨阈值后补齐索引 (首次写入 / 增量积累均会触发)
        self._ensure_indices()

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
                if "file_name" in self._cols:
                    doc["file_name"] = r.get("file_name") or ""
                if "update_time" in self._cols:
                    doc["update_time"] = r.get("update_time") or 0
                if "parent_id" in self._cols:
                    doc["parent_id"] = r.get("parent_id") or ""
                docs.append(doc)
            return docs
        except Exception as e:  # noqa: BLE001 — 检索降级边界: 失败返回空
            global _VECTOR_SEARCH_ERRORS
            _VECTOR_SEARCH_ERRORS += 1
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
            # 纯过滤查询: 不依赖向量检索。零向量 + prefilter 在存在 ANN 向量索引时会失效
            # (HNSW 索引下 prefilter 零向量查询返回 0 行), 故改用无向量 search().where() 全表过滤扫描,
            # 标量索引 (parent_id/doc_id) 可加速 where 下推。
            rows = self.table.search().where(f"id IN ({id_lit})").limit(len(ids)).to_list()
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
                if "file_name" in self._cols:
                    item["file_name"] = r.get("file_name") or ""
                if "update_time" in self._cols:
                    item["update_time"] = r.get("update_time") or 0
                if "parent_id" in self._cols:
                    item["parent_id"] = r.get("parent_id") or ""
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

    def get_by_doc_id(self, doc_id: str) -> list[dict]:
        """按文档 ID 取回该文档全部 chunk (供 U2「来源展开全文」端点)。

        与 get_by_ids 对称的防御性实现: 空 doc_id / 查询失败均返回空列表, 不抛异常
        (端点据此回 404, 而非 500)。双引号转义防过滤表达式注入。返回字段含
        content(完整 chunk 文本) + 元数据(title_chain/doc_title/file_name/update_time/parent_id),
        供前端展开展示。
        """
        if not doc_id:
            return []
        try:
            escaped = doc_id.replace('"', '""')
            # 纯过滤查询 (与 get_by_ids 对称): 无向量 search().where() 全表过滤扫描,
            # 避免 ANN 索引下零向量 prefilter 失效; limit 兜底防止超大文档一次性载入内存
            rows = self.table.search().where(f'doc_id = "{escaped}"').limit(100_000).to_list()
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
                if "file_name" in self._cols:
                    item["file_name"] = r.get("file_name") or ""
                if "update_time" in self._cols:
                    item["update_time"] = r.get("update_time") or 0
                if "parent_id" in self._cols:
                    item["parent_id"] = r.get("parent_id") or ""
                out.append(item)
            return out
        except Exception as e:  # noqa: BLE001 — 查询失败返回空, 端点升级为 404
            logger.error("按 doc_id 查询失败: %s", str(e)[:200])
            return []
