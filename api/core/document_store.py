"""文档存储 — SQLite 持久化。

替代内存 dict，重启后文档状态不丢失。
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# 基于项目根解析绝对路径 (api/core/document_store.py → 项目根/data/documents.db),
# 消除启动目录依赖 — 原相对路径从不同 CWD 启动会创建不同库
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = str(_PROJECT_ROOT / "data" / "documents.db")


class DocumentStore:
    """SQLite 文档存储"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """获取数据库连接。

        WAL 模式: 读与写可并发, 不再因单写者阻塞所有读; busy_timeout: 写争用时
        等待而非立即抛 "database is locked"; synchronous=NORMAL: WAL 下安全且提升写吞吐。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'processing',
                    page_count INTEGER,
                    chunk_count INTEGER,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 幂等迁移: 旧库缺 knowledge_base / doc_type 列时补加
            cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
            if "knowledge_base" not in cols:
                conn.execute("ALTER TABLE documents ADD COLUMN knowledge_base TEXT DEFAULT 'documents'")
            if "doc_type" not in cols:
                conn.execute("ALTER TABLE documents ADD COLUMN doc_type TEXT DEFAULT 'policy'")
            # QA 日志表 (异步写入, 用于成本/质量追踪)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT,
                    skill TEXT,
                    kb_id TEXT,
                    routing_source TEXT,
                    degradation_level INTEGER DEFAULT 0,
                    latency_ms INTEGER,
                    tokens_total INTEGER,
                    sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 幂等迁移: 旧库缺 sources 列时补加 (标注样本需检索上下文)
            qa_cols = {row[1] for row in conn.execute("PRAGMA table_info(qa_logs)").fetchall()}
            if "sources" not in qa_cols:
                conn.execute("ALTER TABLE qa_logs ADD COLUMN sources TEXT")
            # 创建更新触发器
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_timestamp
                AFTER UPDATE ON documents
                BEGIN
                    UPDATE documents SET updated_at = CURRENT_TIMESTAMP
                    WHERE doc_id = NEW.doc_id;
                END
            """)

    def save(
        self,
        doc_id: str,
        filename: str,
        status: str = "processing",
        page_count: int | None = None,
        chunk_count: int | None = None,
        error: str | None = None,
        knowledge_base: str = "documents",
        doc_type: str = "policy",
    ):
        """保存或更新文档"""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, filename, status, page_count, chunk_count, error, knowledge_base, doc_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    status = excluded.status,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    error = excluded.error,
                    knowledge_base = excluded.knowledge_base,
                    doc_type = excluded.doc_type
            """,
                (doc_id, filename, status, page_count, chunk_count, error, knowledge_base, doc_type),
            )
        logger.debug("文档已保存: doc_id=%s, status=%s, kb=%s", doc_id, status, knowledge_base)

    def log_qa(
        self,
        question: str,
        answer: str,
        skill: str = "",
        kb_id: str | None = None,
        routing_source: str | None = None,
        degradation_level: int = 0,
        latency_ms: int = 0,
        tokens_total: int = 0,
        sources: list | None = None,
    ):
        """写入 QA 日志（异步调用，不阻塞响应）。sources 为检索上下文, 供质量标注使用。"""
        try:
            sources_json = json.dumps(sources or [], ensure_ascii=False)
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO qa_logs
                       (question, answer, skill, kb_id, routing_source, degradation_level, latency_ms, tokens_total, sources)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        question,
                        answer,
                        skill,
                        kb_id,
                        routing_source,
                        degradation_level,
                        latency_ms,
                        tokens_total,
                        sources_json,
                    ),
                )
        except sqlite3.Error as e:
            logger.warning("QA 日志写入失败: %s", str(e)[:120])

    def get(self, doc_id: str) -> dict | None:
        """获取文档"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def list_all(self, page: int = 1, size: int = 20, kb_in: list[str] | None = None) -> dict:
        """列出文档（分页）。

        kb_in: 仅返回这些知识库的文档 (KB 级 RBAC 用); None = 全部。
        """
        offset = (page - 1) * size
        with self._get_conn() as conn:
            if kb_in is None:
                # None = 不限制 (admin / 未鉴权), 返回全部
                where = ""
                params = []
            elif len(kb_in) == 0:
                # [] = 明确无权访问任何知识库 (空 allowed_kbs 的 reader) → 直接返回空, 不构造 IN() 以免 SQL 参数缺失报错
                return {"items": [], "total": 0}
            else:
                placeholders = ",".join("?" for _ in kb_in)
                where = f"WHERE knowledge_base IN ({placeholders})"
                params = list(kb_in)
            # 获取总数
            total = conn.execute(f"SELECT COUNT(*) FROM documents {where}", params).fetchone()[0]

            # 获取分页数据
            rows = conn.execute(
                f"SELECT * FROM documents {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [size, offset],
            ).fetchall()

            return {
                "items": [dict(r) for r in rows],
                "total": total,
            }

    def update_status(self, doc_id: str, status: str, **kwargs):
        """更新文档状态"""
        allowed_fields = {"page_count", "chunk_count", "error"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        updates["status"] = status

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [doc_id]

        with self._get_conn() as conn:
            conn.execute(f"UPDATE documents SET {set_clause} WHERE doc_id = ?", values)
        logger.debug("文档状态已更新: doc_id=%s, status=%s", doc_id, status)

    def delete(self, doc_id: str) -> bool:
        """删除文档"""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount > 0

    def reset_stale_processing(self) -> int:
        """服务重启时重置卡在 processing 的文档 (后台任务随进程中断, 状态永不变)。"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE documents SET status = 'failed', error = '服务重启，处理中断，请重新上传' "
                "WHERE status = 'processing'"
            )
        if cursor.rowcount:
            logger.warning("重置 %d 个中断的 processing 文档为 failed", cursor.rowcount)
        return cursor.rowcount


# 全局单例
_document_store: DocumentStore | None = None


def get_document_store() -> DocumentStore:
    """获取全局文档存储单例"""
    global _document_store
    if _document_store is None:
        _document_store = DocumentStore()
    return _document_store
