"""文档存储 — SQLite 持久化。

替代内存 dict，重启后文档状态不丢失。
"""
import sqlite3
import json
import logging
from typing import Optional, List, Dict
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# 默认数据库路径
DEFAULT_DB_PATH = "./data/documents.db"


class DocumentStore:
    """SQLite 文档存储"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
            # 创建更新触发器
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_timestamp
                AFTER UPDATE ON documents
                BEGIN
                    UPDATE documents SET updated_at = CURRENT_TIMESTAMP
                    WHERE doc_id = NEW.doc_id;
                END
            """)

    def save(self, doc_id: str, filename: str, status: str = "processing",
             page_count: int = None, chunk_count: int = None, error: str = None):
        """保存或更新文档"""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO documents (doc_id, filename, status, page_count, chunk_count, error)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    status = excluded.status,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    error = excluded.error
            """, (doc_id, filename, status, page_count, chunk_count, error))
        logger.debug("文档已保存: doc_id=%s, status=%s", doc_id, status)

    def get(self, doc_id: str) -> Optional[Dict]:
        """获取文档"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row:
                return dict(row)
            return None

    def list_all(self, page: int = 1, size: int = 20) -> Dict:
        """列出文档（分页）"""
        offset = (page - 1) * size
        with self._get_conn() as conn:
            # 获取总数
            total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

            # 获取分页数据
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (size, offset)
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
            conn.execute(
                f"UPDATE documents SET {set_clause} WHERE doc_id = ?",
                values
            )
        logger.debug("文档状态已更新: doc_id=%s, status=%s", doc_id, status)

    def delete(self, doc_id: str) -> bool:
        """删除文档"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
            )
            return cursor.rowcount > 0


# 全局单例
_document_store: Optional[DocumentStore] = None


def get_document_store() -> DocumentStore:
    """获取全局文档存储单例"""
    global _document_store
    if _document_store is None:
        _document_store = DocumentStore()
    return _document_store
