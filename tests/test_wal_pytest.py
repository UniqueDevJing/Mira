"""SQLite WAL / 并发加固测试 — 验证高并发写入不再 database is locked。"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

from api.core.document_store import DocumentStore


def test_wal_mode_enabled(tmp_path):
    store = DocumentStore(db_path=str(tmp_path / "docs.db"))
    store.save("d1", "f1.txt")
    raw = sqlite3.connect(str(tmp_path / "docs.db"))
    try:
        mode = raw.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        raw.close()


def test_busy_timeout_configured(tmp_path):
    DocumentStore(db_path=str(tmp_path / "docs.db"))  # 构造即建库并启用 WAL
    raw = sqlite3.connect(str(tmp_path / "docs.db"))
    try:
        timeout = raw.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == 5000
    finally:
        raw.close()


def test_concurrent_writes_no_lock_error(tmp_path):
    """8 线程并发写 20 条, 验证 busy_timeout 下无 database is locked。"""
    store = DocumentStore(db_path=str(tmp_path / "docs.db"))

    def write(i: int):
        store.save(f"doc-{i:03d}", f"file-{i}.txt", status="processing")

    n = 20
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(write, range(n)))

    rows = store.list_all(page=1, size=n)["items"]
    assert len(rows) == n
