#!/usr/bin/env python
"""从向量库重建 BM25 稀疏索引 —— P0 数据一致性修复。

背景
----
BM25 历史版本用 **pickle** 落盘, 现行实现用 **JSON** 读取（见 engines/retrieval/bm25_index.py）。
格式不匹配导致每次启动都在 json.load 抛 UnicodeDecodeError, 被 except 静默吞掉后
**回退空索引**；此后 ingest 只会往这个空索引里追加并覆盖落盘, 于是历史累积被反复丢弃。
结果: 生产长期跑在"BM25 缺失"的残缺状态, 混合检索退化成纯向量, 而日志里只有一条 warning。

实测缺口（重建前）:
    documents    向量 263  → 无索引文件   (默认 kb, 最大的库)
    rag_service  向量  47  → 仅 18 篇
    rag_tech     向量  36  → 无索引文件
    rag_policy   向量   6  →  6 篇 (正常)

格式不兼容已由 bm25_index._read_persisted() 修复（兼容读取 + 迁移 JSON）,
但**已丢失的历史数据无法自动恢复**, 必须由本脚本从向量库（权威数据源）重建。

用法
----
    python scripts/rebuild_bm25.py --dry-run        # 只报告缺口, 不落盘（建议先跑）
    python scripts/rebuild_bm25.py                  # 重建全部不一致的知识库
    python scripts/rebuild_bm25.py --kb tech        # 只重建指定库
    python scripts/rebuild_bm25.py --force          # 一致也强制重建
    python scripts/rebuild_bm25.py --no-backup      # 不备份旧索引

说明: 直接读 lancedb, 不经过 api.state, 因此**不会加载 embedding 模型**, 执行很快。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import lancedb

from engines.retrieval.bm25_index import Bm25Index

DATA_DIR = ROOT / "data"
VECTOR_URI = str(ROOT / "lancedb_data")

# 与 api/state.py:79-80 保持一致: kb ""/"documents" → 表 documents; 其余 → rag_<kb>
LEGACY_KBS = {"", "documents"}


def table_to_kb(table: str) -> str:
    return "documents" if table in LEGACY_KBS else table.removeprefix("rag_")


def index_path(kb: str) -> Path:
    # 与 api/state.py get_bm25_index 保持一致: JSON 格式, 故后缀 .json
    return DATA_DIR / f"bm25_{kb}.json"


def existing_bm25_count(path: Path) -> int | None:
    """现有索引的文档数; 无文件或不可读时返回 None。"""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return len(json.load(f)["docs"])
    except Exception:
        try:  # 历史 pickle 格式
            import pickle

            with open(path, "rb") as f:
                return len(pickle.load(f)["docs"])
        except Exception:
            return None


def load_chunks(table: str) -> list[dict]:
    """从向量库读取全部 chunk, 按 ingest 相同格式构造 BM25 文档。

    字段与 api/routes/documents.py:286-288 严格一致（id/chunk_id/doc_id/content）,
    确保重建结果与后续增量 ingest 完全兼容。
    """
    db = lancedb.connect(VECTOR_URI)
    tb = db.open_table(table)
    df = tb.to_pandas()
    docs = []
    for row in df.to_dict("records"):
        content = row.get("content") or ""
        if not content.strip():
            continue
        cid = row.get("id") or ""
        docs.append(
            {
                "id": cid,
                "chunk_id": cid,
                "doc_id": row.get("doc_id") or "",
                "content": content,
            }
        )
    return docs


def rebuild(kb: str, docs: list[dict], path: Path, backup: bool = True) -> int:
    """重建单个库的索引。返回新索引文档数。"""
    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    # 先删旧文件: 否则 Bm25Index 会加载旧内容, add_documents 变成追加而非重建
    if path.exists():
        path.unlink()
    idx = Bm25Index(persist_path=str(path))
    if docs:
        idx.add_documents(docs)
    return len(idx)


def main() -> int:
    ap = argparse.ArgumentParser(description="从向量库重建 BM25 稀疏索引")
    ap.add_argument("--kb", action="append", default=None, help="只处理指定知识库(可重复); 默认全部")
    ap.add_argument("--dry-run", action="store_true", help="只报告缺口, 不落盘")
    ap.add_argument("--force", action="store_true", help="即使数量一致也强制重建")
    ap.add_argument("--no-backup", action="store_true", help="不备份旧索引")
    args = ap.parse_args()

    db = lancedb.connect(VECTOR_URI)
    tables = db.table_names()

    print(f"{'知识库':<12} {'向量':>6} {'旧BM25':>8} {'新BM25':>8}  状态")
    print("-" * 56)

    planned = []
    for t in tables:
        kb = table_to_kb(t)
        if args.kb and kb not in args.kb:
            continue
        n_vec = db.open_table(t).count_rows()
        old = existing_bm25_count(index_path(kb))
        old_s = "无索引" if old is None else str(old)

        if n_vec == 0 and old is None:
            print(f"{kb:<12} {n_vec:>6} {old_s:>8} {'-':>8}  空库, 跳过")
            continue

        if args.dry_run:
            ok = (old == n_vec)
            print(f"{kb:<12} {n_vec:>6} {old_s:>8} {'-':>8}  {'一致' if ok else '需重建'}")
            if not ok or args.force:
                planned.append(kb)
            continue

        if old == n_vec and not args.force:
            print(f"{kb:<12} {n_vec:>6} {old_s:>8} {'跳过':>8}  已一致")
            continue

        docs = load_chunks(t)
        t0 = time.time()
        new_n = rebuild(kb, docs, index_path(kb), backup=not args.no_backup)
        mark = "OK" if new_n == n_vec else "不符!"
        print(f"{kb:<12} {n_vec:>6} {old_s:>8} {new_n:>8}  {mark} ({time.time() - t0:.1f}s)")

    if args.dry_run:
        todo = [k for k in planned]
        print(f"\n[dry-run] 需重建 {len(todo)} 个知识库: {todo or '无'}")
        return 0

    # 复核
    print("\n复核：")
    bad = []
    for t in tables:
        kb = table_to_kb(t)
        if args.kb and kb not in args.kb:
            continue
        n_vec = db.open_table(t).count_rows()
        now = existing_bm25_count(index_path(kb)) or 0
        if now != n_vec:
            bad.append((kb, n_vec, now))
        print(f"  {kb:<12} 向量={n_vec:<5} BM25={now:<5} {'OK' if now == n_vec else '不一致'}")

    if bad:
        print(f"\nFAIL: {len(bad)} 个库仍不一致 {bad}")
        return 1
    print("\nRESULT: OK — 全部知识库 BM25 与向量库一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
