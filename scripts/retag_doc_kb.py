"""将某个文档从源 KB 表迁移到目标 KB 表（修正数据归属/误标）。

- 默认 dry-run：只报告会从哪个表搬多少行到哪个表，不写数据。
- --execute 才真正执行；执行前自动备份相关 .lance 目录到 backup/。
- 幂等：目标表若已存在该 doc 会先删再插；源表无该 doc 则视为已完成跳过。

用法:
  python scripts/retag_doc_kb.py --doc-id c3dbffff6a960cc1 --from-kb rag_service --to-kb rag_tech
  python scripts/retag_doc_kb.py --doc-id c3dbffff6a960cc1 --from-kb rag_service --to-kb rag_tech --execute
"""
import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import lancedb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retag_doc_kb")

URI = "./lancedb_data"
BACKUP_DIR = Path("./lancedb_data_backup")


def _table(db, kb: str):
    try:
        return db.open_table(kb)
    except Exception:
        return None


def _backup(kb: str):
    """执行前备份 .lance 目录（整体拷贝，便于回滚）。"""
    src = Path(URI) / f"{kb}.lance"
    if not src.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"{kb}.lance.{stamp}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    logger.info("已备份 %s -> %s", src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--from-kb", required=True)
    ap.add_argument("--to-kb", required=True)
    ap.add_argument("--clean", action="append", default=[], metavar="KB",
                    help="额外仅删除该 doc 的表（不插入），可重复；用于清理 legacy/documents 等陈旧副本")
    ap.add_argument("--execute", action="store_true", help="默认 dry-run，加此参数才真正写数据")
    args = ap.parse_args()

    db = lancedb.connect(URI)
    doc_id = args.doc_id

    src_tbl = _table(db, args.from_kb)
    dst_tbl = _table(db, args.to_kb)
    if src_tbl is None:
        logger.error("源表 %s 不存在，退出", args.from_kb)
        return
    if dst_tbl is None:
        logger.error("目标表 %s 不存在，退出", args.to_kb)
        return

    dim = None
    for f in dst_tbl.schema:
        if f.name == "embedding":
            dim = f.type.list_size
    logger.info("目标表 %s embedding 维度=%s", args.to_kb, dim)

    src_rows = src_tbl.search([0.0] * (dim or 512)).where(f"doc_id = '{doc_id}'", prefilter=True).to_list()
    dst_existing = dst_tbl.search([0.0] * (dim or 512)).where(f"doc_id = '{doc_id}'", prefilter=True).to_list()
    logger.info("源表 %s 含该 doc %d 行; 目标表 %s 已存在 %d 行", args.from_kb, len(src_rows), args.to_kb, len(dst_existing))

    if not src_rows:
        logger.info("源表无该 doc，可能已迁移完成。dry-run 结束。")
        return

    # 打印前 2 行内容片段供人工确认
    for r in src_rows[:2]:
        print(f"  chunk {r['id']}: {r['content'][:80]!r}")

    if not args.execute:
        msg = f"DRY-RUN：将把 {len(src_rows)} 行从 {args.from_kb} 迁到 {args.to_kb}"
        if args.clean:
            msg += f"；并清理额外表 {args.clean}"
        msg += "（--execute 才执行）"
        logger.info(msg)
        return

    # 执行：备份 -> 目标表删旧插新 -> 校验 -> 源表删除
    _backup(args.from_kb)
    _backup(args.to_kb)

    # 1) 目标表先清掉该 doc 旧数据（幂等）
    if dst_existing:
        dst_tbl.delete(f"doc_id = '{doc_id}'")
        logger.info("目标表已清除旧 %d 行", len(dst_existing))

    # 2) 插入全部行（只保留目标表 schema 内的列，剔除 _distance 等搜索计算字段）
    valid_cols = set(dst_tbl.schema.names)
    rows = [{k: v for k, v in r.items() if k in valid_cols} for r in src_rows]
    dst_tbl.add(rows)
    after_dst = dst_tbl.search([0.0] * (dim or 512)).where(f"doc_id = '{doc_id}'", prefilter=True).to_list()
    if len(after_dst) != len(src_rows):
        raise RuntimeError(f"目标表插入后行数不符: 期望 {len(src_rows)}, 实际 {len(after_dst)}")
    logger.info("目标表已写入 %d 行（校验通过）", len(after_dst))

    # 3) 源表删除该 doc
    src_tbl.delete(f"doc_id = '{doc_id}'")
    after_src = src_tbl.search([0.0] * (dim or 512)).where(f"doc_id = '{doc_id}'", prefilter=True).to_list()
    logger.info("源表剩余该 doc %d 行（应为 0）", len(after_src))

    # 4) 清理额外的陈旧副本表（如 legacy 'documents'）
    for extra in args.clean:
        et = _table(db, extra)
        if et is None:
            logger.warning("清理表 %s 不存在，跳过", extra)
            continue
        er = et.search([0.0] * (dim or 512)).where(f"doc_id = '{doc_id}'", prefilter=True).to_list()
        if er:
            et.delete(f"doc_id = '{doc_id}'")
            logger.info("已从 %s 清理陈旧副本 %d 行", extra, len(er))
        else:
            logger.info("清理表 %s 无该 doc，跳过", extra)

    logger.info("迁移完成: %s (%d 行) %s -> %s", doc_id, len(src_rows), args.from_kb, args.to_kb)


if __name__ == "__main__":
    main()
