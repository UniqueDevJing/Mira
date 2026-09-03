"""为所有非零生产向量表补齐向量/标量索引 (运维/迁移工具, 幂等)。

背景: LanceDB 0.36 无向量索引时 search(vector) 对近似查询返回 0 行, 导致小表
(rag_tech/rag_policy/rag_service 等) 完全不可检索。本脚本对所有非零表补建
IvfHnswFlat(cosine) 向量索引 + doc_id/parent_id 标量索引, 修复历史入库未建索引的表。

用法: python scripts/ensure_vector_indexes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import lancedb
from lancedb.index import BTree, IvfHnswFlat

from api.config import settings


def _has_index(tbl, col):
    try:
        return any(col in (getattr(ic, "columns", None) or []) for ic in tbl.list_indices())
    except Exception:  # noqa: BLE001
        return False


def main():
    db = lancedb.connect(settings.vector_uri)
    tables = db.table_names()
    print(f"发现 {len(tables)} 张表: {tables}")
    for name in tables:
        tbl = db.open_table(name)
        try:
            n = tbl.count_rows()
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}] 无法读取行数, 跳过: {e}")
            continue
        if n < 1:
            print(f"  [{name}] 空表({n} 行), 跳过 (需先入库数据)")
            continue
        # 向量索引
        if not _has_index(tbl, "embedding"):
            try:
                tbl.create_index("embedding", config=IvfHnswFlat(distance_type="cosine"))
                print(f"  [{name}] 已建向量索引 (IvfHnswFlat, {n} 行)")
            except Exception as e:  # noqa: BLE001
                print(f"  [{name}] 向量索引创建失败: {str(e)[:150]}")
        else:
            print(f"  [{name}] 向量索引已存在")
        # 标量索引
        for col in ("doc_id", "parent_id"):
            cols = [f.name for f in tbl.schema]
            if col in cols and not _has_index(tbl, col):
                try:
                    tbl.create_index(col, config=BTree())
                    print(f"  [{name}] 已建标量索引 ({col})")
                except Exception as e:  # noqa: BLE001
                    print(f"  [{name}] 标量索引({col})失败(可忽略): {str(e)[:100]}")


if __name__ == "__main__":
    main()
