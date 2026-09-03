"""mounted_kbs 不应被 lancedb 注册表(table_names)脱节拖累。

回归防护: 早期实现用 `tn in table_names()` 做门槛, 导致已入库却未登记在注册表的
表(如 rag_service/rag_tech)被永久排除出路由候选 → 库"假死"(数据完好却查不到)。
修复后改为逐 kb open_table + count_rows, 注册表脱节不再影响挂载。
"""
from unittest.mock import MagicMock

import lancedb

import api.state as st
from engines.doc_types import RAG_KBS


def test_mounted_kbs_recovers_tables_outside_registry(monkeypatch):
    # 模拟注册表脱节: list_tables/table_names 都返回空, 但表实际可打开且有数据
    fake_db = MagicMock()

    def _open(name):
        t = MagicMock()
        t.count_rows.return_value = 5  # 非空
        return t

    fake_db.open_table.side_effect = _open
    fake_db.list_tables.return_value = []  # 注册表"看不见"任何表
    fake_db.table_names.return_value = []

    monkeypatch.setattr(lancedb, "connect", lambda *a, **k: fake_db)
    monkeypatch.setattr(st, "_mounted_kbs_cache", {})

    mounted = st.mounted_kbs()
    assert mounted, "非空表应被挂载, 不应因注册表脱节而全空"
    # 注册表脱节下, 真实存在的 service/tech 库必须被恢复挂载
    assert "service" in mounted
    assert "tech" in mounted
    # 全部 RAG_KBS 均可达(模拟环境下每个都非空)
    assert set(RAG_KBS).issubset(set(mounted))
