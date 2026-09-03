"""VectorStore.search 可观测性: 失败可计数, 空结果不误计 (保持 [] 契约)。"""
from unittest.mock import MagicMock

from engines.retrieval.vector_store import VectorStore, get_vector_search_error_count


def _make_vs(monkeypatch, table) -> VectorStore:
    """用假 db/table 构造 VectorStore, 不碰真实 LanceDB。"""
    fake_db = MagicMock()
    fake_db.open_table.return_value = table
    monkeypatch.setattr("engines.retrieval.vector_store.lancedb.connect", lambda *a, **k: fake_db)
    table.schema.names = []  # _cols 守卫: 无额外元数据列
    table.count_rows.return_value = 0  # _ensure_indices: <100 直接返回, 不建索引
    return VectorStore(uri="/tmp/fake_vs", dim=4, table_name="t")


def test_search_exception_returns_empty_and_counts(monkeypatch):
    bad = MagicMock()
    bad.search.return_value.metric.return_value.limit.return_value.to_list.side_effect = RuntimeError("boom")
    vs = _make_vs(monkeypatch, bad)

    before = get_vector_search_error_count()
    out = vs.search([0.1, 0.2, 0.3, 0.4], top_k=5)

    assert out == []  # 降级契约不变
    assert get_vector_search_error_count() == before + 1  # 失败可观测


def test_search_empty_table_does_not_count(monkeypatch):
    good = MagicMock()
    good.search.return_value.metric.return_value.limit.return_value.to_list.return_value = []
    vs = _make_vs(monkeypatch, good)

    before = get_vector_search_error_count()
    out = vs.search([0.1, 0.2, 0.3, 0.4], top_k=5)

    assert out == []  # 正常空结果
    assert get_vector_search_error_count() == before  # 空结果不误计为失败
