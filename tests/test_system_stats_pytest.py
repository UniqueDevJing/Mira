"""api/core/system_stats.py —— 系统级 Gauge 采集。

该模块的设计目标是「指标采集绝不影响服务」，因此存在多层降级路径，
全部需要被测试覆盖：

1. 正常路径        —— 同时采到向量库行数与图谱规模
2. 分支降级        —— count_rows() 抛错 / stats() 抛错 / 缺少预期属性 / 单例为 None
3. 外层兜底        —— 连获取单例本身抛错也必须吞掉

Gauge 为进程级 REGISTRY 累计对象，其他测试会往同一 REGISTRY 灌值。
_reset_metric_registry fixture 在每个用例前 unregister 全部 collector 并
reload 模块重建全新指标对象，使每个用例从干净态开始 ——
这是 prometheus_client 在共享进程下的标准测试隔离手法（与
test_metrics_summary_pytest.py 一致）。
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

import api.core.metrics as _metrics
import api.core.system_stats as ss


@pytest.fixture(autouse=True)
def _reset_metric_registry():
    reg = _metrics.REGISTRY
    for collector in list(reg._collector_to_names):
        reg.unregister(collector)
    importlib.reload(_metrics)
    importlib.reload(ss)
    yield


def _gauge_value(gauge):
    """从 Gauge 采集当前值。

    prometheus_client 的 Gauge 通过内部 _value 暴露单值，
    这里集中封装避免测试里散落私有属性访问。
    """
    return gauge._value.get()


def _make_store(rows=42):
    store = MagicMock()
    store.table.count_rows.return_value = rows
    return store


def _make_graph(nodes=7, edges=11):
    graph = MagicMock()
    graph.graph_store.stats.return_value = {"nodes": nodes, "edges": edges}
    return graph


# ────────────────────────── 正常路径 ──────────────────────────


def test_collects_vector_and_graph_gauges(monkeypatch):
    monkeypatch.setattr(ss, "get_vector_store", lambda: _make_store(rows=42))
    monkeypatch.setattr(ss, "get_graph_rag", lambda: _make_graph(nodes=7, edges=11))

    ss.update_system_gauges()

    assert _gauge_value(ss.vector_store_size) == 42
    assert _gauge_value(ss.graph_nodes_count) == 7
    assert _gauge_value(ss.graph_edges_count) == 11


def test_stats_missing_keys_default_to_zero(monkeypatch):
    """stats() 未返回 nodes/edges 时回落 0，而不是 KeyError。"""
    graph = MagicMock()
    graph.graph_store.stats.return_value = {}
    monkeypatch.setattr(ss, "get_vector_store", lambda: _make_store())
    monkeypatch.setattr(ss, "get_graph_rag", lambda: graph)

    ss.update_system_gauges()

    assert _gauge_value(ss.graph_nodes_count) == 0
    assert _gauge_value(ss.graph_edges_count) == 0


# ────────────────────────── 分支降级 ──────────────────────────


def test_count_rows_failure_does_not_block_graph(monkeypatch):
    """向量库行数采集失败只记 debug，图谱采集仍要继续。"""
    store = MagicMock()
    store.table.count_rows.side_effect = RuntimeError("lancedb unavailable")
    monkeypatch.setattr(ss, "get_vector_store", lambda: store)
    monkeypatch.setattr(ss, "get_graph_rag", lambda: _make_graph(nodes=7, edges=11))

    ss.update_system_gauges()

    # 关键断言：图谱指标仍被正确采集，说明失败被隔离在向量库分支内
    assert _gauge_value(ss.graph_nodes_count) == 7
    assert _gauge_value(ss.graph_edges_count) == 11


def test_store_without_table_attribute_is_skipped(monkeypatch):
    store = MagicMock(spec=[])  # 无 table 属性
    monkeypatch.setattr(ss, "get_vector_store", lambda: store)
    monkeypatch.setattr(ss, "get_graph_rag", lambda: _make_graph())

    ss.update_system_gauges()

    assert _gauge_value(ss.graph_nodes_count) == 7


def test_graph_stats_failure_is_swallowed(monkeypatch):
    """图谱统计失败不应向上抛 —— 指标失败绝不能拖垮 /metrics 端点。"""
    graph = MagicMock()
    graph.graph_store.stats.side_effect = RuntimeError("graph store down")
    monkeypatch.setattr(ss, "get_vector_store", lambda: _make_store(rows=42))
    monkeypatch.setattr(ss, "get_graph_rag", lambda: graph)

    ss.update_system_gauges()

    # 向量库指标已落库，说明流程走到图谱这一步才失败
    assert _gauge_value(ss.vector_store_size) == 42


def test_graph_without_graph_store_is_skipped(monkeypatch):
    graph = MagicMock(spec=[])
    monkeypatch.setattr(ss, "get_vector_store", lambda: _make_store())
    monkeypatch.setattr(ss, "get_graph_rag", lambda: graph)

    ss.update_system_gauges()

    assert _gauge_value(ss.vector_store_size) == 42


def test_none_vector_store_still_collects_graph(monkeypatch):
    """单例尚未初始化时返回 None 属正常启动阶段，不应报错。"""
    monkeypatch.setattr(ss, "get_vector_store", lambda: None)
    monkeypatch.setattr(ss, "get_graph_rag", lambda: _make_graph(nodes=3, edges=5))

    ss.update_system_gauges()

    assert _gauge_value(ss.graph_nodes_count) == 3


def test_both_singletons_none_is_safe(monkeypatch):
    monkeypatch.setattr(ss, "get_vector_store", lambda: None)
    monkeypatch.setattr(ss, "get_graph_rag", lambda: None)

    ss.update_system_gauges()  # 不应抛 AttributeError


# ────────────────────────── 外层兜底 ──────────────────────────


def test_singleton_accessor_raising_is_swallowed(monkeypatch):
    """最外层兜底：连获取单例本身抛错也不应影响服务。"""
    monkeypatch.setattr(
        ss, "get_vector_store", MagicMock(side_effect=RuntimeError("singleton not ready"))
    )
    monkeypatch.setattr(ss, "get_graph_rag", lambda: None)

    ss.update_system_gauges()  # 关键：不抛异常


def test_update_is_idempotent_and_repeatable(monkeypatch):
    """重复调用不应累积或破坏状态（/metrics 会被周期性轮询）。"""
    monkeypatch.setattr(ss, "get_vector_store", lambda: _make_store(rows=42))
    monkeypatch.setattr(ss, "get_graph_rag", lambda: _make_graph())

    ss.update_system_gauges()
    ss.update_system_gauges()

    assert _gauge_value(ss.vector_store_size) == 42