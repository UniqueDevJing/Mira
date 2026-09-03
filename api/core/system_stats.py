"""系统级 Gauge 采集 — 从 metrics.py 拆出, 修分层倒置。

原 `update_system_gauges` 定义在 metrics.py 内并 `from api.state import ...`,
导致最底层的纯指标设施反向依赖了上层的单例容器。本模块位于两者之上,
由 main.py 在 /metrics 端点显式调用, 依赖方向变为:

    main → system_stats → {state, metrics}

metrics.py 重新成为零业务依赖的纯设施, 且不会再有 metrics ↔ state 的成环风险。
"""

import logging

from api.core.metrics import (
    graph_edges_count,
    graph_nodes_count,
    vector_store_size,
)
from api.state import get_graph_rag, get_vector_store

logger = logging.getLogger(__name__)


def update_system_gauges() -> None:
    """采集向量库行数与图谱规模。采集失败一律静默 — 指标不应影响服务。"""
    try:
        store = get_vector_store()
        if store and hasattr(store, "table"):
            try:
                vector_store_size.set(store.table.count_rows())
            except Exception as e:  # noqa: BLE001 — 指标采集失败不影响服务
                logger.debug("向量库行数采集失败: %s", str(e)[:80])
        graph_rag = get_graph_rag()
        if graph_rag and hasattr(graph_rag, "graph_store"):
            stats = graph_rag.graph_store.stats()
            graph_nodes_count.set(stats.get("nodes", 0))
            graph_edges_count.set(stats.get("edges", 0))
    except Exception as e:  # noqa: BLE001 — 指标采集失败不影响服务
        logger.debug("系统指标采集失败: %s", str(e)[:80])
