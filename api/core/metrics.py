"""Prometheus 指标定义与收集。

暴露 /metrics 端点供 Prometheus 抓取。指标命名前缀 rag_。
"""

import logging

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

# ── API 级别指标 ──
qa_requests_total = Counter(
    "rag_qa_requests_total",
    "QA 请求总数",
    ["mode", "status"],  # status: success / error / fallback
)

qa_latency_seconds = Histogram(
    "rag_qa_latency_seconds",
    "QA 请求延迟 (秒)",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

document_uploads_total = Counter(
    "rag_document_uploads_total",
    "文档上传总数",
    ["status"],
)

# ── QA 质量信号 (RAG 上线核心可观测面, 现有指标体系的真实缺口) ──
# 让 /metrics 能监控"回答靠不靠谱"的劣化趋势, 而非仅 QPS/延迟。
# 注: 降级/兜底次数已由 rag_degradation_levels_total 覆盖, 此处不重复建 counter。
qa_faithfulness = Histogram(
    "rag_qa_faithfulness",
    "回答忠实度分数 (0-1, 越高幻觉风险越低)",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)

qa_top1_score = Histogram(
    "rag_qa_top1_score",
    "首条检索相关性分数 (0-1)",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)

# ── 检索级别指标 ──
retrieval_rounds = Histogram(
    "rag_retrieval_rounds",
    "Self-Retrieval 轮数分布",
    buckets=[1, 2, 3, 4, 5],
)

retrieved_docs_count = Histogram(
    "rag_retrieved_docs_count",
    "每次检索返回文档数",
    buckets=[0, 1, 5, 10, 20, 50],
)

# ── LLM 级别指标 (定义下沉 engines/common/metrics.py, 引擎层复用; 此处 re-export 保持调用方兼容) ──
from engines.common.metrics import llm_errors_total, llm_tokens_total  # noqa: F401

# ── 系统级别指标 ──
vector_store_size = Gauge(
    "rag_vector_store_size",
    "向量库中的向量数量",
)

graph_nodes_count = Gauge(
    "rag_graph_nodes_count",
    "知识图谱节点数",
)

graph_edges_count = Gauge(
    "rag_graph_edges_count",
    "知识图谱边数",
)

# ── Router / Skill 级别指标 ──
track_routing = Counter(
    "rag_routing_sources_total",
    "路由来源计数 (rule/llm/fallback/manual)",
    ["source", "skill"],
)

track_degradation = Counter(
    "rag_degradation_levels_total",
    "按降级等级计数 (0/1/2/3)",
    ["level"],
)

track_retrieval_latency = Histogram(
    "rag_retrieval_latency_seconds",
    "检索阶段耗时 (秒)",
    buckets=[0.1, 0.2, 0.4, 0.8, 1.5, 3.0],
)

track_rerank_latency = Histogram(
    "rag_rerank_latency_seconds",
    "Rerank 阶段耗时 (秒)",
    buckets=[0.05, 0.1, 0.2, 0.4, 0.8],
)

track_llm_latency = Histogram(
    "rag_llm_latency_seconds",
    "LLM 生成阶段耗时 (秒)",
    buckets=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
)

cross_kb_fallback_total = Counter(
    "rag_cross_kb_fallback_total",
    "跨库兜底触发次数",
    ["from_kb", "to_kb"],
)

embed_cache_hits_total = Counter(
    "rag_embed_cache_hits_total",
    "Query Embedding 缓存命中次数",
)

embed_cache_misses_total = Counter(
    "rag_embed_cache_misses_total",
    "Query Embedding 缓存未命中次数",
)

qa_cache_hits_total = Counter(
    "rag_qa_cache_hits_total",
    "QA 结果缓存命中次数",
)

qa_cache_misses_total = Counter(
    "rag_qa_cache_misses_total",
    "QA 结果缓存未命中次数",
)


def update_system_gauges():
    """更新系统级 Gauge 指标（在需要时调用）"""
    try:
        from api.state import get_graph_rag, get_vector_store

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


def get_metrics():
    """生成 Prometheus 文本格式指标"""
    update_system_gauges()
    return generate_latest(REGISTRY)
