"""Prometheus 指标定义与收集。

暴露 /metrics 端点供 Prometheus 抓取。指标命名前缀 rag_。
"""
import time
import functools
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

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

# ── LLM 级别指标 ──
llm_tokens_total = Counter(
    "rag_llm_tokens_total",
    "LLM Token 消耗累计",
    ["type"],  # prompt / completion
)

llm_errors_total = Counter(
    "rag_llm_errors_total",
    "LLM 调用错误总数",
    ["error_type"],  # timeout / 5xx / rate_limit
)

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


def track_qa_latency(mode: str = "hybrid"):
    """装饰器：追踪 QA 请求延迟和状态"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                # 检查降级
                if isinstance(result, dict) and result.get("answer", "").startswith("（LLM"):
                    status = "fallback"
                return result
            except Exception:
                status = "error"
                raise
            finally:
                elapsed = time.time() - start
                qa_latency_seconds.observe(elapsed)
                qa_requests_total.labels(mode=mode, status=status).inc()
        return wrapper
    return decorator


def update_system_gauges():
    """更新系统级 Gauge 指标（在需要时调用）"""
    try:
        from api.state import get_vector_store, get_graph_rag
        store = get_vector_store()
        if store and hasattr(store, 'table'):
            try:
                vector_store_size.set(store.table.count_rows())
            except Exception:
                pass
        graph_rag = get_graph_rag()
        if graph_rag and hasattr(graph_rag, 'graph_store'):
            stats = graph_rag.graph_store.stats()
            graph_nodes_count.set(stats.get("nodes", 0))
            graph_edges_count.set(stats.get("edges", 0))
    except Exception:
        pass


def get_metrics():
    """生成 Prometheus 文本格式指标"""
    update_system_gauges()
    return generate_latest(REGISTRY)
