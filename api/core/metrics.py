"""Prometheus 指标定义与收集。

暴露 /metrics 端点供 Prometheus 抓取。指标命名前缀 rag_。
"""

import logging
import math
from datetime import UTC, datetime

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

chunking_duration_seconds = Histogram(
    "rag_chunking_duration_seconds",
    "Chunking duration",
    ["doc_type"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")],
)

chunk_size_chars = Histogram(
    "rag_chunk_size_chars",
    "Chunk size distribution",
    ["doc_type"],
    # 原用 prometheus 默认 bucket (.005~10), 对"字符数"量纲完全不适用 —
    # 几乎所有样本都落进 +Inf, 分位数恒等于 10 字符 (假数据)。按常见
    # max_chars=480 的配置重新设定, 使 P50/P95 具备真实意义。
    buckets=[50, 100, 200, 300, 480, 600, 800, 1200, 2000, 5000, float("inf")],
)

chunks_per_document = Histogram(
    "rag_chunks_per_document",
    "Chunks per document",
    ["doc_type"],
    # 同上: 默认 bucket 上限 10, 稍长的文档全部落到 +Inf, 分位数失真。
    buckets=[1, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf")],
)

vector_store_errors_total = Counter(
    "rag_vector_store_errors_total",
    "Vector store errors",
    ["error_type"],
)

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

# BM25 稀疏索引与向量库的行数差(0=一致)。历史事故: BM25 索引格式不兼容被静默回退空索引,
# 混合检索长期退化为纯向量而仅有一行日志 —— 该 Gauge 把"数据完整性缺口"变成可告警的指标。
bm25_index_gap = Gauge(
    "rag_bm25_index_gap",
    "BM25 文档数与向量库行数差 (0=一致; >0=稀疏检索不完整)",
    ["kb"],
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

track_degradation_stage = Counter(
    "rag_degradation_stage_total",
    "按降级发生的阶段计数 (retrieval/rerank/llm), 与等级维度互补: 一次查询可同时命中多阶段",
    ["stage"],
)

# 降级阶段维度稳定标签集 (未触发的阶段也要以 0 出现, 保证前端图表维度不抖动)
DEGRADATION_STAGES: tuple[str, ...] = ("retrieval", "rerank", "llm")

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

# 语义近重复命中 = qa_cache_hits_total 的子集(精确哈希未中、同作用域 cosine 超阈值命中)。
# 单独计数才能量化语义层实际省下多少 LLM+rerank, 并据此调优阈值。
qa_cache_semantic_hits_total = Counter(
    "rag_qa_cache_semantic_hits_total",
    "QA 结果缓存近重复(语义)命中次数",
)


def get_metrics():
    """生成 Prometheus 文本格式指标。

    纯函数: 不做任何采集, 不依赖 api.state。系统级 Gauge 由
    api.core.system_stats.update_system_gauges() 在调用前刷新
    (见 main.py 的 /metrics 端点)。
    """
    return generate_latest(REGISTRY)


# ─────────────────── 只读汇总视图 (前端指标面板/降级可视化) ───────────────────
# /metrics 是 Prometheus 文本协议, 前端难以直接消费; 这里把同一份 REGISTRY
# 折算成 JSON。全程只读, 不修改任何指标状态; 任何采集失败一律降级为 0 而非抛错
# (指标是观测设施, 绝不能反过来影响服务可用性)。

# 降级等级语义 — 与 api/core/orchestrator.py 模块 docstring 的定义保持一致
DEGRADATION_LABELS: dict[str, str] = {
    "0": "正常",
    "1": "Rerank 跳过",
    "2": "向量失败仅 BM25",
    "3": "LLM 失败返回摘要",
}

# 前端图表维度稳定: 未触发的等级/来源/状态也要以 0 出现, 否则图表维度会随数据抖动
QA_STATUSES: tuple[str, ...] = ("success", "error", "fallback")
ROUTING_SOURCES: tuple[str, ...] = ("rule", "llm", "fallback", "manual")


def _fill_zero(observed: dict[str, int], known: tuple[str, ...]) -> dict[str, int]:
    """按 known 顺序输出, 缺失补 0; known 之外的实际取值追加在后 (保序去重)。"""
    return {k: observed.get(k, 0) for k in dict.fromkeys(known + tuple(sorted(observed)))}


def _collect_families() -> dict[str, dict]:
    """收集默认 REGISTRY 的全部指标族。

    返回 {指标名: {"type": str, "samples": [(sample_name, labels, value)]}}。
    histogram 的 sample_name 带 _bucket/_count/_sum 后缀, 由 _hist_stats 拆分。
    """
    families: dict[str, dict] = {}
    for metric in REGISTRY.collect():
        entry = {
            "type": metric.type,
            "samples": [(s.name, dict(s.labels), float(s.value)) for s in metric.samples],
        }
        families[metric.name] = entry
        # ⚠️ prometheus_client 命名约定: 定义为 "xxx_total" 的 Counter, 其
        # metric.name 会被剥掉 _total 后缀 ("xxx"), 但 sample name 仍带 _total。
        # 因此按定义名查族会静默落空 (返回 0 而非报错)。这里补一条别名索引,
        # 使调用方既能用 "xxx" 也能用 "xxx_total" 查到同一族。
        families.setdefault(metric.name + "_total", entry)
    return families


def _quantile(buckets: list[tuple[float, float]], count: float, q: float) -> float:
    """Prometheus histogram_quantile 的线性插值实现。

    受 bucket 边界粒度限制, 结果是插值估计值而非精确分位数 (与 Prometheus 一致)。
    """
    if not buckets or count <= 0:
        return 0.0
    rank = q * count
    prev_le, prev_cum = 0.0, 0.0
    for le, cum in buckets:
        if cum >= rank:
            span = cum - prev_cum
            if span <= 0:
                return round(le, 6)
            return round(prev_le + (le - prev_le) * ((rank - prev_cum) / span), 6)
        prev_le, prev_cum = le, cum
    return round(buckets[-1][0], 6)


def _hist_stats(samples, label_filter: dict | None = None) -> dict:
    """从 histogram 样本算 count/avg/P50/P95/P99 (保持原始单位, 通常是秒)。"""
    buckets: list[tuple[float, float]] = []
    count = 0.0
    total = 0.0
    for name, labels, value in samples:
        if label_filter and any(labels.get(k) != v for k, v in label_filter.items()):
            continue
        if name.endswith("_bucket"):
            try:
                le = float(labels.get("le", "nan"))
            except (TypeError, ValueError):
                continue
            if math.isinf(le):  # +Inf bucket 仅用于兜底计数, 不参与插值
                continue
            buckets.append((le, value))
        elif name.endswith("_count"):
            count += value
        elif name.endswith("_sum"):
            total += value
    buckets.sort(key=lambda x: x[0])
    return {
        "count": int(count),
        "avg": round(total / count, 6) if count else 0.0,
        "p50": _quantile(buckets, count, 0.50),
        "p95": _quantile(buckets, count, 0.95),
        "p99": _quantile(buckets, count, 0.99),
    }


def _to_ms(stats: dict) -> dict:
    """秒 → 毫秒 (前端展示统一用 ms)。"""
    return {
        "count": stats["count"],
        "avg_ms": round(stats["avg"] * 1000, 2),
        "p50_ms": round(stats["p50"] * 1000, 2),
        "p95_ms": round(stats["p95"] * 1000, 2),
        "p99_ms": round(stats["p99"] * 1000, 2),
    }


def _counter_total(samples) -> int:
    """Counter 总量 (跳过 _created 时间戳样本)。"""
    return int(sum(value for name, _labels, value in samples if not name.endswith("_created")))


def _counter_by_label(samples, key: str) -> dict[str, int]:
    """按指定 label 聚合 Counter: {label值: 计数}。"""
    acc: dict[str, float] = {}
    for name, labels, value in samples:
        if name.endswith("_created"):
            continue
        k = labels.get(key) or "unknown"
        acc[k] = acc.get(k, 0.0) + value
    return {k: int(v) for k, v in acc.items()}


def _label_values(samples, key: str) -> list[str]:
    """取某个 label 在 Counter 上出现过的全部取值。"""
    seen = {labels.get(key) for _name, labels, _value in samples if not _name.endswith("_created")}
    return sorted(v for v in seen if v)


def _gauge_value(samples) -> float:
    vals = [value for _name, _labels, value in samples]
    return vals[0] if vals else 0.0


def _rate(hits: int, misses: int) -> float:
    total = hits + misses
    return round(hits / total, 4) if total > 0 else 0.0


def metrics_summary() -> dict:
    """只读: 把 Prometheus 指标折算成前端可直接消费的 JSON 汇总。

    覆盖请求量/成功率、延迟与阶段耗时、质量信号(top1 分数/忠实度)、
    降级等级分布、路由来源分布、缓存命中率、跨库兜底、系统规模,
    以及按文档类型的真实切分统计 (与 /documents/type-strategies 的 type_id 可 join)。
    """
    try:
        families = _collect_families()
    except Exception as e:  # noqa: BLE001 — 指标汇总失败不应影响服务
        logger.warning("指标汇总采集失败: %s", str(e)[:120])
        return {"ok": False, "reason": "metrics_unavailable"}

    def samples_of(name: str):
        family = families.get(name)
        return family["samples"] if family else []

    # ── 请求量与成功率 (未出现的状态补 0) ──
    by_status = _fill_zero(_counter_by_label(samples_of("rag_qa_requests_total"), "status"), QA_STATUSES)
    req_total = sum(by_status.values())
    req_ok = by_status.get("success", 0)

    # ── 降级分布 (0 正常 / 1 rerank 跳过 / 2 仅 BM25 / 3 LLM 失败; 未触发的等级补 0) ──
    deg_levels = _fill_zero(
        _counter_by_label(samples_of("rag_degradation_levels_total"), "level"),
        tuple(DEGRADATION_LABELS),
    )
    deg_total = sum(deg_levels.values())
    deg_degraded = deg_total - deg_levels.get("0", 0)

    # ── 降级阶段维度 (与等级维度互补: 一次查询可同时经历多阶段) ──
    deg_stages = _fill_zero(
        _counter_by_label(samples_of("rag_degradation_stage_total"), "stage"),
        DEGRADATION_STAGES,
    )

    # ── 路由来源分布 ──
    route_sources = _fill_zero(
        _counter_by_label(samples_of("rag_routing_sources_total"), "source"), ROUTING_SOURCES
    )

    # ── 缓存命中率 ──
    embed_hits = _counter_total(samples_of("rag_embed_cache_hits_total"))
    embed_misses = _counter_total(samples_of("rag_embed_cache_misses_total"))
    qa_hits = _counter_total(samples_of("rag_qa_cache_hits_total"))
    qa_misses = _counter_total(samples_of("rag_qa_cache_misses_total"))
    qa_semantic_hits = _counter_total(samples_of("rag_qa_cache_semantic_hits_total"))

    # ── 按文档类型的真实切分统计 (label doc_type 即 type_id, 可与策略表 join) ──
    chunking_samples = samples_of("rag_chunk_size_chars")
    chunking_by_type: dict[str, dict] = {}
    for doc_type in _label_values(chunking_samples, "doc_type"):
        size_stats = _hist_stats(chunking_samples, {"doc_type": doc_type})
        per_doc = _hist_stats(samples_of("rag_chunks_per_document"), {"doc_type": doc_type})
        duration = _hist_stats(samples_of("rag_chunking_duration_seconds"), {"doc_type": doc_type})
        chunking_by_type[doc_type] = {
            "docs": per_doc["count"],
            "chunks": size_stats["count"],
            "avg_chunks_per_doc": per_doc["avg"],
            "avg_chunk_chars": round(size_stats["avg"], 1),
            "p95_chunk_chars": round(size_stats["p95"], 1),
            "avg_duration_ms": round(duration["avg"] * 1000, 2),
        }

    return {
        "ok": True,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "requests": {
            "total": req_total,
            "by_status": by_status,
            "success_rate": round(req_ok / req_total, 4) if req_total else 0.0,
        },
        "latency_ms": _to_ms(_hist_stats(samples_of("rag_qa_latency_seconds"))),
        "stages_ms": {
            "retrieval": _to_ms(_hist_stats(samples_of("rag_retrieval_latency_seconds"))),
            "rerank": _to_ms(_hist_stats(samples_of("rag_rerank_latency_seconds"))),
            "llm": _to_ms(_hist_stats(samples_of("rag_llm_latency_seconds"))),
        },
        "quality": {
            "faithfulness": _hist_stats(samples_of("rag_qa_faithfulness")),
            "top1_score": _hist_stats(samples_of("rag_qa_top1_score")),
            "retrieved_docs": _hist_stats(samples_of("rag_retrieved_docs_count")),
        },
        "degradation": {
            "levels": deg_levels,
            "level_labels": DEGRADATION_LABELS,
            "stages": deg_stages,
            "total": deg_total,
            "degraded": deg_degraded,
            "degraded_rate": round(deg_degraded / deg_total, 4) if deg_total else 0.0,
        },
        "routing": {
            "sources": route_sources,
            "total": sum(route_sources.values()),
        },
        "cache": {
            "embed_hits": embed_hits,
            "embed_misses": embed_misses,
            "embed_hit_rate": _rate(embed_hits, embed_misses),
            "qa_hits": qa_hits,
            "qa_misses": qa_misses,
            "qa_hit_rate": _rate(qa_hits, qa_misses),
            # 语义近重复命中数及其在总命中中的占比: 占比高说明精确哈希之外确有大量
            # 措辞变体被短路(省 LLM+rerank); 占比过高则需复核阈值是否过松。
            "qa_semantic_hits": qa_semantic_hits,
            "qa_semantic_share_of_hits": _rate(qa_semantic_hits, qa_hits),
        },
        "cross_kb_fallback": {
            "total": _counter_total(samples_of("rag_cross_kb_fallback_total")),
        },
        "chunking": {"by_type": chunking_by_type},
        "system": {
            "vector_store_size": int(_gauge_value(samples_of("rag_vector_store_size"))),
            "graph_nodes": int(_gauge_value(samples_of("rag_graph_nodes_count"))),
            "graph_edges": int(_gauge_value(samples_of("rag_graph_edges_count"))),
        },
    }
