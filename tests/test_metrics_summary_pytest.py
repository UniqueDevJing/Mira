"""metrics_summary() 计算正确性 — 守护 /api/v1/metrics/summary 只读汇总接口。

两类断言:
  1) 空 REGISTRY: 结构完整、全 0、不抛错 (健壮性)
  2) 注入已知观测: 关键字段与手算结果逐一比对 (计算正确性, 而非仅"不报错")

本测试是两条真实缺陷的回归防线:
  · prometheus_client 会剥掉 Counter 名末尾的 _total 作为 metric.name,
    按定义名查族会**静默落空返回 0** (不报错, 前端拿到全零假数据)。
  · chunk_size_chars / chunks_per_document 原用 prometheus 默认 bucket
    (0.005~10), 对"字符数/块数"量纲完全无效, 样本全落 +Inf, 分位数恒等于
    上限值 — 看似有数, 实为假数据。

⚠️ 跨测试隔离: 指标为进程级 REGISTRY 累计, 全量测试时其他测试 (真实 API/路由/QA)
会往同一 REGISTRY 灌观测值, 导致本文件断言的精确值失真。_reset_metric_registry
fixture 在每个指标测试前 unregister 全部 collector 并 reload 模块重建全新指标对象,
使每个测试从干净的累计态开始 —— 这是 prometheus_client 在共享进程下的标准测试隔离手法。
"""

import importlib

import pytest

import api.core.metrics as _metrics


@pytest.fixture(autouse=True)
def _reset_metric_registry():
    """每指标测试前清空进程级 REGISTRY 并重建指标对象, 隔离其他测试的观测值污染。"""
    reg = _metrics.REGISTRY
    for collector in list(reg._collector_to_names):
        reg.unregister(collector)
    importlib.reload(_metrics)
    yield


def test_metrics_summary_empty_registry():
    """空 REGISTRY: 不抛错, 结构完整, 维度齐全 (未触发的等级/来源也要补 0)。"""
    s = _metrics.metrics_summary()
    assert s["ok"] is True
    assert s["requests"]["total"] == 0
    assert s["latency_ms"]["count"] == 0
    # 前端图表维度稳定: 未出现过的等级/来源/状态也要以 0 出现
    assert sorted(s["degradation"]["levels"]) == ["0", "1", "2", "3"]
    assert sorted(s["degradation"]["stages"]) == ["llm", "rerank", "retrieval"]
    assert s["degradation"]["stages"]["llm"] == 0
    assert sorted(s["routing"]["sources"]) == ["fallback", "llm", "manual", "rule"]
    assert s["chunking"]["by_type"] == {}


def test_metrics_summary_calculation():
    """注入已知观测 → 与手算结果逐一比对。"""
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got={got!r} want={want!r}")

    # 延迟 0.1/0.2/0.3/1.0/2.0 秒 → 手算 avg=0.72s, P50=0.375s, P95=1.75s, P99=1.95s
    for v in (0.1, 0.2, 0.3, 1.0, 2.0):
        _metrics.qa_latency_seconds.observe(v)
    for _ in range(3):
        _metrics.qa_requests_total.labels(mode="hybrid", status="success").inc()
    _metrics.qa_requests_total.labels(mode="hybrid", status="error").inc()
    for lvl, n in (("0", 5), ("1", 2), ("2", 1)):  # level=3 一次都没触发
        for _ in range(n):
            _metrics.track_degradation.labels(level=lvl).inc()
    for stg, n in (("retrieval", 3), ("rerank", 1), ("llm", 2)):
        for _ in range(n):
            _metrics.track_degradation_stage.labels(stage=stg).inc()
    for src, n in (("rule", 4), ("llm", 1)):
        for _ in range(n):
            _metrics.track_routing.labels(source=src, skill="qa").inc()
    for _ in range(7):
        _metrics.embed_cache_hits_total.inc()
    for _ in range(3):
        _metrics.embed_cache_misses_total.inc()
    # 块大小 100/200/300 字符 → avg=200; 修正 bucket 后手算 P95=285
    for v in (100, 200, 300):
        _metrics.chunk_size_chars.labels(doc_type="contract").observe(v)
    _metrics.chunks_per_document.labels(doc_type="contract").observe(3)

    s = _metrics.metrics_summary()

    check("latency.count", s["latency_ms"]["count"], 5)
    check("latency.avg_ms", s["latency_ms"]["avg_ms"], 720.0)
    check("latency.p50_ms", s["latency_ms"]["p50_ms"], 375.0)
    check("latency.p95_ms", s["latency_ms"]["p95_ms"], 1750.0)
    check("latency.p99_ms", s["latency_ms"]["p99_ms"], 1950.0)

    check("requests.total", s["requests"]["total"], 4)
    check("requests.success", s["requests"]["by_status"]["success"], 3)
    check("requests.error", s["requests"]["by_status"]["error"], 1)
    check("requests.fallback 补 0", s["requests"]["by_status"]["fallback"], 0)
    check("requests.success_rate", s["requests"]["success_rate"], 0.75)

    check("degradation.total", s["degradation"]["total"], 8)
    check("degradation.L0", s["degradation"]["levels"]["0"], 5)
    check("degradation.L1", s["degradation"]["levels"]["1"], 2)
    check("degradation.L2", s["degradation"]["levels"]["2"], 1)
    check("degradation.L3 未触发补 0", s["degradation"]["levels"]["3"], 0)
    check("degradation.degraded", s["degradation"]["degraded"], 3)
    check("degradation.degraded_rate", s["degradation"]["degraded_rate"], 0.375)
    check("degradation.level_labels", s["degradation"]["level_labels"]["1"], "Rerank 跳过")
    check("degradation.stages.retrieval", s["degradation"]["stages"]["retrieval"], 3)
    check("degradation.stages.rerank", s["degradation"]["stages"]["rerank"], 1)
    check("degradation.stages.llm", s["degradation"]["stages"]["llm"], 2)

    check("routing.total", s["routing"]["total"], 5)
    check("routing.rule", s["routing"]["sources"]["rule"], 4)
    check("routing.llm", s["routing"]["sources"]["llm"], 1)
    check("routing.manual 补 0", s["routing"]["sources"]["manual"], 0)

    check("cache.embed_hit_rate", s["cache"]["embed_hit_rate"], 0.7)

    ck = s["chunking"]["by_type"]
    check("chunking.contract 存在", "contract" in ck, True)
    check("chunking.contract.docs", ck["contract"]["docs"], 1)
    check("chunking.contract.chunks", ck["contract"]["chunks"], 3)
    check("chunking.contract.avg_chunk_chars", ck["contract"]["avg_chunk_chars"], 200.0)
    check("chunking.contract.avg_chunks_per_doc", ck["contract"]["avg_chunks_per_doc"], 3.0)
    # 修正 bucket 后分位数才有意义 (原默认 bucket 下恒等于 10 字符)
    check("chunking.contract.p95_chunk_chars", ck["contract"]["p95_chunk_chars"], 285.0)

    assert not failures, f"{len(failures)} 项断言失败: {failures}"


def test_degradation_stage_vs_level_complement():
    """stage 维度与 level 维度互补: 多阶段同发时 level 只记最高, stage 各记一次。"""
    _metrics.track_degradation.labels(level="3").inc()
    _metrics.track_degradation_stage.labels(stage="retrieval").inc()
    _metrics.track_degradation_stage.labels(stage="rerank").inc()
    _metrics.track_degradation_stage.labels(stage="llm").inc()
    s = _metrics.metrics_summary()
    assert s["degradation"]["levels"]["3"] == 1
    assert s["degradation"]["levels"]["1"] == 0  # 中间阶段不计入 level
    assert s["degradation"]["stages"]["retrieval"] == 1
    assert s["degradation"]["stages"]["rerank"] == 1
    assert s["degradation"]["stages"]["llm"] == 1


def test_deg_bump_accumulates_multiple_stages():
    """orchestrator 辅助: 一次查询命中多阶段时, _deg_bump 累加全部阶段 (不丢中间阶段)。"""
    from api.core.orchestrator import _deg_bump, _deg_reset, _deg_stages

    _deg_reset()
    d = _deg_bump(0, 2, "retrieval")
    d = _deg_bump(d, 1, "rerank")
    d = _deg_bump(d, 3, "llm")
    assert d == 3  # 等级取最高
    assert _deg_stages.get() == {"retrieval", "rerank", "llm"}
    _deg_reset()  # 清理 contextvar, 避免泄漏到后续测试
