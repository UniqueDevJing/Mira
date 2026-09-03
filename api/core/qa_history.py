"""历史问答基线聚合 — 跨进程/跨重启的项目级历史指标。

与 Prometheus 指标 (api.core.metrics) 互补:
  · metrics     → 本次进程运行期间的实时累计 (**重启清零**)
  · qa_history  → 项目历史累积基线 (导出快照, 不随重启丢失)
前端指标/降级面板因此能在服务冷启动时也展示真实数据, 而不是一片零。

数据源: data/qa_export.json (导出快照, 约 5.6MB / 2838 条)。
文件较大, 故按 (mtime, size) 做进程内缓存, 避免每次请求重复解析。

⚠️ 数据口径 (前端展示时须如实标注, 不可与实时指标混为一谈):
  · 该快照含压测/评测期间的探针请求 (如 question="x") 与 LLM 不可用期间的
    兜底记录 (degradation_level=3 占比高), 不代表线上真实流量构成。
  · 延迟分位数只基于 latency_ms>0 的有效样本; 零值样本不计入均值,
    否则会人为拉低平均值 (零延迟是未计时/探针, 不是"真的很快")。
  · 分位数用最近秩法在原始样本上精确计算, 与 metrics 侧的 histogram
    bucket 插值不同 — 这里没有分桶损失。
"""

import json
import logging
import os
from collections import Counter, defaultdict

from api.core.metrics import DEGRADATION_LABELS

logger = logging.getLogger(__name__)

_HISTORY_RELPATH = os.path.join("data", "qa_export.json")

# 进程内缓存: key = (mtime, size), 文件变化自动失效
_cache: dict = {"key": None, "value": None}


def _resolve_path() -> str | None:
    """定位历史数据文件: 优先 cwd, 回退到项目根 (兼容不同启动目录)。"""
    if os.path.exists(_HISTORY_RELPATH):
        return os.path.abspath(_HISTORY_RELPATH)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(root, _HISTORY_RELPATH)
    return p if os.path.exists(p) else None


def _quantile(sorted_vals: list[float], q: float) -> float:
    """精确分位数 (最近秩法) — 原始样本可用, 无需 bucket 插值。"""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = min(n - 1, max(0, round(q * (n - 1))))
    return round(float(sorted_vals[idx]), 2)


def _avg(vals: list[float]) -> float:
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _rate(part: int, total: int) -> float:
    return round(part / total, 4) if total else 0.0


def _aggregate(cases: list[dict], exported_at, source: str) -> dict:
    """把原始问答记录聚合成前端可直接消费的历史基线。"""
    deg = Counter()
    route = Counter()
    kb = Counter()
    latencies: list[float] = []
    faiths: list[float] = []
    tokens: list[int] = []
    created: list[str] = []
    probe_like = 0
    by_day: dict[str, dict] = defaultdict(lambda: {"cases": 0, "degraded": 0, "lat": []})

    for c in cases:
        lvl = str(c.get("degradation_level", 0) or 0)
        deg[lvl] += 1
        route[str(c.get("routing_source") or "unknown")] += 1
        kb[str(c.get("kb_id") or "-")] += 1

        lat = c.get("latency_ms") or 0
        if lat > 0:
            latencies.append(float(lat))
        faith = c.get("faithfulness") or 0
        if faith > 0:
            faiths.append(float(faith))
        tok = c.get("tokens_total") or 0
        if tok:
            tokens.append(int(tok))
        if len(str(c.get("question") or "").strip()) <= 2:
            probe_like += 1

        ts = str(c.get("created_at") or "")
        if ts:
            created.append(ts)
            day = by_day[ts[:10]]
            day["cases"] += 1
            if lvl != "0":
                day["degraded"] += 1
            if lat > 0:
                day["lat"].append(float(lat))

    latencies.sort()
    faiths.sort()
    deg_levels = {lvl: deg.get(lvl, 0) for lvl in DEGRADATION_LABELS}
    deg_total = sum(deg_levels.values())
    deg_degraded = deg_total - deg_levels.get("0", 0)

    trend = [
        {
            "date": day,
            "cases": d["cases"],
            "degraded": d["degraded"],
            "degraded_rate": _rate(d["degraded"], d["cases"]),
            "avg_latency_ms": _avg(d["lat"]),
        }
        for day, d in sorted(by_day.items())
    ]

    return {
        "available": True,
        "source": os.path.basename(source),
        "exported_at": exported_at,
        "cases": len(cases),
        "range": {"from": min(created) if created else None, "to": max(created) if created else None},
        "degradation": {
            "levels": deg_levels,
            "level_labels": DEGRADATION_LABELS,
            "total": deg_total,
            "degraded": deg_degraded,
            "degraded_rate": _rate(deg_degraded, deg_total),
        },
        "routing": {"sources": dict(route.most_common()), "total": sum(route.values())},
        "knowledge_bases": dict(kb.most_common()),
        "latency_ms": {
            "samples": len(latencies),
            "avg": _avg(latencies),
            "p50": _quantile(latencies, 0.50),
            "p95": _quantile(latencies, 0.95),
            "p99": _quantile(latencies, 0.99),
            "max": round(latencies[-1], 2) if latencies else 0.0,
        },
        "faithfulness": {
            "samples": len(faiths),
            "avg": round(sum(faiths) / len(faiths), 4) if faiths else 0.0,
            "p50": _quantile(faiths, 0.50),
        },
        "tokens": {"total": sum(tokens), "avg": round(sum(tokens) / len(tokens), 1) if tokens else 0.0},
        "trend": trend,
        "data_quality": {
            "cases": len(cases),
            "zero_latency": len(cases) - len(latencies),
            "probe_like": probe_like,
            "faithfulness_missing": len(cases) - len(faiths),
            "note": "导出快照含压测探针与 LLM 不可用期间兜底记录, 非线上真实流量构成; 延迟统计已剔除零值样本",
        },
    }


def history_summary(force: bool = False) -> dict:
    """历史问答基线汇总 (带 mtime/size 缓存)。

    文件缺失或解析失败时返回 available=False, 不抛错 — 历史基线是观测增强,
    绝不能影响服务可用性。force=True 用于测试绕过缓存。
    """
    path = _resolve_path()
    if not path:
        return {"available": False, "reason": "history_file_not_found"}

    try:
        stat = os.stat(path)
        key = (stat.st_mtime, stat.st_size)
        if not force and _cache["key"] == key and _cache["value"] is not None:
            return _cache["value"]

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cases = data.get("cases") or []
        value = _aggregate(cases, data.get("exported_at"), path)
        _cache["key"] = key
        _cache["value"] = value
        return value
    except Exception as e:  # noqa: BLE001 — 历史基线失败不影响服务
        logger.warning("历史基线聚合失败: %s", str(e)[:120])
        return {"available": False, "reason": "history_parse_failed"}
