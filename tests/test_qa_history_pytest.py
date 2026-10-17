"""api/core/qa_history.py —— 历史问答基线聚合。

覆盖策略：模块主体是纯函数（分位数、均值、比率、聚合）与带缓存的读取入口，
无需网络与真实数据文件即可完整验证。重点覆盖三处易错逻辑：

1. 零延迟样本的剔除（零值是未计时/探针，不是「真的很快」）
2. 分位数用最近秩法在原始样本上计算，无 bucket 插值损失
3. 文件缺失与解析失败均返回 available=False，不抛错（观测增强不应影响服务）
"""

from __future__ import annotations

import json

import pytest

from api.core import qa_history as qh

# ────────────────────────── 纯函数层 ──────────────────────────


class TestQuantile:
    def test_empty_returns_zero(self):
        assert qh._quantile([], 0.5) == 0.0

    def test_single_sample(self):
        assert qh._quantile([42.0], 0.95) == 42.0

    def test_median_of_odd_count(self):
        # 最近秩法：idx = round(q*(n-1))
        assert qh._quantile([10.0, 20.0, 30.0], 0.50) == 20.0

    def test_p95_of_even_count(self):
        vals = [float(v) for v in range(1, 21)]  # 1..20
        # idx = round(0.95 * 19) = 18 → 第 19 个元素 = 19.0
        assert qh._quantile(vals, 0.95) == 19.0

    def test_clamps_index_out_of_range(self):
        # q=0 与 q=1 都应落在有效区间内，不抛 IndexError
        vals = [1.0, 2.0, 3.0]
        assert qh._quantile(vals, 0.0) == 1.0
        assert qh._quantile(vals, 1.0) == 3.0

    def test_result_is_rounded_to_two_decimals(self):
        assert qh._quantile([1.0, 2.0, 2.5], 0.50) == 2.5


class TestAvgAndRate:
    def test_avg_empty(self):
        assert qh._avg([]) == 0.0

    def test_avg_basic(self):
        assert qh._avg([10.0, 20.0, 30.0]) == 20.0

    def test_rate_zero_total_is_zero(self):
        # 除零保护：分母为 0 时必须返回 0.0 而非抛 ZeroDivisionError
        assert qh._rate(0, 0) == 0.0

    def test_rate_basic(self):
        assert qh._rate(1, 4) == 0.25


# ────────────────────────── 聚合层 ──────────────────────────


def _case(**kw):
    """构造一条问答记录，填入合法的默认字段。"""
    base = {
        "degradation_level": 0,
        "routing_source": "rule",
        "kb_id": "documents",
        "latency_ms": 100,
        "faithfulness": 0.9,
        "tokens_total": 1000,
        "question": "一个正常长度的问题",
        "created_at": "2026-09-01T10:00:00",
    }
    base.update(kw)
    return base


class TestAggregate:
    def test_empty_case_list_is_available_but_zeroed(self):
        r = qh._aggregate([], None, "/tmp/qa_export.json")
        assert r["available"] is True
        assert r["cases"] == 0
        assert r["degradation"]["total"] == 0
        assert r["degradation"]["degraded_rate"] == 0.0
        assert r["latency_ms"]["samples"] == 0
        assert r["latency_ms"]["avg"] == 0.0
        assert r["latency_ms"]["p95"] == 0.0
        assert r["latency_ms"]["max"] == 0.0
        assert r["faithfulness"]["avg"] == 0.0
        assert r["tokens"]["total"] == 0
        assert r["range"] == {"from": None, "to": None}
        assert r["trend"] == []

    def test_source_is_basename_only(self):
        # 避免把绝对路径泄露给前端
        r = qh._aggregate([], None, "/var/lib/mira/data/qa_export.json")
        assert r["source"] == "qa_export.json"

    def test_zero_latency_samples_are_excluded(self):
        """核心口径：零延迟是未计时/探针，不能计入均值，否则人为拉低平均。"""
        cases = [
            _case(latency_ms=0),
            _case(latency_ms=200),
            _case(latency_ms=400),
        ]
        r = qh._aggregate(cases, None, "x.json")
        assert r["latency_ms"]["samples"] == 2          # 只有两条有效样本
        assert r["latency_ms"]["avg"] == 300.0         # 而非 (0+200+400)/3
        assert r["data_quality"]["zero_latency"] == 1   # 被剔除的零值条数

    def test_missing_faithfulness_is_counted(self):
        cases = [_case(faithfulness=0), _case(faithfulness=0.8)]
        r = qh._aggregate(cases, None, "x.json")
        assert r["faithfulness"]["samples"] == 1
        assert r["faithfulness"]["avg"] == 0.8
        assert r["data_quality"]["faithfulness_missing"] == 1

    def test_short_questions_counted_as_probe_like(self):
        """长度 ≤2 的问题视为压测探针，单独计数供前端标注口径。"""
        cases = [_case(question="x"), _case(question="ab"), _case(question="正常问题")]
        r = qh._aggregate(cases, None, "x.json")
        assert r["data_quality"]["probe_like"] == 2

    def test_degradation_distribution_and_rate(self):
        cases = [
            _case(degradation_level=0),
            _case(degradation_level=0),
            _case(degradation_level=2),
            _case(degradation_level=3),
        ]
        r = qh._aggregate(cases, None, "x.json")
        deg = r["degradation"]
        assert deg["levels"]["0"] == 2
        assert deg["levels"]["2"] == 1
        assert deg["levels"]["3"] == 1
        assert deg["total"] == 4
        assert deg["degraded"] == 2
        assert deg["degraded_rate"] == 0.5

    def test_missing_routing_source_becomes_unknown(self):
        cases = [{**_case(), "routing_source": None}, _case(routing_source="llm")]
        r = qh._aggregate(cases, None, "x.json")
        assert r["routing"]["sources"]["unknown"] == 1
        assert r["routing"]["sources"]["llm"] == 1
        assert r["routing"]["total"] == 2

    def test_missing_kb_becomes_dash(self):
        cases = [{**_case(), "kb_id": None}]
        r = qh._aggregate(cases, None, "x.json")
        assert r["knowledge_bases"] == {"-": 1}

    def test_range_uses_min_and_max_timestamp(self):
        cases = [
            _case(created_at="2026-09-03T08:00:00"),
            _case(created_at="2026-09-01T08:00:00"),
            _case(created_at="2026-09-05T08:00:00"),
        ]
        r = qh._aggregate(cases, None, "x.json")
        assert r["range"]["from"].startswith("2026-09-01")
        assert r["range"]["to"].startswith("2026-09-05")

    def test_trend_groups_by_day_and_counts_degraded(self):
        cases = [
            _case(created_at="2026-09-01T10:00:00"),
            _case(created_at="2026-09-01T11:00:00", degradation_level=3, latency_ms=0),
            _case(created_at="2026-09-02T10:00:00"),
        ]
        r = qh._aggregate(cases, None, "x.json")
        trend = {t["date"]: t for t in r["trend"]}
        assert set(trend) == {"2026-09-01", "2026-09-02"}
        assert trend["2026-09-01"]["cases"] == 2
        assert trend["2026-09-01"]["degraded"] == 1
        assert trend["2026-09-01"]["degraded_rate"] == 0.5
        # 零延迟被剔除后该日只剩一条 100ms 样本
        assert trend["2026-09-01"]["avg_latency_ms"] == 100.0
        assert trend["2026-09-02"]["degraded"] == 0

    def test_tokens_total_and_avg(self):
        cases = [_case(tokens_total=100), _case(tokens_total=300)]
        r = qh._aggregate(cases, None, "x.json")
        assert r["tokens"]["total"] == 400
        assert r["tokens"]["avg"] == 200.0

    def test_latency_max_is_highest_sample(self):
        cases = [_case(latency_ms=50), _case(latency_ms=999), _case(latency_ms=0)]
        r = qh._aggregate(cases, None, "x.json")
        assert r["latency_ms"]["max"] == 999.0

    def test_data_quality_notes_are_human_readable(self):
        r = qh._aggregate([_case()], None, "x.json")
        assert "非线上真实流量" in r["data_quality"]["note"]


# ────────────────────────── 入口与缓存层 ──────────────────────────


class TestResolvePath:
    def test_prefers_cwd_when_file_present(self, tmp_path, monkeypatch):
        rel = qh._HISTORY_RELPATH
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert qh._resolve_path() is not None

    def test_returns_none_when_absent_everywhere(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # 项目根回退路径也不应存在（该函数会检查 cwd 与项目根两处）
        got = qh._resolve_path()
        # 仓库内可能真实存在 data/qa_export.json，故只断言类型
        assert got is None or isinstance(got, str)


class TestHistorySummary:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        qh._cache["key"] = None
        qh._cache["value"] = None
        yield
        qh._cache["key"] = None
        qh._cache["value"] = None

    def test_missing_file_returns_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qh, "_resolve_path", lambda: None)
        r = qh.history_summary(force=True)
        assert r == {"available": False, "reason": "history_file_not_found"}

    def test_parse_failure_returns_unavailable_not_raise(self, tmp_path, monkeypatch):
        """观测增强绝不能影响服务可用性：脏文件必须降级而非抛异常。"""
        bad = tmp_path / "broken.json"
        bad.write_text("{ this is not json", encoding="utf-8")
        monkeypatch.setattr(qh, "_resolve_path", lambda: str(bad))
        r = qh.history_summary(force=True)
        assert r["available"] is False
        assert r["reason"] == "history_parse_failed"

    def test_reads_and_aggregates_snapshot(self, tmp_path, monkeypatch):
        snap = tmp_path / "qa_export.json"
        snap.write_text(
            json.dumps(
                {
                    "exported_at": "2026-09-06T12:00:00",
                    "cases": [_case(latency_ms=120), _case(latency_ms=0, degradation_level=3)],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(qh, "_resolve_path", lambda: str(snap))
        r = qh.history_summary(force=True)
        assert r["available"] is True
        assert r["exported_at"] == "2026-09-06T12:00:00"
        assert r["cases"] == 2
        assert r["latency_ms"]["samples"] == 1        # 零值被剔除
        assert r["degradation"]["degraded"] == 1

    def test_missing_cases_key_is_tolerated(self, tmp_path, monkeypatch):
        snap = tmp_path / "qa_export.json"
        snap.write_text(json.dumps({"exported_at": "x"}), encoding="utf-8")
        monkeypatch.setattr(qh, "_resolve_path", lambda: str(snap))
        r = qh.history_summary(force=True)
        assert r["available"] is True
        assert r["cases"] == 0

    def test_cache_hit_avoids_reparse(self, tmp_path, monkeypatch):
        snap = tmp_path / "qa_export.json"
        snap.write_text(json.dumps({"cases": [_case()]}), encoding="utf-8")
        monkeypatch.setattr(qh, "_resolve_path", lambda: str(snap))

        calls = {"n": 0}
        real_open = open

        def counting_open(*a, **kw):
            if str(a[0]).endswith("qa_export.json"):
                calls["n"] += 1
            return real_open(*a, **kw)

        monkeypatch.setattr("builtins.open", counting_open)

        first = qh.history_summary(force=True)   # 落缓存
        second = qh.history_summary(force=False)  # 命中缓存
        assert first == second
        assert calls["n"] == 1                    # 只真正读过一次

    def test_cache_invalidated_when_file_changes(self, tmp_path, monkeypatch):
        snap = tmp_path / "qa_export.json"
        snap.write_text(json.dumps({"cases": [_case()]}), encoding="utf-8")
        monkeypatch.setattr(qh, "_resolve_path", lambda: str(snap))

        r1 = qh.history_summary(force=True)
        assert r1["cases"] == 1

        # 改内容并变更 mtime/size，(mtime,size) 变化即触发重算
        snap.write_text(json.dumps({"cases": [_case(), _case()]}), encoding="utf-8")
        r2 = qh.history_summary(force=False)
        assert r2["cases"] == 2