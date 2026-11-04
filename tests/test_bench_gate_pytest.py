"""性能回归门禁的自测（P1 / Round 3）。

CI 里没有起服务的成本，所以把"阈值判定"这层纯逻辑单测掉：
分位数口径、延迟 vs 吞吐的变差方向、以及各种不可比场景必须**失败而不是放过**。
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_common as bc  # noqa: E402
import check_bench_regression as gate  # noqa: E402


def _payload(**over) -> dict:
    base = {
        "tool": "load_test",
        "ts": "2026-11-03T00:00:00Z",
        "commit": "abc1234",
        "concurrency": 20,
        "requests": 200,
        "ok": 200,
        "failed": 0,
        "failure_rate": 0.0,
        "qps": 10.0,
        "latency_ms_p50": 100.0,
        "latency_ms_p95": 200.0,
        "latency_ms_p99": 250.0,
        "latency_ms_max": 300.0,
    }
    base.update(over)
    return base


# ───────────────────────── 分位数口径 ─────────────────────────

def test_nearest_rank_single_sample() -> None:
    assert bc.nearest_rank([42.0], 0.95) == 42.0


def test_nearest_rank_uses_ceiling() -> None:
    values = [float(i) for i in range(1, 11)]  # 1..10
    assert bc.nearest_rank(values, 0.50) == 5.0
    assert bc.nearest_rank(values, 0.95) == 10.0  # ceil(9.5)-1 = 9 -> 10.0


def test_summarize_is_monotonic() -> None:
    s = bc.summarize([float(i) for i in range(100)], "latency_ms")
    assert s["latency_ms_n"] == 100
    assert s["latency_ms_p50"] <= s["latency_ms_p95"] <= s["latency_ms_p99"] <= s["latency_ms_max"]


def test_summarize_empty_is_nan() -> None:
    s = bc.summarize([], "ttft_ms")
    assert s["ttft_ms_n"] == 0
    assert s["ttft_ms_p95"] != s["ttft_ms_p95"]  # nan


# ───────────────────────── 变差方向 ─────────────────────────

def test_latency_regression_direction() -> None:
    ok, change = gate.compare(200.0, 218.0, "latency_ms.p95", 0.10)
    assert ok and abs(change - 0.09) < 1e-9

    ok, change = gate.compare(200.0, 222.0, "latency_ms.p95", 0.10)
    assert not ok and abs(change - 0.11) < 1e-9


def test_throughput_regression_direction() -> None:
    ok, _ = gate.compare(10.0, 9.1, "qps", 0.10)  # -9% 吞吐，仍通过
    assert ok

    ok, _ = gate.compare(10.0, 8.8, "qps", 0.10)  # -12% 吞吐，判失败
    assert not ok


# ───────────────────────── 门禁端到端 ─────────────────────────

def _write(tmp_path: pathlib.Path, name: str, payload: dict) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def _run_gate(monkeypatch, tmp_path, baseline: dict, current: dict,
              metric: str = "latency_ms.p95") -> int:
    monkeypatch.setattr(sys, "argv", [
        "check_bench_regression.py",
        "--baseline", _write(tmp_path, "b.json", baseline),
        "--current", _write(tmp_path, "c.json", current),
        "--metric", metric,
    ])
    return gate.main()


def test_gate_passes_within_threshold(monkeypatch, tmp_path) -> None:
    assert _run_gate(monkeypatch, tmp_path, _payload(), _payload(latency_ms_p95=215.0)) == 0


def test_gate_fails_beyond_threshold(monkeypatch, tmp_path) -> None:
    assert _run_gate(monkeypatch, tmp_path, _payload(), _payload(latency_ms_p95=245.0)) == 1


def test_gate_fails_when_failure_rate_jumps(monkeypatch, tmp_path) -> None:
    assert _run_gate(monkeypatch, tmp_path, _payload(),
                     _payload(latency_ms_p95=205.0, failure_rate=0.09)) == 1


def test_gate_fails_on_concurrency_mismatch(monkeypatch, tmp_path) -> None:
    assert _run_gate(monkeypatch, tmp_path, _payload(), _payload(concurrency=5)) == 1


def test_gate_fails_when_current_has_no_success(monkeypatch, tmp_path) -> None:
    assert _run_gate(monkeypatch, tmp_path, _payload(), _payload(ok=0, latency_ms_p95=0.0)) == 1


def test_gate_fails_when_metric_missing(monkeypatch, tmp_path) -> None:
    cur = _payload()
    cur.pop("latency_ms_p95")
    assert _run_gate(monkeypatch, tmp_path, _payload(), cur) == 1