"""召回回归门禁 (scripts/eval_gate.py) 单元测试 — 门槛逻辑必须本身可靠。"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_gate import main

_BASE_OFF = {
    "recall@1": 0.347,
    "recall@3": 0.740,
    "recall@5": 0.891,
    "recall@10": 0.979,
    "hit@1": 0.536,
    "hit@3": 0.949,
    "hit@5": 0.977,
    "hit@10": 0.992,
    "mrr": 0.743,
}


def _write(path, off):
    path.write_text(
        json.dumps({"sample_count": 390, "rerank_off": off, "rerank_on": {}}),
        encoding="utf-8",
    )
    return str(path)


def _run(monkeypatch, baseline, current):
    """main() 直接返回退出码(sys.exit 只在 __main__ 守卫里) —— 便于测试直接断言。"""
    monkeypatch.setattr(sys, "argv", ["eval_gate", "--baseline", baseline, "--current", current])
    return main()


def test_identical_summaries_pass(monkeypatch, tmp_path):
    p = _write(tmp_path / "a.json", dict(_BASE_OFF))
    assert _run(monkeypatch, p, p) == 0


def test_regression_beyond_tolerance_blocks(monkeypatch, tmp_path):
    base = _write(tmp_path / "base.json", dict(_BASE_OFF))
    off = dict(_BASE_OFF, **{"recall@1": _BASE_OFF["recall@1"] - 0.05})  # -5pp
    cur = _write(tmp_path / "cur.json", off)
    assert _run(monkeypatch, base, cur) == 1


def test_mrr_regression_beyond_tolerance_blocks(monkeypatch, tmp_path):
    base = _write(tmp_path / "base.json", dict(_BASE_OFF))
    cur = _write(tmp_path / "cur.json", dict(_BASE_OFF, mrr=_BASE_OFF["mrr"] - 0.03))
    assert _run(monkeypatch, base, cur) == 1


def test_regression_within_tolerance_passes(monkeypatch, tmp_path):
    """容差内的小幅波动不阻断 —— 门禁要挡真实回退, 不做噪声放大器。"""
    base = _write(tmp_path / "base.json", dict(_BASE_OFF))
    cur = _write(tmp_path / "cur.json", dict(_BASE_OFF, **{"recall@1": _BASE_OFF["recall@1"] - 0.005}))  # -0.5pp
    assert _run(monkeypatch, base, cur) == 0


def test_improvement_passes(monkeypatch, tmp_path):
    base = _write(tmp_path / "base.json", dict(_BASE_OFF))
    cur = _write(tmp_path / "cur.json", dict(_BASE_OFF, **{"recall@1": _BASE_OFF["recall@1"] + 0.02}))
    assert _run(monkeypatch, base, cur) == 0


def test_missing_metric_in_current_blocks(monkeypatch, tmp_path):
    """当前结果缺指标 = 评测没跑出数, 必须阻断而非静默放行。"""
    base = _write(tmp_path / "base.json", dict(_BASE_OFF))
    off = dict(_BASE_OFF)
    off.pop("recall@1")
    cur = _write(tmp_path / "cur.json", off)
    assert _run(monkeypatch, base, cur) == 1


def test_unreadable_summary_returns_usage_error(monkeypatch, tmp_path):
    base = _write(tmp_path / "base.json", dict(_BASE_OFF))
    assert _run(monkeypatch, base, str(tmp_path / "nope.json")) == 2
