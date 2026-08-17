"""评估/标定工具的纯逻辑契约锁 — 防止度量数学静默回归。

这些函数是观察层 (evaluate.py / calibrate_threshold.py / calibrate_fidelity.py) 的核心,
一旦被"顺手简化"会导致后续所有评估数字悄悄失真, 故单测固化其行为。
"""

import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)


def test_calibrate_f1_basic():
    from calibrate_threshold import _f1

    assert _f1(0, 0, 0) == 0.0  # 无样本 → 0
    assert _f1(1, 0, 0) == 1.0  # 全 TP → 1
    assert _f1(1, 1, 0) == pytest.approx(2 / 3, abs=1e-9)  # P=0.5 R=1
    assert _f1(1, 0, 1) == pytest.approx(2 / 3, abs=1e-9)  # P=1 R=0.5


def test_calibrate_picks_f1_max():
    from calibrate_threshold import _f1

    # t=0.5 时 TP=2 FP=0 FN=0 → F1=1, 其余更低
    cases = [
        {"top1": 0.2, "should": True},
        {"top1": 0.3, "should": True},
        {"top1": 0.9, "should": False},
    ]
    best_t, best_f1 = None, None
    for t in [round(0.3 + 0.05 * i, 2) for i in range(11)]:
        tp = fp = fn = 0
        for c in cases:
            fb = c["top1"] < t
            if c["should"] and fb:
                tp += 1
            elif (not c["should"]) and fb:
                fp += 1
            elif c["should"] and (not fb):
                fn += 1
        f1 = _f1(tp, fp, fn)
        if best_f1 is None or f1 > best_f1:
            best_t, best_f1 = t, f1
    assert best_t == pytest.approx(0.35, abs=1e-9) and best_f1 == 1.0


def test_evaluate_cosine():
    from evaluate import _cosine

    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    assert _cosine(a, b) == pytest.approx(1.0, abs=1e-9)
    assert _cosine(a, c) == pytest.approx(0.0, abs=1e-9)
    assert _cosine([], []) == 0.0


def test_evaluate_ragas_json_extract():
    from evaluate import _extract_ragas_json

    text = '闲聊 {"faithfulness":0.9,"context_precision":0.8,"context_recall":0.7,"answer_relevancy":0.85} 完毕'
    d = _extract_ragas_json(text)
    assert d == {
        "faithfulness": 0.9,
        "context_precision": 0.8,
        "context_recall": 0.7,
        "answer_relevancy": 0.85,
    }
    # 缺键 → None
    assert _extract_ragas_json('{"faithfulness":0.5}') is None
    # 无 JSON → None
    assert _extract_ragas_json("no json here") is None


def test_calibrate_fidelity_sweep_perfect_separation():
    from calibrate_fidelity import sweep_fidelity

    # 两个 bad (score 0.1,0.2), t>=0.3 时全部 reject 且全 bad → F1=1
    cases = [{"score": 0.1, "is_bad": True}, {"score": 0.2, "is_bad": True}]
    res = sweep_fidelity(cases)
    assert res["best_f1"] == 1.0
    # 两 bad score 0.1/0.2, 最小能全拒的 t=0.22 (0.2<0.22 成立, 0.2<0.20 不成立)
    assert res["best_threshold"] == 0.22


def test_calibrate_fidelity_sweep_mixed():
    from calibrate_fidelity import sweep_fidelity

    # 两个 bad (0.1,0.2) + 两个 good (0.9,0.95)
    # 最佳 t 在 (0.2,0.9]: t=0.22 → reject 两个 bad, good 不拒 → TP=2 FP=0 FN=0 F1=1
    mixed = [
        {"score": 0.1, "is_bad": True},
        {"score": 0.2, "is_bad": True},
        {"score": 0.9, "is_bad": False},
        {"score": 0.95, "is_bad": False},
    ]
    r2 = sweep_fidelity(mixed)
    assert r2["best_f1"] == 1.0
    # 最小安全边界: 刚能拒两 bad(>0.2) 且不拒 good(<=0.9) → 首达 0.22
    assert r2["best_threshold"] == 0.22
