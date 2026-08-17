"""build_labeled_skeleton 单测: score 映射 + 结构正确性 + 审核清单生成。"""

import json
import os

from scripts.build_labeled_skeleton import _case_score, _render_review, build_labeled


def _write_summary(tmp_path, cases):
    p = tmp_path / "eval-summary.json"
    p.write_text(json.dumps({"summary": {}, "cases": cases}, ensure_ascii=False), encoding="utf-8")
    return str(p)


def test_case_score_prefers_ragas():
    c = {"ragas": {"faithfulness": 0.12}, "hallucination_rate": 0.3}
    assert _case_score(c) == 0.12


def test_case_score_fallback_to_hallucination():
    # 无 ragas -> 1 - hallucination_rate
    c = {"hallucination_rate": 0.7}
    assert _case_score(c) == 0.3


def test_case_score_ragas_bad_value_falls_back():
    # ragas.faithfulness 非数值 -> 回退
    c = {"ragas": {"faithfulness": "n/a"}, "hallucination_rate": 0.4}
    assert _case_score(c) == 0.6


def test_build_labeled_maps_fields_and_null_is_bad():
    cases = [
        {
            "question": "Q1",
            "kb": "kb1",
            "answer": "A1",
            "contexts": ["ctx1", "ctx2"],
            "ragas": {"faithfulness": 0.55},
            "hallucination_rate": 0.2,
        },
        {
            "question": "Q2",
            "kb": "kb2",
            "answer": "A2",
            "contexts": [],
            "hallucination_rate": 0.8,
        },
    ]
    labeled = build_labeled(cases)
    assert len(labeled) == 2
    assert labeled[0]["score"] == 0.55
    assert labeled[0]["answer"] == "A1"
    assert labeled[0]["contexts"] == ["ctx1", "ctx2"]
    assert labeled[0]["is_bad"] is None
    # 第二条回退: 1 - 0.8 = 0.2
    assert labeled[1]["score"] == 0.2
    assert labeled[1]["is_bad"] is None


def test_render_review_contains_all_questions(tmp_path):
    cases = [
        {"question": "Q1", "kb": "kb1", "answer": "A1", "contexts": ["c1"], "ragas": {"faithfulness": 0.5}},
        {"question": "Q2", "kb": "kb2", "answer": "A2", "contexts": [], "hallucination_rate": 0.1},
    ]
    labeled = build_labeled(cases)
    md = _render_review(labeled, "data/eval-summary.json")
    assert "Q1" in md and "Q2" in md
    assert "A1" in md and "c1" in md
    assert "is_bad=false" in md and "is_bad=true" in md
    assert "calibrate_fidelity.py" in md


def test_end_to_end_cli(tmp_path):
    cases = [{"question": "Q", "kb": "k", "answer": "A", "contexts": ["x"], "ragas": {"faithfulness": 0.4}}]
    summary = _write_summary(tmp_path, cases)
    out = tmp_path / "labeled.json"
    review = tmp_path / "labeled_review.md"

    import scripts.build_labeled_skeleton as mod

    # 直接调用内部函数验证 CLI 核心逻辑 (与 main 同一实现)
    labeled = mod.build_labeled(cases)
    out.write_text(json.dumps({"cases": labeled}, ensure_ascii=False), encoding="utf-8")
    review.write_text(mod._render_review(labeled, summary), encoding="utf-8")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cases"][0]["score"] == 0.4
    assert os.path.getsize(review) > 0
