"""M1 错误账本单元测试: 五类分类规则 + 入账/聚合。

命名遵循仓库 pytest 收集规则 (test_*_pytest.py)。
"""
import pytest

from api.core.error_ledger import (
    CLASS_LABELS,
    CONTEXT_TRUNCATED,
    FALSE_REFUSAL,
    LOW_RANK,
    NO_RECALL,
    NUMBER_REWRITTEN,
    classify,
    record_from_rag,
    summary,
    trend,
)


def _docs(scores, truncated=False, content="FastAPI 是一个高性能的 Python Web 框架"):
    return [
        {"doc_id": "d1", "content": content, "context_full": content, "score": s, "truncated": truncated}
        for s in scores
    ]


def test_no_recall_empty_docs():
    cls, _, _ = classify("如何退款", [], 0.0, "未在知识库找到", None, None, False)
    assert cls == NO_RECALL


def test_no_recall_no_docs_reason():
    cls, _, _ = classify("如何退款", _docs([0.9]), 0.9, "未在知识库找到", "no_docs", None, False)
    assert cls == NO_RECALL


def test_false_refusal_high_top1():
    cls, _, _ = classify("FastAPI 怎么用", _docs([0.85]), 0.85, "拒答", "low_relevance", None, False)
    assert cls == FALSE_REFUSAL


def test_false_refusal_overlap():
    # 零词重合会被 low_relevance 拒, 但问题含 FastAPI 与文档强相关 → 误拒
    cls, _, _ = classify("FastAPI 是什么", _docs([0.4]), 0.4, "拒答", "low_relevance", None, False)
    assert cls == FALSE_REFUSAL


def test_false_refusal_low_fidelity_high_overlap():
    cls, _, _ = classify("FastAPI 是什么框架", _docs([0.8]), 0.8, "拒答", "low_fidelity", None, False)
    assert cls == FALSE_REFUSAL


def test_number_rewritten_finance_fail():
    cls, _, _ = classify("余额多少", _docs([0.9]), 0.9, "余额 100 元", None, "fail", False)
    assert cls == NUMBER_REWRITTEN


def test_number_rewritten_mismatch():
    # 答案含上下文中没有的数字 → 数字型幻觉
    cls, _, _ = classify("价格多少", _docs([0.9], content="价格 50 元"), 0.9, "价格是 99 元", None, None, False)
    assert cls == NUMBER_REWRITTEN


def test_low_rank():
    cls, _, _ = classify("如何部署", _docs([0.3]), 0.3, "请按文档部署", None, None, False)
    assert cls == LOW_RANK


def test_context_truncated():
    cls, _, _ = classify("如何部署", _docs([0.9], truncated=True), 0.9, "请按文档部署", None, None, True)
    assert cls == CONTEXT_TRUNCATED


def test_not_badcase():
    cls, _, _ = classify("如何部署", _docs([0.95]), 0.95, "请按文档部署步骤执行", None, None, False)
    assert cls is None


def test_priority_no_recall_over_false_refusal():
    # 空 docs 且带 refusal_reason, 应判 no_recall 而非 false_refusal
    cls, _, _ = classify("x", [], 0.0, "拒答", "low_relevance", None, False)
    assert cls == NO_RECALL


def test_record_and_summary(tmp_path, monkeypatch):
    from api.config import settings

    monkeypatch.setattr(settings, "m1_ledger_path", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(settings, "m1_enabled", True)

    # 写入一条 low_rank badcase
    record_from_rag(
        "如何部署", "kb1", docs=_docs([0.2]), top1_score=0.2,
        answer="a", refusal_reason=None, finance_verdict=None, truncated=False,
    )
    s = summary()
    assert s["total"] == 1
    assert s["by_class"][LOW_RANK] == 1
    assert s["labels"] == CLASS_LABELS

    t = trend()
    assert any(LOW_RANK in v for v in t.values())


def test_disabled_does_not_record(tmp_path, monkeypatch):
    from api.config import settings

    monkeypatch.setattr(settings, "m1_enabled", False)
    monkeypatch.setattr(settings, "m1_ledger_path", str(tmp_path / "ledger.jsonl"))

    before = summary()["total"]
    rec = record_from_rag(
        "如何部署", "kb1", docs=_docs([0.2]), top1_score=0.2,
        answer="a", refusal_reason=None, finance_verdict=None, truncated=False,
    )
    assert rec is None
    # 禁用时不得向账本新增任何记录(进程内单例共享, 只验证「不增量」)
    assert summary()["total"] == before
