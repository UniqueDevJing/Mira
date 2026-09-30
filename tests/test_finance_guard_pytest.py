"""金融数值校验护栏单测：三重闸门 / 三态裁决 / 来源映射 / 故障放行。

护栏是「第二道出口防线」，它的正确性直接决定金融问答会不会给出错误数字，
所以这里把每个分支都钉死：fail 必须拒、pass 必须零打扰、warn 只提示不拒。
"""

import time

import pytest

from api.config import settings
from api.core.finance_guard import (
    docs_to_sources,
    notice_text,
    refusal_text,
    run_finance_guard,
    should_verify,
)


def _doc(title: str, text: str, *, ts: int | None = None, as_of: str | None = None) -> dict:
    d = {
        "doc_title": title,
        "file_name": f"{title}.txt",
        "content": text[:200],
        "context_full": text,
    }
    if ts is not None:
        d["update_time"] = ts
    if as_of is not None:
        d["as_of"] = as_of
    return d


FRESH = int(time.time())
OLD = int(time.time()) - 400 * 86400  # 超过 365 天上限


# ── 三重闸门 ──────────────────────────────────────────────
def test_skips_non_finance_question():
    """非金融问题不启用（避免拖累普通问答）。"""
    docs = [_doc("售后政策", "退货流程：订单页提交申请。", ts=FRESH)]
    assert run_finance_guard("怎么申请退货", "请在订单页提交退货申请。", docs) is None


def test_skips_when_no_docs():
    assert should_verify("公司营收多少", []) is False
    assert run_finance_guard("公司营收多少", "营收 1.2 亿元。", []) is None


def test_switch_off_disables_guard(monkeypatch):
    monkeypatch.setattr(settings, "finance_guard_enabled", False)
    docs = [_doc("年报", "营业收入 12000 万元。", ts=FRESH)]
    assert run_finance_guard("公司营收多少", "营业收入 1.2 亿元。", docs) is None


# ── 三态裁决 ──────────────────────────────────────────────
def test_pass_when_numbers_traceable():
    """1.2 亿元 与来源 12000 万元 语义等值（量级归一）→ pass，零打扰。"""
    docs = [_doc("2026年报", "公司 2026 年营业收入为 12000 万元。", ts=FRESH)]
    out = run_finance_guard("公司营收多少", "公司 2026 年营业收入为 1.2 亿元。", docs)
    assert out is not None and out.verdict == "pass"
    assert notice_text(out).startswith("\n\n（数值校验提示")


def test_fail_on_magnitude_tampering():
    """1.2 万元 vs 来源 12000 万元：差 4 个数量级 → fail（臆造/篡改）。"""
    docs = [_doc("2026年报", "公司 2026 年营业收入为 12000 万元。", ts=FRESH)]
    out = run_finance_guard("公司营收多少", "公司 2026 年营业收入为 1.2 万元。", docs)
    assert out is not None and out.verdict == "fail"
    assert "校验未通过" in refusal_text(out)


def test_fail_on_cross_source_conflict():
    """两个独立来源给出互斥数值 → fail（必有其一错误，拒答防错）。"""
    docs = [
        _doc("2026年报", "公司 2026 年营业收入为 12000 万元。", ts=FRESH),
        _doc("券商研报", "公司 2026 年营业收入为 9800 万元。", ts=FRESH),
    ]
    out = run_finance_guard("公司营收多少", "公司 2026 年营业收入为 1.2 亿元。", docs)
    assert out is not None and out.verdict == "fail"


def test_warn_on_stale_source():
    """来源过期 → warn（放行 + 提示），绝不拒答。"""
    docs = [_doc("2023年报", "公司 2023 年营业收入为 8000 万元。", ts=OLD)]
    out = run_finance_guard("公司营收多少", "公司营业收入为 8000 万元。", docs)
    assert out is not None and out.verdict == "warn"
    assert "过期" in out.summary


# ── 来源映射 ──────────────────────────────────────────────
def test_sources_name_unique_and_group_by_file():
    """chunk 级 name 必须唯一，group 用文件身份（跨源矛盾分级依赖它）。"""
    docs = [
        _doc("年报", "营收 12000 万元。", ts=FRESH),
        _doc("年报", "净利润 3000 万元。", ts=FRESH),
        _doc("研报", "营收 9800 万元。", ts=FRESH),
    ]
    srcs = docs_to_sources(docs)
    assert len({s.name for s in srcs}) == 3, "name 必须逐个 chunk 唯一"
    assert [s.group for s in srcs] == ["年报.txt", "年报.txt", "研报.txt"]


def test_as_of_field_takes_priority():
    """文档自带 as_of（报告期）优先于 update_time（入库时刻）。"""
    docs = [_doc("2026半年报", "净利润 3000 万元。", ts=OLD, as_of="2026-06-30")]
    out = run_finance_guard("净利润多少", "净利润为 3000 万元。", docs)
    assert out is not None and out.verdict == "pass"


def test_empty_docs_map_to_no_sources():
    assert docs_to_sources([]) == []
    assert docs_to_sources([{"doc_title": "空文档", "content": ""}]) == []


# ── 故障策略 ──────────────────────────────────────────────
def test_guard_failure_allows_answer(monkeypatch):
    """护栏自身故障必须放行，不阻断回答。"""
    import api.core.finance_guard as fg

    def _boom(*_a, **_kw):
        raise RuntimeError("engine down")

    monkeypatch.setattr(fg, "verify_finance_answer", _boom)
    docs = [_doc("年报", "营业收入 12000 万元。", ts=FRESH)]
    assert run_finance_guard("公司营收多少", "营业收入 1.2 亿元。", docs) is None
