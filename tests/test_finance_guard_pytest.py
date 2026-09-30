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


def test_cross_source_conflict_does_not_refuse():
    """跨源矛盾在**护栏层**不再拒答（收窄理由见 finance_guard 模块 docstring）。

    引擎原生仍判 fail —— 该行为由 engines/finance 的对抗门禁覆盖，不受本收窄影响。
    但 RAG 出口的 sources 是多指标检索块，"同一指标不同值"的判定会大面积误报，
    把正确答案拒掉的代价高于漏一次提示，故护栏只以 unsupported 作为拒答依据。
    """
    docs = [
        _doc("2026年报", "公司 2026 年营业收入为 12000 万元。", ts=FRESH),
        _doc("券商研报", "公司 2026 年营业收入为 9800 万元。", ts=FRESH),
    ]
    out = run_finance_guard("公司营收多少", "公司 2026 年营业收入为 1.2 亿元。", docs)
    assert out is not None
    assert out.verdict != "fail", "跨源矛盾不应导致拒答（会误拒正常回答）"


# ── 多指标段落回归（线上实测出来的误报，务必保持通过）─────────
# 财务年报段落在同一段里同时包含总营收、分部营收、同比百分比与章节序号「一、」，
# 引擎的 conflict / intra_conflict 会把这些**不同口径**的数字互相比较
# （实测：22.75% vs 23.42% 被当成同一指标矛盾；"一、" 与日期「12 月 31 日」被抽成数值），
# 进而在 RAG 出口把正确答案判成"来源矛盾"而拒答。
_MULTI_METRIC_PARA = (
    "一、主要会计数据\n"
    "2025 年度营业总收入为 128400 万元，2024 年度营业总收入为 104600 万元，同比增长 22.75%。\n"
    "2025 年度营业成本为 103300 万元，2024 年度营业成本为 83700 万元，同比增长 23.42%。\n"
    "2025 年度毛利为 25100 万元。\n"
    "二、分部情况\n"
    "2025 年度智能硬件业务营业收入为 62400 万元，云服务订阅业务营业收入为 38700 万元。\n"
    "三、截至 2025 年 12 月 31 日，资产负债率为 38.60%。"
)


def test_multi_metric_paragraph_does_not_refuse():
    """整段多指标报表文本不得触发拒答（正常回答要放行）。"""
    docs = [_doc("2025年报", _MULTI_METRIC_PARA, ts=FRESH)]
    out = run_finance_guard("公司营收多少", "2025 年度营业总收入为 128400 万元。", docs)
    assert out is not None and out.verdict == "pass", "多指标段落不应误拒正常回答"


def test_multi_metric_paragraph_still_catches_tampering():
    """反向断言：同一个多指标段落里，数量级被篡改仍必须拒答（别把误报修成漏报）。"""
    docs = [_doc("2025年报", _MULTI_METRIC_PARA, ts=FRESH)]
    out = run_finance_guard("公司营收多少", "2025 年度营业总收入为 128400 元。", docs)
    assert out is not None and out.verdict == "fail"
    assert "无来源支撑" in out.summary


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
