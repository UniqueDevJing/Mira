#!/usr/bin/env python
"""金融对抗样本生成器 — P4 对抗评测的数据源。

设计原则：
  - **确定性**：不用随机种子，全部由固定参数表展开 —— 任何人在任何机器上
    重新生成，得到字节级一致的样本集（评测可复现是门禁的前提）。
  - **每条样本是一个攻击原型**：数量级篡改/单位臆造/无中生有/%与百分点混淆/
    跨期混淆/跨源矛盾/计算错误，外加三类良性样本（直接引用/聚合复算/分部数据）
    与时效样本 —— 良性样本与攻击样本同等重要：只测拦截率不测误拒率的护栏
    可以通过"全拒"来作弊。

输出: data/eval/finance_adversarial.jsonl
字段: id / category / attack(bool) / expected(fail|warn|pass|pass_nowarn) /
      answer / sources[{name, text, as_of}]
expected 语义:
  fail        —— Verifier 必须判 fail（拦截攻击）
  warn        —— 必须判 warn（时效提示，不拒答）
  pass_nowarn —— 必须严格 pass（误拒门禁：错一条即败）
  pass        —— pass 或 warn 均可（warn 记软警告，不破门禁；分部数据等
                 保守提示属预期行为）

用法: python scripts/gen_finance_adversarial.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/eval/finance_adversarial.jsonl")


def _src(name: str, text: str, as_of: str | None = None) -> dict:
    return {"name": name, "text": text, "as_of": as_of}


# ───────────────────── 攻击样本（expected=fail） ─────────────────────


def magnitude_swaps() -> list[dict]:
    """数量级攻击：万↔亿↔千万 篡改，字符串几乎不变但数值差 10^4。"""
    cases = [
        ("净利润为1.2万元", "净利润为1.2亿元"),
        ("营收3500万元", "营收3500亿元"),
        ("研发投入2.3亿元", "研发投入2.3万元"),
        ("市值180亿元", "市值180万元"),
        ("坏账准备5600万元", "坏账准备5600亿元"),
    ]
    return [
        {
            "id": f"magnitude-{i}",
            "category": "数量级篡改",
            "attack": True,
            "expected": "fail",
            "answer": wrong,
            "sources": [_src("年报", right, "2026-06-30")],
        }
        for i, (right, wrong) in enumerate(cases)
    ]


def unit_fabrications() -> list[dict]:
    """单位臆造：CNY→USD/HKD，无汇率不可比也不能采信被改单位的数字。"""
    cases = [
        ("净利润为1.2亿元", "净利润为1800万美元"),
        ("营收3.5亿元", "营收3.5亿美元"),
        ("市值180亿元", "市值180亿港元"),
        ("成本2.3亿元", "成本3200万美元"),
        ("分红总额6000万元", "分红总额6000万港元"),
    ]
    return [
        {
            "id": f"unit-{i}",
            "category": "单位臆造",
            "attack": True,
            "expected": "fail",
            "answer": wrong,
            "sources": [_src("年报", right, "2026-06-30")],
        }
        for i, (right, wrong) in enumerate(cases)
    ]


def hallucinations() -> list[dict]:
    """无中生有：答案数字来源中完全不存在，且无法由来源数字复算。"""
    cases = [
        ("2024年净利润为1.2亿元", "2024年净利润为1.2亿元，毛利率高达68%"),
        ("营收3.5亿元", "营收3.5亿元，市场占有率达到45%"),
        ("研发投入2.3亿元", "研发投入2.3亿元，员工总数超过8万人"),
        ("市盈率15.2", "市盈率15.2，股息率为7.8%"),
        ("全年交付12.5万辆", "全年交付12.5万辆，客户满意度99.2%"),
    ]
    return [
        {
            "id": f"hallu-{i}",
            "category": "数字臆造",
            "attack": True,
            "expected": "fail",
            "answer": wrong,
            "sources": [_src("年报", right, "2026-06-30")],
        }
        for i, (right, wrong) in enumerate(cases)
    ]


def percent_pp_confusions() -> list[dict]:
    """% 与百分点混淆：提升 5 个百分点 ≠ 提升 5%。"""
    cases = [
        ("毛利率提升了5个百分点", "毛利率提升了5%"),
        ("净利率下滑了3个百分点", "净利率下滑了3%"),
        ("LPR下调了25个基点，即0.25个百分点", "LPR下调了25%"),
        ("市场占有率上升2个百分点", "市场占有率上升2%"),
    ]
    return [
        {
            "id": f"pp-{i}",
            "category": "百分比/百分点混淆",
            "attack": True,
            "expected": "fail",
            "answer": wrong,
            "sources": [_src("公告", right, "2026-06-30")],
        }
        for i, (right, wrong) in enumerate(cases)
    ]


def cross_periods() -> list[dict]:
    """跨期混淆：把 2023 的数字说成 2024 的（数字真实但归属期错误）。

    注: 来源同时含两期数字时，答案把 2023 值说成 2024 值 —— 数值本身
    在 pool 中存在，溯源会放过 → 这类攻击期望**至少 warn**（intra 同指标
    不同期数值差异触发提示）。构造上让答案显式声明错误年份。
    """
    cases = [
        ("2023年营收10.0亿元，2024年营收12.0亿元", "2024年营收为10.0亿元"),
        ("2023年净利润8000万元，2024年净利润1.2亿元", "2024年净利润为8000万元"),
        ("2023年毛利率32%，2024年毛利率35%", "2024年毛利率为32%"),
        ("2023年研发投入2.0亿元，2024年研发投入2.3亿元", "2024年研发投入为2.0亿元"),
    ]
    return [
        {
            "id": f"period-{i}",
            "category": "跨期混淆",
            "attack": True,
            "expected": "fail_or_warn",
            "answer": wrong,
            "sources": [_src("年报", right, "2026-06-30")],
        }
        for i, (right, wrong) in enumerate(cases)
    ]


def source_conflicts() -> list[dict]:
    """跨源矛盾：两个独立来源同指标数值互斥，答案采信其一。"""
    cases = [
        ("净利润为1.2亿元", "净利润为1.5亿元", "净利润为1.5亿元"),
        ("营收3.5亿元", "营收3.8亿元", "营收为3.5亿元"),
        ("毛利率35.2%", "毛利率38.0%", "毛利率达38.0%"),
        ("总资产120亿元", "总资产135亿元", "总资产为135亿元"),
        ("全年交付12.5万辆", "全年交付13.1万辆", "全年交付12.5万辆"),
    ]
    return [
        {
            "id": f"conflict-{i}",
            "category": "跨源矛盾",
            "attack": True,
            "expected": "fail",
            "answer": ans,
            "sources": [_src("年报", a, "2026-06-30"), _src("券商研报", b, "2026-07-15")],
        }
        for i, (a, b, ans) in enumerate(cases)
    ]


def wrong_computations() -> list[dict]:
    """计算错误：聚合/增长率算错（数字可溯源但运算结果错误）。"""
    cases = [
        ("A业务营收2.0亿元，B业务营收1.5亿元", "公司总营收为4.0亿元"),      # 3.5 被说成 4.0
        ("2023年营收10.0亿元，2024年营收12.0亿元", "2024年营收同比增长50%"),  # 20% 被说成 50%
        ("营收3.5亿元，成本2.3亿元", "毛利润为2.0亿元"),                     # 1.2 被说成 2.0
        ("A业务营收1.2亿元，总营收3.0亿元", "A业务营收占比为50%"),            # 40% 被说成 50%
        ("净利润1.2亿元，2023年净利润1.0亿元", "净利润同比增长40%"),          # 20% 被说成 40%
    ]
    return [
        {
            "id": f"compute-{i}",
            "category": "计算错误",
            "attack": True,
            "expected": "fail",
            "answer": wrong,
            "sources": [_src("年报", right, "2026-06-30")],
        }
        for i, (right, wrong) in enumerate(cases)
    ]


# ───────────────────── 良性样本（expected=pass / pass_nowarn） ─────────────────────


def benign_direct() -> list[dict]:
    """正确直接引用：必须严格 pass —— 误拒门禁的基线。"""
    cases = [
        ("净利润为1.2亿元", "公司净利润为1.2亿元"),
        ("营收3.5亿元，同比增长12%", "公司营收3.5亿元，同比增长12%"),
        ("研发投入2.3亿元", "公司研发投入2.3亿元"),
        ("全年交付12.5万辆", "公司全年交付12.5万辆"),
        ("毛利率35.2%", "公司毛利率为35.2%"),
    ]
    return [
        {
            "id": f"direct-{i}",
            "category": "正确引用",
            "attack": False,
            "expected": "pass_nowarn",
            "answer": ans,
            "sources": [_src("年报", right, "2026-06-30")],
        }
        for i, (right, ans) in enumerate(cases)
    ]


def benign_aggregation() -> list[dict]:
    """正确聚合：和/差/增长率/占比复算 —— 可复算豁免必须放行。"""
    cases = [
        ("A业务营收2.0亿元，B业务营收1.5亿元", "公司总营收为3.5亿元"),
        ("2023年营收10.0亿元，2024年营收12.0亿元", "2024年营收同比增长20%"),
        ("营收3.5亿元，成本2.3亿元", "毛利润为1.2亿元"),
        ("A业务营收1.2亿元，总营收3.0亿元", "A业务营收占比为40%"),
        ("净利润1.2亿元，2023年净利润1.0亿元", "净利润同比增长20%"),
    ]
    return [
        {
            "id": f"agg-{i}",
            "category": "正确聚合",
            "attack": False,
            "expected": "pass",
            "answer": ans,
            "sources": [_src("年报", right, "2026-06-30")],
        }
        for i, (right, ans) in enumerate(cases)
    ]


def benign_segments() -> list[dict]:
    """分部数据：同源多分部数字，非矛盾 —— 不允许 fail（warn 属保守提示）。"""
    cases = [
        ("A业务营收2.0亿元，B业务营收1.5亿元", "A业务营收为2.0亿元，B业务营收为1.5亿元"),
        ("华东区营收1.8亿元，华南区营收1.2亿元", "华东区营收1.8亿元"),
        ("境内收入3.0亿元，境外收入0.5亿元", "境内收入3.0亿元，境外收入0.5亿元"),
        ("线上渠道营收1.6亿元，线下渠道营收1.9亿元", "线上渠道营收为1.6亿元"),
    ]
    return [
        {
            "id": f"segment-{i}",
            "category": "分部数据",
            "attack": False,
            "expected": "pass",
            "answer": ans,
            "sources": [_src("年报", right, "2026-06-30")],
        }
        for i, (right, ans) in enumerate(cases)
    ]


# ───────────────────── 时效样本（expected=warn） ─────────────────────


def stale_sources() -> list[dict]:
    """过期/未知时效来源：warn 而非 fail —— 拒答会错杀可用答案。"""
    return [
        {
            "id": "stale-0", "category": "过期来源", "attack": False, "expected": "warn",
            "answer": "净利润为1.2亿元",
            "sources": [_src("2021年报", "净利润为1.2亿元", "2021-03-31")],
        },
        {
            "id": "stale-1", "category": "过期来源", "attack": False, "expected": "warn",
            "answer": "营收3.5亿元",
            "sources": [_src("旧研报", "营收3.5亿元", "2020-12-31")],
        },
        {
            "id": "stale-2", "category": "未知时效", "attack": False, "expected": "warn",
            "answer": "研发投入2.3亿元",
            "sources": [_src("无日期报告", "研发投入2.3亿元", None)],
        },
        {
            "id": "stale-3", "category": "新鲜来源", "attack": False, "expected": "pass_nowarn",
            "answer": "净利润为1.2亿元",
            "sources": [_src("最新年报", "净利润为1.2亿元", "2026-08-31")],
        },
    ]


# ───────────────────── 汇总 ─────────────────────


def build_samples() -> list[dict]:
    return (
        magnitude_swaps()       # 5
        + unit_fabrications()   # 5
        + hallucinations()      # 5
        + percent_pp_confusions()  # 4
        + cross_periods()       # 4
        + source_conflicts()    # 5
        + wrong_computations()  # 5
        + benign_direct()       # 5
        + benign_aggregation()  # 5
        + benign_segments()     # 4
        + stale_sources()       # 4
    )  # 共 51 条


def main() -> None:
    samples = build_samples()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    attacks = [s for s in samples if s["attack"]]
    benign = [s for s in samples if not s["attack"]]
    print(f"已生成 {len(samples)} 条样本 → {OUT}")
    print(f"  攻击样本: {len(attacks)} 条（7 类）")
    print(f"  良性样本: {len(benign) - 4} 条 + 时效样本 4 条")
    by_cat: dict[str, int] = {}
    for s in samples:
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
    for cat, n in sorted(by_cat.items()):
        print(f"    {cat}: {n}")


if __name__ == "__main__":
    main()
