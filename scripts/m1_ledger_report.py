#!/usr/bin/env python
"""M1 错误账本报告 — 生成「badcase 分类归因表」。

两种用法:
  1) 离线演示/自检: 无参数运行, 对 5 类各构造一个合成样本走 classify(),
     打印「类别 / 英文键 / 归因规则 / 修复动作 / 示例命中」表, 验证分类器工作正常。
  2) 运行时聚合: 传入 JSONL 账本路径(或读 settings.m1_ledger_path),
     聚合真实 badcase 的每类计数、占比与按日期趋势, 供"每类占比趋势"指标。

示例:
  python scripts/m1_ledger_report.py                 # 演示表
  python scripts/m1_ledger_report.py ledger.jsonl    # 聚合真实账本
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# 允许从仓库根目录直接运行: 把 repo 根加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.core.error_ledger import (  # noqa: E402
    CLASS_FIX,
    CLASS_LABELS,
    NO_RECALL,
    LOW_RANK,
    CONTEXT_TRUNCATED,
    NUMBER_REWRITTEN,
    FALSE_REFUSAL,
    classify,
)


def _docs(scores, truncated=False, content="FastAPI 是一个高性能的 Python Web 框架"):
    return [
        {"doc_id": "d1", "content": content, "context_full": content, "score": s, "truncated": truncated}
        for s in scores
    ]


def _demo_classify_rows():
    """5 类 + 1 个正常样本, 证明分类优先级与规则生效。"""
    cases = [
        # (类, question, docs, top1, answer, refusal_reason, finance_verdict, truncated)
        (NO_RECALL, "如何申请退款", [], 0.0, "未在知识库找到", None, None, False),
        (FALSE_REFUSAL, "FastAPI 怎么用", _docs([0.85]), 0.85, "拒答", "low_relevance", None, False),
        (NUMBER_REWRITTEN, "账户余额多少", _docs([0.9]), 0.9, "余额 100 元", None, "fail", False),
        (LOW_RANK, "如何部署上线", _docs([0.3]), 0.3, "请按文档部署步骤执行", None, None, False),
        (CONTEXT_TRUNCATED, "如何部署上线", _docs([0.9], truncated=True), 0.9, "请按文档部署步骤执行", None, None, True),
        (None, "如何部署上线", _docs([0.95]), 0.95, "请按文档部署步骤执行", None, None, False),
    ]
    rows = []
    for expect, q, docs, top1, ans, rr, fv, tr in cases:
        cls, reason, fix = classify(q, docs, top1, ans, rr, fv, tr)
        assert cls == expect, f"期望 {expect}, 实得 {cls} (q={q})"
        rows.append((cls, q, reason, fix))
    return rows


def _print_demo_table():
    print("# M1 错误账本 · badcase 分类归因表（演示/自检）\n")
    print("| 类别 | 英文键 | 归因规则 | 修复动作 | 示例命中 |")
    print("| --- | --- | --- | --- | --- |")
    rows = _demo_classify_rows()
    for cls, q, reason, fix in rows[:-1]:  # 跳过最后一个 None(正常样本)
        label = CLASS_LABELS.get(cls, cls)
        print(f"| {label} | {cls} | {reason} | {fix} | {q} |")
    print("\n> 正常样本(非 badcase, 不入账):", rows[-1][0])


def _aggregate_ledger(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    by_class = Counter(r["class"] for r in rows)
    by_date: dict[str, Counter] = {}
    for r in rows:
        by_date.setdefault(r["date"], Counter())[r["class"]] += 1
    return total, by_class, by_date


def _print_ledger_report(path: Path):
    total, by_class, by_date = _aggregate_ledger(path)
    print(f"# M1 错误账本聚合报告 — {path.name}\n")
    print(f"总 badcase 数: {total}\n")
    print("## 每类占比\n")
    print("| 类别 | 英文键 | 计数 | 占比 |")
    print("| --- | --- | --- | --- |")
    for k in CLASS_LABELS:
        c = by_class.get(k, 0)
        pct = f"{c / total * 100:.1f}%" if total else "0%"
        print(f"| {CLASS_LABELS[k]} | {k} | {c} | {pct} |")
    print("\n## 按日期趋势 (date → {class: count})\n")
    for d in sorted(by_date):
        print(f"- {d}: {dict(by_date[d])}")


def main():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            print(f"账本文件不存在: {p}", file=sys.stderr)
            raise SystemExit(2)
        _print_ledger_report(p)
    else:
        _print_demo_table()


if __name__ == "__main__":
    main()
