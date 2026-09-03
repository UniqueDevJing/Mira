"""A/B 评估对比 — 同问题双配置跑分 diff (P2#10)。

用法:
  python scripts/ab_eval.py --a data/eval-summary-A.json --b data/eval-summary-B.json \
      --label-a "baseline" --label-b "candidate"

比较两个 evaluate.py 产出的 eval-summary.json, 输出整体 + RAGAS + 各 KB 的指标差异,
对回退(下降)指标打 ↓ 标记, 末尾汇总回退项数量与明细。不依赖 LLM, 纯离线。
"""

import argparse
import json


def _load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("summary", d) if isinstance(d, dict) else {}


# (指标, 越高越好?) — 决定回退方向
_METRICS = [
    ("accuracy", True),
    ("recall", True),
    ("recall_doc", True),
    ("precision", True),
    ("mrr", True),
    ("routing_accuracy", True),
    ("refusal_rate", False),
    ("hallucination_rate", False),
    ("hallucination_rate_non_refusal", False),
]
_RAGAS_KEYS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")


def _fmt(v):
    if v is None:
        return "  n/a"
    if isinstance(v, float):
        return f"{v * 100:5.1f}%"
    return str(v)


def _compare_block(name, sa, sb, higher_map, out):
    out.append(f"[{name}]")
    regressions = 0
    for key, higher in higher_map:
        va, vb = sa.get(key), sb.get(key)
        if va is None and vb is None:
            continue
        marker = " "
        if va is not None and vb is not None:
            d = vb - va
            if (higher and d < -1e-9) or ((not higher) and d > 1e-9):
                marker = "↓"  # 回退
                regressions += 1
            elif abs(d) > 1e-9:
                marker = "↑"
        out.append(f"  {marker} {key:<28} {_fmt(va):>8}  ->  {_fmt(vb):>8}")
    return regressions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="A 侧 eval-summary.json (baseline)")
    ap.add_argument("--b", required=True, help="B 侧 eval-summary.json (candidate)")
    ap.add_argument("--label-a", default="A(baseline)")
    ap.add_argument("--label-b", default="B(candidate)")
    args = ap.parse_args()

    sa, sb = _load(args.a), _load(args.b)
    out = [f"A/B 对比: {args.label_a} vs {args.label_b}", "=" * 60]
    total_regress = 0

    total_regress += _compare_block("OVERALL", sa, sb, _METRICS, out)

    ra, rb = sa.get("ragas") or {}, sb.get("ragas") or {}
    ragas_lines = []
    for k in _RAGAS_KEYS:
        va, vb = ra.get(k), rb.get(k)
        marker = " "
        if va is not None and vb is not None:
            d = vb - va
            if d < -1e-9:
                marker = "↓"
            elif d > 1e-9:
                marker = "↑"
        ragas_lines.append(f"  {marker} {k:<28} {_fmt(va):>8}  ->  {_fmt(vb):>8}")
    if ragas_lines:
        out.append("[RAGAS]")
        out.extend(ragas_lines)
        total_regress += sum(1 for ln in ragas_lines if ln.strip().startswith("↓"))

    by_a, by_b = sa.get("by_kb") or {}, sb.get("by_kb") or {}
    for kb in sorted(set(by_a) | set(by_b)):
        total_regress += _compare_block(f"KB={kb}", by_a.get(kb, {}), by_b.get(kb, {}), _METRICS, out)

    out.append("=" * 60)
    if total_regress == 0:
        out.append("✅ 无指标回退 (A→B 全部持平或提升)")
    else:
        out.append(f"⚠️  共 {total_regress} 项指标回退, 详见上方 ↓ 标记")
    print("\n".join(out))


if __name__ == "__main__":
    main()
