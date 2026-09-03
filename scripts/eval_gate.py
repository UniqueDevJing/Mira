#!/usr/bin/env python
"""召回回归门禁 (CI 用)。

比对当前检索评测 summary 与已提交基线, 关键召回指标回退超过容差则 exit 1 阻断 PR。

为什么只 gate 首阶段档:
  首阶段(--no-rerank) 390 问约 45s, 可在 CI 上跑;
  全量重排档 390 问在 CPU 上约 70min(Cross-Encoder 逐条重排), 不适合进 CI。
  首阶段指标对检索侧改动(融合/切分/增广/向量库)最敏感, 足以兜住召回回归。

用法:
  venv/Scripts/python.exe scripts/eval_retrieval.py --no-rerank --out /tmp/cur.json
  venv/Scripts/python.exe scripts/eval_gate.py \
      --baseline data/eval/baseline_retrieval_first_stage.json \
      --current  /tmp/cur.json

退出码: 0=通过(无回退或回退在容差内), 1=存在超出容差的回退, 2=用法/数据错误。
"""

import argparse
import json
import sys

# (指标, 容差类型) — pp=百分点(用于比率类), abs=绝对值(用于 MRR)
_GATE_METRICS = [
    ("recall@1", "pp"),
    ("recall@3", "pp"),
    ("recall@5", "pp"),
    ("recall@10", "pp"),
    ("hit@1", "pp"),
    ("mrr", "abs"),
]


def _load_metrics(path: str, branch: str) -> dict:
    """读取 summary 中指定分支的指标字典。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 顶层不是对象")
    if branch in data and isinstance(data[branch], dict):
        return data[branch], data.get("sample_count")
    # summary 直接就是指标字典(无分支嵌套)时也兼容
    if branch == "rerank_off" and "recall@1" in data:
        return data, data.get("sample_count")
    raise ValueError(f"{path}: 找不到分支 '{branch}' (可用: {[k for k in data if isinstance(data[k], dict)]})")


def _fmt(kind: str, v: float) -> str:
    return f"{v * 100:.1f}%" if kind == "pp" else f"{v:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="召回回归门禁")
    ap.add_argument("--baseline", required=True, help="已提交的基线 summary json")
    ap.add_argument("--current", required=True, help="本次评测产出的 summary json")
    ap.add_argument(
        "--branch",
        default="rerank_off",
        choices=["rerank_off", "rerank_on"],
        help="比对的分支(默认 rerank_off = 首阶段档)",
    )
    ap.add_argument("--tolerance-pp", type=float, default=1.0, help="比率类指标容差(百分点, 默认 1.0)")
    ap.add_argument("--mrr-tolerance", type=float, default=0.01, help="MRR 容差(绝对值, 默认 0.01)")
    args = ap.parse_args()

    try:
        base, base_n = _load_metrics(args.baseline, args.branch)
        cur, cur_n = _load_metrics(args.current, args.branch)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[eval-gate] 无法读取评测结果: {e}", file=sys.stderr)
        return 2

    print("=== 召回回归门禁 ===")
    print(f"基线: {args.baseline} (n={base_n})")
    print(f"当前: {args.current} (n={cur_n})")
    print(f"分支: {args.branch}   容差: 比率 {args.tolerance_pp}pp / MRR {args.mrr_tolerance}")
    if base_n and cur_n and base_n != cur_n:
        print(f"⚠️  样本数不一致 ({base_n} -> {cur_n}), 指标不可直接比较")
    print()
    print(f"{'指标':<12} {'基线':>9} {'当前':>9} {'变化':>9}  {'判定':<6}")
    print("-" * 52)

    regressions = []
    for key, kind in _GATE_METRICS:
        bv, cv = base.get(key), cur.get(key)
        if bv is None:
            continue  # 基线没有该指标 → 不设门禁
        if cv is None:
            print(f"{key:<12} {_fmt(kind, bv):>9} {'缺失':>9} {'-':>9}  {'FAIL':<6}")
            regressions.append((key, None))
            continue
        tol = args.tolerance_pp / 100.0 if kind == "pp" else args.mrr_tolerance
        delta = cv - bv
        drop = bv - cv  # 正数 = 回退幅度
        if drop > tol:
            verdict = "FAIL"
            regressions.append((key, drop))
        else:
            verdict = "ok"
        sign = f"{delta * 100:+.1f}pp" if kind == "pp" else f"{delta:+.3f}"
        print(f"{key:<12} {_fmt(kind, bv):>9} {_fmt(kind, cv):>9} {sign:>9}  {verdict:<6}")

    print("-" * 52)
    if not regressions:
        print("✅ 通过: 无超出容差的召回回退")
        return 0
    print(f"❌ 阻断: {len(regressions)} 项指标回退超容差")
    for key, drop in regressions:
        if drop is None:
            print(f"   - {key}: 当前结果缺失")
        else:
            d = f"{drop * 100:.1f}pp" if key != "mrr" else f"{drop:.3f}"
            print(f"   - {key}: 回退 {d}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
