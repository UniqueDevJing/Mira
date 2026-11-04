"""性能回归门禁：把当前压测结果与基线比对，超出阈值即失败（exit 1）。

用法:
    # 1) 先把"可接受"的一次结果固化为基线（人工确认后提交）
    python scripts/load_test.py --url http://127.0.0.1:8911 \
        --concurrency 20 --requests 200 --out docs/benchmarks/load-baseline.json

    # 2) 之后每次改动，跑同一套参数并比对
    python scripts/load_test.py ... --out /tmp/load-current.json
    python scripts/check_bench_regression.py \
        --baseline docs/benchmarks/load-baseline.json \
        --current /tmp/load-current.json \
        --metric latency_ms.p95 --max-regression 0.10

判定规则：
- 延迟类指标（*_ms.*）：**上升**超过 max-regression 即失败（越小越好）；
- 吞吐类指标（qps）：**下降**超过 max-regression 即失败（越大越好，方向自动反转）；
- 失败率（failure_rate）：绝对差超过 --max-failure-delta（默认 0.02）即失败；
- 基线里缺该指标、或当前结果样本为 0，直接失败并说明原因（不允许"无声通过"）。

退出码：0 通过 / 1 回归或不可比 / 2 参数或文件错误。
"""

from __future__ import annotations

import argparse
import json
import sys

THROUGHPUT_KEYS = ("qps",)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dig(payload: dict, dotted: str) -> float | None:
    """取指标值：同时支持嵌套（latency_ms.p95）与落盘用的扁平键（latency_ms_p95）。"""
    node = payload
    nested_ok = True
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            nested_ok = False
            break
        node = node[part]
    if not nested_ok:
        node = payload.get(dotted.replace(".", "_"))
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    return float(node)


def is_throughput(metric: str) -> bool:
    return metric in THROUGHPUT_KEYS


def compare(baseline: float, current: float, metric: str, max_regression: float) -> tuple[bool, float]:
    """返回 (是否通过, 变化率)。变化率为正表示"变差"。"""
    if baseline == 0:
        return (True, 0.0) if current == 0 else (False, float("inf"))
    if is_throughput(metric):
        change = (baseline - current) / baseline  # 吞吐下降 = 变差
    else:
        change = (current - baseline) / baseline  # 延迟上升 = 变差
    return change <= max_regression, change


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="基线 JSON 路径")
    ap.add_argument("--current", required=True, help="本次结果 JSON 路径")
    ap.add_argument("--metric", default="latency_ms.p95",
                    help="要比的指标路径，如 latency_ms.p95 / ttft_ms.p95 / qps")
    ap.add_argument("--max-regression", type=float, default=0.10,
                    help="允许的最大变差比例（0.10 = 10%%）")
    ap.add_argument("--max-failure-delta", type=float, default=0.02,
                    help="失败率允许的绝对增量（0.02 = 2 个百分点）")
    args = ap.parse_args()

    try:
        base = load(args.baseline)
        cur = load(args.current)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[EXIT 2] 读取结果失败: {exc}")
        return 2

    base_v = dig(base, args.metric)
    cur_v = dig(cur, args.metric)
    if base_v is None or cur_v is None:
        print(f"[EXIT 1] 指标 {args.metric} 缺失（baseline={base_v} current={cur_v}），"
              f"无法比对；请先在两边用同一套参数采集。")
        return 1
    if cur.get("ok", 0) == 0:
        print("[EXIT 1] 本次无成功样本（ok=0），拒绝判定为通过。")
        return 1
    if base.get("concurrency") != cur.get("concurrency"):
        print(f"[EXIT 1] 并发口径不一致（baseline={base.get('concurrency')} "
              f"current={cur.get('concurrency')}），不可比。")
        return 1

    passed, change = compare(base_v, cur_v, args.metric, args.max_regression)
    direction = "下降" if is_throughput(args.metric) else "上升"
    print(f"指标      {args.metric}")
    print(f"基线      {base_v:.2f}   ({base.get('commit') or 'n/a'} / {base.get('ts')})")
    print(f"本次      {cur_v:.2f}   ({cur.get('commit') or 'n/a'} / {cur.get('ts')})")
    print(f"变化      {change * 100:+.1f}%  （{direction}即变差；阈值 {args.max_regression * 100:.0f}%）")

    base_f = base.get("failure_rate")
    cur_f = cur.get("failure_rate")
    failure_ok = True
    if isinstance(base_f, (int, float)) and isinstance(cur_f, (int, float)):
        delta = cur_f - base_f
        failure_ok = delta <= args.max_failure_delta
        print(f"失败率    {base_f * 100:.1f}% -> {cur_f * 100:.1f}%  "
              f"({delta * 100:+.1f}pp；阈值 +{args.max_failure_delta * 100:.0f}pp)")

    if passed and failure_ok:
        print("\n[PASS] 性能未回归")
        return 0
    print("\n[FAIL] 性能回归或失败率上升，请查看 docs/performance-benchmark.md 的调优建议")
    return 1


if __name__ == "__main__":
    sys.exit(main())