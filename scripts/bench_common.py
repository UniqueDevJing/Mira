"""压测脚本共用：分位数、结果落盘、git 元信息。

设计约定（P1 / Round 3）：
- 分位数用 nearest-rank（`ceil(n*p)-1`），与 bench_ttft 保持一致，避免两套口径；
- 结果统一落 JSON，字段名稳定，供 `check_bench_regression.py` 做阈值门禁；
- 失败样本必须由调用方排除后再算分位数，本模块不做"兜底填充"。
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
from datetime import UTC, datetime


def nearest_rank(sorted_values: list[float], p: float) -> float:
    """nearest-rank 分位数；空列表返回 nan。"""
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * p) - 1))
    return sorted_values[idx]


def summarize(values: list[float], prefix: str) -> dict:
    """把一列样本压成 {prefix_p50/p95/p99/max/n}；空样本返回 nan。"""
    ordered = sorted(values)
    return {
        f"{prefix}_n": len(ordered),
        f"{prefix}_p50": round(nearest_rank(ordered, 0.50), 2),
        f"{prefix}_p95": round(nearest_rank(ordered, 0.95), 2),
        f"{prefix}_p99": round(nearest_rank(ordered, 0.99), 2),
        f"{prefix}_max": round(ordered[-1], 2) if ordered else float("nan"),
    }


def git_commit() -> str:
    """当前仓库短 commit；取不到就返回空串（CI/容器里可能没有 git）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build_payload(tool: str, **fields) -> dict:
    """统一的结果信封：谁、什么时候、在哪个 commit、什么参数、什么结论。"""
    return {
        "tool": tool,
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git_commit(),
        "app_commit": fields.pop("app_commit", ""),
        **fields,
    }


def emit(path: str | None, payload: dict) -> None:
    """落盘 JSON（父目录自动创建）；path 为空则只在 stdout 提示。"""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if not path:
        print("\n结果 JSON（未落盘，加 --out 可保存）:")
        print(text)
        return
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    print(f"\n结果已落盘: {target}")