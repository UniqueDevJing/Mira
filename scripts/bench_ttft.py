"""TTFT（首 token 延迟）压测 — 命中 SSE 流首个 delta 事件的时间。

与 scripts/load_test.py 互补：load_test 测端到端总延迟，本脚本测
"用户多久看到第一个字"，这是流式体验的核心指标。

用法:
    # 顺序单发基线（编排层开销，默认）
    python scripts/bench_ttft.py --url http://127.0.0.1:8911 \
        --question "退货流程是什么" --requests 100 --concurrency 1
    # 并发压测
    python scripts/bench_ttft.py --url http://127.0.0.1:8911 --requests 100 --concurrency 20

依赖: httpx（已在项目依赖）。

口径说明（修复协同评审 P0-3/4/5）：
- --concurrency 控制并发：1=顺序单发，>1=并发。原脚本实际并发 30 却标「顺序」，已更正。
- 失败的请求（异常或 HTTP 非 200）单独计数并排除出延迟分位，不再被计为高延迟成功样本。
- 默认先发 3 次预热（丢弃）以排除连接建立/服务冷启动；--requests 默认 100（样本量足够 P99 才有意义）。
- 分位用 nearest-rank；成功样本 < 100 时 P99 标注「仅供参考」。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time

import bench_common as bc
import httpx

DEFAULT_QUESTION = "退货流程是什么"
CLIENT_TIMEOUT_S = 30.0


async def _one_ttft(client: httpx.AsyncClient, url: str, question: str,
                    sem: asyncio.Semaphore) -> tuple[float, int, bool]:
    """返回 (ttft_ms, http_status, ok)。

    ttft = 发请求到收到首个 type=delta 事件的耗时。
    ok=False 表示请求失败（异常或 HTTP 非 200），不计入延迟分位，只计入失败率。
    """
    async with sem:
        t0 = time.perf_counter()
        first_delta_ms = -1.0
        status = 0
        ok = False
        try:
            async with client.stream(
                "POST", f"{url}/api/v1/qa/ask/stream",
                json={"question": question, "top_k": 5}, timeout=CLIENT_TIMEOUT_S,
            ) as r:
                status = r.status_code
                if r.status_code != 200:
                    return (0.0, status, False)
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if not payload:
                        continue
                    try:
                        ev = json.loads(payload)
                    except ValueError:
                        continue
                    if ev.get("type") == "delta":
                        first_delta_ms = (time.perf_counter() - t0) * 1000
                        break
            ok = first_delta_ms >= 0
        except httpx.HTTPError:
            pass
        return (first_delta_ms if ok else 0.0, status, ok)


async def _warmup(client: httpx.AsyncClient, url: str, question: str, n: int, concurrency: int):
    if n <= 0:
        return
    sem = asyncio.Semaphore(max(1, concurrency))
    await asyncio.gather(*[_one_ttft(client, url, question, sem) for _ in range(n)])


async def _run(url: str, question: str, total: int, concurrency: int, warmup: int):
    sem = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient() as client:
        await _warmup(client, url, question, warmup, concurrency)
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[_one_ttft(client, url, question, sem) for _ in range(total)])
        wall = time.perf_counter() - t0

    ttfts = sorted(r[0] for r in results if r[2])
    failures = [r for r in results if not r[2]]
    n_total = len(results)
    n_ok = len(ttfts)
    n_fail = n_total - n_ok

    def pct(p: float) -> float:
        if not ttfts:
            return float("nan")
        idx = min(n_ok - 1, int(math.ceil(n_ok * p)) - 1)
        return ttfts[idx]

    fail_codes: dict[int, int] = {}
    for _, c, _ in failures:
        fail_codes[c] = fail_codes.get(c, 0) + 1

    print(f"请求数={n_total}  预热={warmup}  并发={concurrency}  总耗时={wall:.1f}s")
    print(f"成功={n_ok}  失败={n_fail}  失败率={(n_fail / n_total * 100) if n_total else 0:.1f}%")
    if ttfts:
        p99_note = "  (n<100，仅供参考)" if n_ok < 100 else ""
        print(f"TTFT(ms)  P50={pct(0.50):.1f}  P95={pct(0.95):.1f}  "
              f"P99={pct(0.99):.1f}{p99_note}  max={ttfts[-1]:.1f}")
    else:
        print("TTFT(ms)  无成功样本（全部失败）")
    print(f"失败 HTTP 状态: {fail_codes if fail_codes else '无'}")

    return bc.build_payload(
        "bench_ttft",
        url=url,
        question=question,
        concurrency=concurrency,
        warmup=warmup,
        requests=n_total,
        ok=n_ok,
        failed=n_fail,
        failure_rate=round((n_fail / n_total), 4) if n_total else 0.0,
        failure_codes={str(k): v for k, v in fail_codes.items()},
        wall_s=round(wall, 3),
        **bc.summarize(ttfts, "ttft_ms"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8911")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="并发数：1=顺序单发（默认，测编排层基线），>1=并发压测")
    ap.add_argument("--warmup", type=int, default=3,
                    help="预热请求数（丢弃，不计入分位）；排除连接建立/冷启动")
    ap.add_argument("--out", default="", help="结果 JSON 落盘路径（供阈值门禁比对）")
    args = ap.parse_args()
    payload = asyncio.run(_run(args.url.rstrip("/"), args.question,
                               args.requests, args.concurrency, args.warmup))
    bc.emit(args.out, payload)


if __name__ == "__main__":
    main()
