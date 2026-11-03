"""TTFT（首 token 延迟）压测 — 命中 SSE 流首个 delta 事件的时间。

与 scripts/load_test.py 互补：load_test 测端到端总延迟，本脚本测
"用户多久看到第一个字"，这是流式体验的核心指标。

用法:
    python scripts/bench_ttft.py --url http://127.0.0.1:8911 \
        --question "退货流程是什么" --requests 30

依赖: httpx（已在项目依赖）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx

DEFAULT_QUESTION = "退货流程是什么"
CLIENT_TIMEOUT_S = 30.0


async def _one_ttft(client: httpx.AsyncClient, url: str, question: str) -> tuple[float, int]:
    """返回 (ttft_ms, http_status)。ttft = 发请求到收到首个 type=delta 事件的耗时。"""
    t0 = time.perf_counter()
    first_delta_ms = -1.0
    status = 0
    try:
        async with client.stream(
            "POST", f"{url}/api/v1/qa/ask/stream",
            json={"question": question, "top_k": 5}, timeout=CLIENT_TIMEOUT_S,
        ) as r:
            status = r.status_code
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
    except httpx.HTTPError:
        pass
    return (first_delta_ms if first_delta_ms >= 0 else (time.perf_counter() - t0) * 1000), status


async def _run(url: str, question: str, total: int):
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[_one_ttft(client, url, question) for _ in range(total)])
        wall = time.perf_counter() - t0

    ttfts = sorted(r[0] for r in results)
    codes = {}
    for _, c in results:
        codes[c] = codes.get(c, 0) + 1
    n = len(ttfts)

    def pct(p):
        return ttfts[min(n - 1, int(n * p))]

    print(f"请求数={n}  总耗时={wall:.1f}s")
    print(f"TTFT(ms)  P50={pct(0.50):.1f}  P95={pct(0.95):.1f}  P99={pct(0.99):.1f}  max={ttfts[-1]:.1f}")
    print(f"HTTP 状态: {codes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8911")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--requests", type=int, default=30)
    args = ap.parse_args()
    asyncio.run(_run(args.url.rstrip("/"), args.question, args.requests))


if __name__ == "__main__":
    main()
