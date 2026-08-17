"""全链路压测脚本 — 并发请求 /qa/ask，输出 P50/P95/P99。

用法:
    python scripts/load_test.py --url http://127.0.0.1:8000 \
        --question "退货流程是什么" --concurrency 20 --requests 200
    # 或用文件提供多问题轮询
    python scripts/load_test.py --questions q.txt --concurrency 10 --requests 100

依赖: httpx (已在项目依赖)。报告降级等级分布, 便于发现 L1/L2/L3 频发。
"""

import argparse
import asyncio
import sys
import time

import httpx

DEFAULT_QUESTION = "退货流程是什么"
CLIENT_TIMEOUT_S = 30.0  # 大于 LLM 生成预算, 避免客户端超时被误计为 500


async def _one(client: httpx.AsyncClient, url: str, question: str):
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{url}/api/v1/qa/ask", json={"question": question, "top_k": 5}, timeout=CLIENT_TIMEOUT_S)
        latency = (time.perf_counter() - t0) * 1000
        deg = 0
        try:
            data = r.json()
            deg = data.get("degradation_level", 0)
        except ValueError:
            deg = 0  # 非 JSON 响应(如 5xx 错误页), 降级等级未知
        return latency, r.status_code, deg
    except httpx.HTTPError:
        return (time.perf_counter() - t0) * 1000, 0, -1


async def _run(url: str, questions, concurrency: int, total: int):
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency)) as client:
        latencies, degs, codes = [], {}, {}
        sem = asyncio.Semaphore(concurrency)

        async def _worker(i):
            q = questions[i % len(questions)]
            async with sem:
                lat, code, deg = await _one(client, url, q)
                latencies.append(lat)
                degs[deg] = degs.get(deg, 0) + 1
                codes[code] = codes.get(code, 0) + 1

        t0 = time.perf_counter()
        await asyncio.gather(*[_worker(i) for i in range(total)])
        wall = time.perf_counter() - t0

    latencies.sort()
    n = len(latencies)
    pct = lambda p: latencies[min(n - 1, int(n * p))]
    print(f"请求数={n}  并发={concurrency}  总耗时={wall:.1f}s  QPS={n / wall:.1f}")
    print(f"延迟(ms)  P50={pct(0.50):.1f}  P95={pct(0.95):.1f}  P99={pct(0.99):.1f}  max={latencies[-1]:.1f}")
    print(f"HTTP 状态: {codes or {0: n}}")
    print(f"降级等级分布: {dict(sorted(degs.items()))}  (0=正常 1=跳过rerank 2=仅BM25 3=LLM降级)")


def _load_questions(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--questions", default="", help="问题文件路径, 每行一个")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--requests", type=int, default=200)
    args = ap.parse_args()

    questions = _load_questions(args.questions) if args.questions else [args.question]
    if not questions:
        print("问题列表为空", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_run(args.url.rstrip("/"), questions, args.concurrency, args.requests))


if __name__ == "__main__":
    main()
