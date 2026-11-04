"""全链路压测脚本 — 并发请求 /qa/ask，输出 QPS 与 P50/P95/P99。

用法:
    python scripts/load_test.py --url http://127.0.0.1:8000 \
        --question "退货流程是什么" --concurrency 20 --requests 200 \
        --out docs/benchmarks/load-$(git rev-parse --short HEAD).json
    # 或用文件提供多问题轮询
    python scripts/load_test.py --questions q.txt --concurrency 10 --requests 100

依赖: httpx (已在项目依赖)。报告降级等级分布, 便于发现 L1/L2/L3 频发。

口径（P1 / Round 3 修订）：
- 失败样本（异常 / HTTP 非 200）单独计数，**排除出延迟分位**，只进失败率与状态码分布
  （旧版把失败耗时也算进分位，会把"连不上"误读成"很慢但成功"）；
- 分位用 nearest-rank，与 bench_ttft 一致；
- QPS = 成功数 / 墙钟时间（不含失败，避免失败快而虚高）；
- `--out` 落盘 JSON，供 scripts/check_bench_regression.py 做阈值门禁。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import bench_common as bc
import httpx

DEFAULT_QUESTION = "退货流程是什么"
CLIENT_TIMEOUT_S = 30.0  # 大于 LLM 生成预算, 避免客户端超时被误计为 500


async def _one(client: httpx.AsyncClient, url: str, question: str):
    """返回 (latency_ms, status, degradation_level, ok)。"""
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{url}/api/v1/qa/ask",
            json={"question": question, "top_k": 5},
            timeout=CLIENT_TIMEOUT_S,
        )
        latency = (time.perf_counter() - t0) * 1000
        deg = 0
        try:
            deg = r.json().get("degradation_level", 0)
        except ValueError:
            deg = 0  # 非 JSON 响应(如 5xx 错误页), 降级等级未知
        return latency, r.status_code, deg, r.status_code == 200
    except httpx.HTTPError:
        return (time.perf_counter() - t0) * 1000, 0, -1, False


async def _run(url: str, questions: list[str], concurrency: int, total: int) -> dict:
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency)) as client:
        latencies: list[float] = []
        degs: dict[int, int] = {}
        codes: dict[int, int] = {}
        sem = asyncio.Semaphore(concurrency)

        async def _worker(i: int):
            q = questions[i % len(questions)]
            async with sem:
                lat, code, deg, ok = await _one(client, url, q)
                codes[code] = codes.get(code, 0) + 1
                degs[deg] = degs.get(deg, 0) + 1
                if ok:
                    latencies.append(lat)

        t0 = time.perf_counter()
        await asyncio.gather(*[_worker(i) for i in range(total)])
        wall = time.perf_counter() - t0

    n_ok = len(latencies)
    n_fail = total - n_ok
    qps = round(n_ok / wall, 2) if wall > 0 else 0.0

    print(f"请求数={total}  成功={n_ok}  失败={n_fail}  并发={concurrency}  "
          f"总耗时={wall:.1f}s  QPS={qps}")
    if latencies:
        s = bc.summarize(latencies, "latency_ms")
        p99_note = "  (n<100，仅供参考)" if n_ok < 100 else ""
        print(f"延迟(ms)  P50={s['latency_ms_p50']:.1f}  P95={s['latency_ms_p95']:.1f}  "
              f"P99={s['latency_ms_p99']:.1f}{p99_note}  max={s['latency_ms_max']:.1f}")
    else:
        print("延迟(ms)  无成功样本（全部失败）")
    print(f"HTTP 状态: {dict(sorted(codes.items()))}")
    print(f"降级等级分布: {dict(sorted(degs.items()))}  (0=正常 1=跳过rerank 2=仅BM25 3=LLM降级)")

    return bc.build_payload(
        "load_test",
        url=url,
        concurrency=concurrency,
        questions=len(questions),
        requests=total,
        ok=n_ok,
        failed=n_fail,
        failure_rate=round(n_fail / total, 4) if total else 0.0,
        wall_s=round(wall, 3),
        qps=qps,
        http_codes={str(k): v for k, v in sorted(codes.items())},
        degradation={str(k): v for k, v in sorted(degs.items())},
        **bc.summarize(latencies, "latency_ms"),
    )


def _load_questions(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--questions", default="", help="问题文件路径, 每行一个")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--out", default="", help="结果 JSON 落盘路径（供阈值门禁比对）")
    args = ap.parse_args()

    questions = _load_questions(args.questions) if args.questions else [args.question]
    if not questions:
        print("问题列表为空", file=sys.stderr)
        sys.exit(1)

    payload = asyncio.run(
        _run(args.url.rstrip("/"), questions, args.concurrency, args.requests))
    bc.emit(args.out, payload)


if __name__ == "__main__":
    main()