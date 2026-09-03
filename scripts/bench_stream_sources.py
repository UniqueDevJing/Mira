#!/usr/bin/env python
"""P1-4 流式 sources 先行事件并发压测。

真实启动 uvicorn (127.0.0.1), 注入 FakeLLM 绕过外部 LLM 额度 (monkeypatch orchestrator.get_llm_client),
用 httpx.AsyncClient 并发打 POST /api/v1/qa/ask/stream, 解析 SSE, 统计:
  - sources 先行(final=False) 到达延迟 TTFB
  - 最终 sources(final=True) 到达延迟
  - 端到端(done) 延迟
  - 成功率 / 错误数

不依赖外部 LLM/鉴权: 127.0.0.1 绑定 + 鉴权默认关 + 限流默认关。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import types

# 离线环境变量: 避免 HF 下载 / 限流误触
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RAG_RATE_LIMIT_ENABLED", "0")
# 本地压测解耦 .env 的强制鉴权 (与 tests/conftest.py 一致): 127.0.0.1 绑定下匿名 admin
os.environ["RAG_API_KEY_ENABLED"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import uvicorn

# monkeypatch 必须打在 orchestrator 模块名上 (因其 `from api.core.llm_client import get_llm_client`
# 直接绑定了名字, 改 llm_client 模块属性不生效)。
from api.core import orchestrator as _orch


class FakeLLM:
    """压测用假 LLM: 即时返回, 不耗外部额度。"""

    base_url = "fake://local"
    model = "fake-llm"
    api_key = "fake"

    async def chat(self, messages, **kwargs):
        return types.SimpleNamespace(content="压测假答案", latency_ms=1.0)

    async def stream_chat(self, messages, **kwargs):
        for chunk in ("压测", "流式", "答案", "。"):
            yield {"type": "delta", "content": chunk}


_orch.get_llm_client = lambda *a, **k: FakeLLM()  # type: ignore[assignment]

from api.main import app


def _start_server(port: int) -> uvicorn.Server:
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    return server


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vs = sorted(values)
    idx = max(0, min(len(vs) - 1, round((q / 100) * (len(vs) - 1))))
    return round(vs[idx] * 1000, 1)


async def _one(client: httpx.AsyncClient, url: str, question: str, barrier: asyncio.Barrier) -> dict:
    await barrier.wait()
    start = time.perf_counter()
    first_sources_dt = final_sources_dt = done_dt = None
    err = status = None
    try:
        async with client.stream("POST", url, json={"question": question}) as resp:
            status = resp.status_code
            if status != 200:
                body = await resp.aread()
                err = f"HTTP {status}: {body[:160]!r}"
                return {"status": status, "err": err, "first_sources_dt": None,
                        "final_sources_dt": None, "done_dt": None, "e2e": time.perf_counter() - start}
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    ev = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter() - start
                et = ev.get("type")
                if et == "sources":
                    if first_sources_dt is None:
                        first_sources_dt = now
                    if ev.get("final"):
                        final_sources_dt = now
                elif et == "done":
                    done_dt = now
                elif et == "error":
                    err = ev.get("detail")
    except Exception as e:  # noqa: BLE001 — 压测需捕获连接/超时等, 计入错误
        err = f"{type(e).__name__}: {str(e)[:160]}"
    return {"status": status, "err": err, "first_sources_dt": first_sources_dt,
            "final_sources_dt": final_sources_dt, "done_dt": done_dt, "e2e": time.perf_counter() - start}


async def _bench(url: str, questions: list[str], concurrency: int, warmup: int = 1) -> list[dict]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for _ in range(warmup):
            async with client.stream("POST", url, json={"question": questions[0]}) as r:
                async for _ in r.aiter_lines():
                    pass
        barrier = asyncio.Barrier(concurrency)
        tasks = [_one(client, url, questions[i % len(questions)], barrier) for i in range(concurrency)]
        return await asyncio.gather(*tasks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--concurrency", type=str, default="10", help="逗号分隔的并发梯度, 如 1,10,20")
    ap.add_argument("--questions-file", default="data/eval/eval_dataset.json")
    ap.add_argument("--out", default="data/eval/stream_bench.json")
    args = ap.parse_args()

    srv = _start_server(args.port)
    time.sleep(3.0)  # 等 lifespan + 模型预热线程就绪

    # 显式预热 embedder + reranker, 确保各梯度公平(均走真实重排, 非降级跳过重排)
    try:
        from api.state import get_embedder, get_reranker

        get_embedder().embed_query("warmup")
        get_reranker().warmup()
    except Exception as e:  # noqa: BLE001 — 预热失败不阻断压测
        print(f"预热警告: {e}", file=sys.stderr)

    url = f"http://127.0.0.1:{args.port}/api/v1/qa/ask/stream"
    with open(args.questions_file, encoding="utf-8") as f:
        ds = json.load(f)
    gradients = [int(x) for x in args.concurrency.split(",") if x.strip()]
    reports = []
    for conc in gradients:
        questions = [f"压测 {i}: {d['question']}" for i, d in enumerate(ds[:conc])]
        results = asyncio.run(_bench(url, questions, conc))

        ok = [r for r in results if r["status"] == 200 and r["done_dt"] is not None]
        first_dt = [r["first_sources_dt"] for r in results if r["first_sources_dt"] is not None]
        final_dt = [r["final_sources_dt"] for r in results if r["final_sources_dt"] is not None]
        e2e = [r["e2e"] for r in results if r["e2e"] is not None]
        errs = [r["err"] for r in results if r["err"]]
        reports.append({
            "concurrency": conc,
            "total": len(results),
            "success": len(ok),
            "success_rate": round(len(ok) / len(results), 4) if results else 0,
            "first_sources_ttfb_ms_p50": _pct(first_dt, 50),
            "first_sources_ttfb_ms_p95": _pct(first_dt, 95),
            "final_sources_ms_p50": _pct(final_dt, 50),
            "final_sources_ms_p95": _pct(final_dt, 95),
            "e2e_ms_p50": _pct(e2e, 50),
            "e2e_ms_p95": _pct(e2e, 95),
            "errors": errs[:10],
        })
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"gradients": reports}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"gradients": reports}, ensure_ascii=False, indent=2))
    srv.should_exit = True


if __name__ == "__main__":
    main()
