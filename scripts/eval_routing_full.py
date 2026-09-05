"""全量路由准确率评测 — 80 题 × 10 知识库 × 真实路由器 (规则 + LLM 兜底)。

数据源: tests/eval_dataset_kb.json (gen_kb_eval.py 生成, KB 标签在生产 SKILLS 体系内)。
每问调用线上同款 IntentRouter.route (async), 对比 RoutingResult.kb 与期望 KB。
输出: 总体命中率 / 各 KB 命中率 / 路由来源分布 (rule/llm/fallback) / 错路由明细。

用法: python scripts/eval_routing_full.py [--limit 0] [--concurrency 6]
"""
import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DATASET = os.path.join("tests", "eval_dataset_kb.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    from engines.router.intent_router import IntentRouter

    from api.core.llm_client import get_llm_client
    router = IntentRouter(llm_client=get_llm_client())  # 线上同款: 规则命中直达, 模糊时 LLM 兜底
    dataset = json.load(open(args.dataset, encoding="utf-8"))
    items = dataset if isinstance(dataset, list) else dataset.get("questions", [])
    if args.limit:
        items = items[: args.limit]
    print(f"[eval] 路由评测题数: {len(items)}")

    def one(item: dict) -> dict:
        t0 = time.time()
        try:
            res = asyncio.run(router.route(item["question"]))
            predicted_kb = res.kb
            source = res.source
        except Exception as e:  # noqa: BLE001
            return {**{k: item.get(k) for k in ("id", "kb", "question")}, "predicted_kb": "error",
                    "source": "error", "hit": False, "detail": str(e)[:80], "latency_s": round(time.time() - t0, 2)}
        hit = predicted_kb == item["kb"]
        return {**{k: item.get(k) for k in ("id", "kb", "question")},
                "predicted_kb": predicted_kb, "source": str(source), "hit": hit,
                "latency_s": round(time.time() - t0, 2)}

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, it) for it in items]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 20 == 0:
                hits = sum(1 for r in results if r["hit"])
                print(f"[eval] {i}/{len(items)}, 当前命中率 {hits/len(results):.1%}, 耗时 {time.time()-t0:.0f}s")

    n = len(results)
    hits = sum(1 for r in results if r["hit"])
    by_kb = {}
    for r in results:
        d = by_kb.setdefault(r["kb"], {"n": 0, "hit": 0})
        d["n"] += 1
        d["hit"] += r["hit"]
    by_source = {}
    for r in results:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    summary = {
        "sample_count": n,
        "routing_accuracy": round(hits / n, 4),
        "by_kb": {k: {"accuracy": round(v["hit"] / v["n"], 4), **v} for k, v in sorted(by_kb.items())},
        "by_source": by_source,
        "avg_latency_ms": round(sum(r["latency_s"] for r in results) / n * 1000, 1),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = os.path.join("data", "eval", "routing_full_summary.json")
    json.dump({"summary": summary, "misses": [r for r in results if not r["hit"]],
               "results": results}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"报告已写入 {out}")


if __name__ == "__main__":
    main()
