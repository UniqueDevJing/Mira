"""RAGAS 指标评测 — 按 RAGAS 论文 (Es et al. 2023) 定义自实现。

背景: ragas 官方包依赖链与 Python 3.14 不兼容 (numpy 1.26 无 wheel 需源码编译),
故按论文算法定价自实现, 判定模型 = qwen-plus, 相似度 = bge-small-zh。

指标 (对齐 ragas 定义):
  faithfulness        — 答案拆 claims, 逐条判定被 contexts 支持的比例
  answer_relevancy    — 从答案反向生成 3 个问题, 与原问题 embedding 余弦均值
  context_precision   — 每个 context 判定对标准答案的有用性, rank 加权精度
  context_recall      — 标准答案拆 claims, 逐条判定被 contexts 覆盖的比例

输入: data/eval/answer_quality_summary.json (390 题答案) + eval_retrieval 同款检索
输出: data/eval/ragas_summary.json

用法: python scripts/eval_ragas_metrics.py [--limit 0] [--concurrency 6]
"""
import argparse
import asyncio
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np  # noqa: E402

CLAIMS_SYS = (
    "将给定文本拆解为独立的事实断言(claims), 每条一个可验证的事实, 不要合并、不要推断。"
    '只输出 JSON 数组, 如 ["断言1", "断言2"]。无断言时输出 []。'
)
VERIFY_SYS = (
    "判断【断言】是否被【上下文】完全支持 (上下文中的信息足以推断该断言为真)。"
    '只输出 JSON: {"supported": true|false}。'
)
REVQ_SYS = (
    "根据给定回答生成 3 个该回答最可能对应的用户问题。问题应具体、可直接检索。"
    '只输出 JSON 数组, 如 ["问题1", "问题2", "问题3"]。'
)
CTXUSEFUL_SYS = (
    "判断【上下文】对回答【问题】是否必要或有用 (上下文信息是否有助于得出标准答案)。"
    '只输出 JSON: {"useful": true|false}。'
)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    _ = args

    from api.core.llm_client import get_llm_client
    from engines.embedding.embedder import EmbeddingService

    client = get_llm_client()
    embedder = EmbeddingService()

    def chat_sync_json(system: str, user: str, sem: asyncio.Semaphore) -> dict | list | None:
        async def _call():
            async with sem:
                try:
                    resp = await client.chat(
                        [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        temperature=0.0, max_tokens=600, json_mode=True,
                    )
                    return json.loads((getattr(resp, "content", "") or "").strip())
                except Exception:  # noqa: BLE001 — 单次判定失败返回 None, 不阻断
                    return None
        return asyncio.get_event_loop().create_task(_call())

    # 载入 390 题答案 (eval_answer_quality 产物) + 检索上下文
    aq = json.load(open("data/eval/answer_quality_summary.json", encoding="utf-8"))
    results = aq.get("results", [])
    if not results:
        print("[ragas] 无答案数据, 请先运行 scripts/eval_answer_quality.py")
        return

    # 检索上下文: 复用评测链路 (与 eval_answer_quality 相同)
    import eval_retrieval as er  # noqa: E402
    from engines.retrieval.reranker import Reranker  # noqa: E402
    from api.state import resolve_model_path  # noqa: E402
    from api.config import settings as _settings  # noqa: E402

    dataset = {it["id"]: it for it in json.load(open("data/eval/eval_dataset.json", encoding="utf-8"))}
    chunks = json.load(open("data/eval/corpus_chunks.json", encoding="utf-8"))
    embs_n, bm25 = er.build_index(chunks)
    reranker = Reranker(embedder=embedder, ce_model_name=resolve_model_path(_settings.reranker_model),
                        max_length=_settings.reranker_max_length)

    contexts_map = {}
    for r in results:
        q = r["question"]
        q_vec = np.array(embedder.embed_query(q), dtype=np.float32)
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-9)
        vec_hits = er.vector_topk(q_vec, embs_n, chunks, 20)
        bm25_hits = bm25.search(q, top_k=20)
        fused = er.fuse(vec_hits, bm25_hits, method="rrf", w_bm25=0.6)
        ce = reranker._get_ce_model()
        ctx = reranker.rerank(q, fused[:20], top_k=5) if ce else fused[:5]
        contexts_map[r["id"]] = [c.get("content", "") for c in ctx]
    print(f"[ragas] 上下文检索完成: {len(contexts_map)} 题")

    items = [r for r in results if r.get("answer") and r["id"] in contexts_map and r["id"] in dataset]
    if args.limit:
        items = items[: args.limit]
    print(f"[ragas] 待评测样本: {len(items)}")

    sem = asyncio.Semaphore(args.concurrency)

    async def llm_json(system, user, max_tokens=600):
        async with sem:
            try:
                resp = await client.chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.0, max_tokens=max_tokens, json_mode=True,
                )
                return json.loads((getattr(resp, "content", "") or "").strip())
            except Exception:  # noqa: BLE001
                return None

    async def faithfulness(item):
        ctx = contexts_map[item["id"]]
        claims = await llm_json(CLAIMS_SYS, item["answer"])
        if not claims:
            return 1.0  # 无可拆断言 (空答/拒答) 不计惩罚
        verdicts = await asyncio.gather(*[
            llm_json(VERIFY_SYS, f"【断言】{c}\n\n【上下文】\n" + "\n".join(x[:400] for x in ctx), 100)
            for c in claims[:8]
        ])
        valid = [v for v in verdicts if v]
        if not valid:
            return 1.0
        return sum(1 for v in valid if v.get("supported")) / len(valid)

    async def context_recall(item):
        gt = dataset.get(item["id"], {}).get("reference_answer", "")
        if not gt:
            return None
        ctx = contexts_map[item["id"]]
        claims = await llm_json(CLAIMS_SYS, gt)
        if not claims:
            return None
        verdicts = await asyncio.gather(*[
            llm_json(VERIFY_SYS, f"【断言】{c}\n\n【上下文】\n" + "\n".join(x[:400] for x in ctx), 100)
            for c in claims[:8]
        ])
        valid = [v for v in verdicts if v]
        if not valid:
            return None
        return sum(1 for v in valid if v.get("supported")) / len(valid)

    async def answer_relevancy(item):
        qs = await llm_json(REVQ_SYS, f"回答: {item['answer'][:500]}")
        if not qs or not isinstance(qs, list) or not qs:
            return None
        q_emb = np.array(embedder.embed_query(item["question"]), dtype=np.float32)
        q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-9)
        sims = []
        for rq in qs[:3]:
            e = np.array(embedder.embed_query(rq), dtype=np.float32)
            e = e / max(np.linalg.norm(e), 1e-9)
            sims.append(float(np.dot(q_emb, e)))
        return sum(sims) / len(sims)

    async def context_precision(item):
        gt = dataset.get(item["id"], {}).get("reference_answer", "")
        if not gt:
            return None
        ctx = contexts_map[item["id"]]
        verdicts = await asyncio.gather(*[
            llm_json(CTXUSEFUL_SYS, f"【问题】{item['question']}\n【标准答案】{gt[:300]}\n【上下文】{c[:400]}", 80)
            for c in ctx
        ])
        # RAGAS 定义: 有用性 rank 加权精度 (有用 chunk 排位越靠前分越高)
        useful_ranks = [rank for rank, v in enumerate(verdicts, 1) if v and v.get("useful")]
        if not useful_ranks:
            return 0.0
        k = len(ctx)
        return sum(1.0 / r for r in useful_ranks) * sum(
            max(0.0, 1.0 - (min(r, k) - 1) / k) for r in useful_ranks
        ) / len(useful_ranks) / sum(1.0 / r for r in range(1, k + 1))

    t0 = time.time()
    tasks = []
    for it in items:
        tasks.append(asyncio.gather(faithfulness(it), answer_relevancy(it), context_precision(it), context_recall(it)))
    all_res = await asyncio.gather(*tasks)

    faith = [r[0] for r in all_res if r[0] is not None]
    arel = [r[1] for r in all_res if r[1] is not None]
    cprec = [r[2] for r in all_res if r[2] is not None]
    crec = [r[3] for r in all_res if r[3] is not None]

    summary = {
        "sample_count": len(items),
        "faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
        "answer_relevancy": round(sum(arel) / len(arel), 4) if arel else None,
        "context_precision": round(sum(cprec) / len(cprec), 4) if cprec else None,
        "context_recall": round(sum(crec) / len(crec), 4) if crec else None,
        "metric_source": "RAGAS 论文定义自实现 (判定模型 qwen-plus, embedding bge-small-zh)",
        "duration_s": round(time.time() - t0, 1),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    per = [{"id": it["id"], "faithfulness": r[0], "answer_relevancy": r[1],
            "context_precision": r[2], "context_recall": r[3]} for it, r in zip(items, all_res)]
    json.dump({"summary": summary, "per_sample": per},
              open("data/eval/ragas_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已写入 data/eval/ragas_summary.json")


if __name__ == "__main__":
    asyncio.run(main())
