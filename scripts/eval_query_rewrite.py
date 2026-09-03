"""P1-1 离线验证 — LLM 查询改写能否提升检索 top5 golden 覆盖。

对评测集每问: 用 DeepSeek 改写查询(并发) -> 规则预处理 -> 向量+BM25+RRF 融合 -> top5 golden 命中。
同时算基线(原查询规则预处理)。对比 hit5 / cp5, 验证改写是否突破 0.318 天花板。

用法:
  python scripts/eval_query_rewrite.py [--limit 0] [--concurrency 8]
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

import numpy as np

from api.config import settings
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index
from engines.retrieval.fusion import rrf_fuse
from engines.retrieval.query_preprocessor import preprocess_query
from engines.retrieval.retrieval_query_rewriter import RetrievalQueryRewriter as QueryRewriter


def build_index(chunks):
    embs = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.clip(norms, 1e-9, None)
    bm25 = Bm25Index()
    bm25.add_documents([{"id": c["chunk_id"], "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"]} for c in chunks])
    return embs_n, bm25


def _vtopk(q_emb, embs_n, chunks, k):
    sims = embs_n @ q_emb
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        c = chunks[int(i)]
        out.append({"chunk_id": c["chunk_id"], "id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"]})
    return out


def _ids(order):
    return [d.get("chunk_id") or d.get("id") for d in order]


async def _rewrite_one(rewriter, question, sem):
    async with sem:
        return await rewriter.rewrite(question)


async def main(chunks: list, dataset: list, concurrency: int = 8):
    embs_n, bm25 = build_index(chunks)
    emb = EmbeddingService()
    rewriter = QueryRewriter(enabled=True, timeout_s=settings.query_rewrite_timeout_s)

    questions = [it["question"] for it in dataset]
    expected = [set(it["expected_chunk_ids"]) for it in dataset]

    # 1) 并发改写
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    rewritten = await asyncio.gather(*[_rewrite_one(rewriter, q, sem) for q in questions])
    print(f"[qrewrite] 改写 {len(questions)} 问 耗时 {time.time()-t0:.1f}s")

    base_hit, rw_hit = [], []
    rw_examples = []
    for q, rq, exp in zip(questions, rewritten, expected):
        # 基线: 原查询规则预处理
        vq_b, bq_b = preprocess_query(q)
        vb = np.array(emb.embed_query(vq_b), dtype=np.float32)
        fused_b = rrf_fuse(_vtopk(vb, embs_n, chunks, 40), bm25.search(bq_b, 40))
        base_hit.append(len(set(_ids(fused_b)[:5]) & exp))

        # 改写: 改写查询规则预处理
        vq_r, bq_r = preprocess_query(rq)
        vr = np.array(emb.embed_query(vq_r), dtype=np.float32)
        fused_r = rrf_fuse(_vtopk(vr, embs_n, chunks, 40), bm25.search(bq_r, 40))
        rw_hit.append(len(set(_ids(fused_r)[:5]) & exp))
        if rq != q and len(rw_examples) < 8:
            rw_examples.append((q, rq, base_hit[-1], rw_hit[-1]))

    n = len(dataset)
    b = sum(base_hit) / n
    r = sum(rw_hit) / n
    print(f"\n=== 查询改写检索验证 (n={n}) ===")
    print(f"基线(原查询)   hit5={b:.3f}  cp5={b/5:.3f}")
    print(f"改写后         hit5={r:.3f}  cp5={r/5:.3f}")
    print(f"提升           {(r-b)*100:+.2f}% hit5   ({(r-b)/5*100:+.2f}% cp5)")
    if r > b:
        print(">>> 改写有效, 建议开启 query_rewrite_enabled=True (注意检索延迟 +~0.3-0.8s)。")
    else:
        print(">>> 改写未提升, 保持关闭。可尝试调整改写 prompt / 模型。")
    print("\n--- 改写样例 (原 -> 改写 | 基线hit5/改写hit5) ---")
    for q, rq, bh, rh in rw_examples:
        print(f"  · {q}\n    -> {rq}   [{bh}/{rh}]")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    with open(os.path.join(a.eval_dir, "corpus_chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(a.eval_dir, "eval_dataset.json"), encoding="utf-8") as f:
        dataset = json.load(f)
    if a.limit:
        dataset = dataset[: a.limit]
    asyncio.run(main(chunks, dataset, a.concurrency))
