"""检索三路诊断 — 量化向量/BM25/融合各自贡献, 并测量嵌入 512 截断损失。

回答三个工程假设:
  1. 向量是不是短板? -> vector@512(线上真实) vs bm25 vs fused
  2. 嵌入 512 截断损失多大? -> vector@512(现有) vs vector@full(不截断重嵌)
  3. 混合检索补位够不够? -> bm25-only 是否显著拉高 fused 相对 vector-only

用法:
  python scripts/diag_retrieval_legs.py [--limit 0] [--chunks corpus_chunks.json]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index
from engines.retrieval.fusion import rrf_fuse


def _norm(embs):
    embs = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-9, None)


def vector_topk(q_emb, embs_n, chunks, k):
    sims = embs_n @ q_emb
    idx = np.argsort(-sims)[:k]
    return [chunks[int(i)]["chunk_id"] for i in idx]


def _metrics(order_ids, expected, ks):
    hit_rank = None
    for r, cid in enumerate(order_ids, 1):
        if cid in expected:
            hit_rank = r
            break
    res = {}
    for k in ks:
        top = set(order_ids[:k])
        res[f"recall@{k}"] = len(top & expected) / len(expected) if expected else 0.0
        res[f"hit@{k}"] = 1.0 if (top & expected) else 0.0
    res["mrr"] = (1.0 / hit_rank) if hit_rank else 0.0
    return res


def _avg(rows, ks):
    out = {}
    for k in ks:
        out[f"recall@{k}"] = round(np.mean([r[f"recall@{k}"] for r in rows]).item(), 4)
        out[f"hit@{k}"] = round(np.mean([r[f"hit@{k}"] for r in rows]).item(), 4)
    out["mrr"] = round(np.mean([r["mrr"] for r in rows]).item(), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--chunks", default="corpus_chunks.json")
    ap.add_argument("--dataset", default="eval_dataset.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    ks = [1, 3, 5, 10]

    with open(os.path.join(args.eval_dir, args.chunks), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(args.eval_dir, args.dataset), encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"[diag] chunks={len(chunks)} questions={len(dataset)}")

    # 现有(截断)向量索引
    emb_512 = _norm([c["embedding"] for c in chunks])
    # BM25
    bm25 = Bm25Index()
    bm25.add_documents([{"id": c["chunk_id"], "chunk_id": c["chunk_id"],
                        "doc_id": c["doc_id"], "content": c["content"]} for c in chunks])

    # 不截断重嵌(诊断截断损失) — 仅对 query 侧也用 full, 公平对比
    emb = EmbeddingService()
    t0 = time.time()
    full_texts = [c["content"] for c in chunks]
    emb_full = _norm(emb.embed_batch(full_texts, max_length=0))
    print(f"[diag] 重嵌全文向量 {len(chunks)} 块耗时 {time.time()-t0:.1f}s")

    rows_v512, rows_vfull, rows_bm25, rows_fused = [], [], [], []
    for it in dataset:
        q = it["question"]
        expected = set(it["expected_chunk_ids"])
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)        # query 走默认 512(线上)
        q_emb_full = np.array(emb.embed_query(q, max_length=0), dtype=np.float32)
        v_docs = vector_topk(q_emb, emb_512, chunks, 40)
        vfull_docs = vector_topk(q_emb_full, emb_full, chunks, 40)
        b_docs = [d["chunk_id"] for d in bm25.search(q, 40)]
        fused = [d.get("chunk_id") or d.get("id") for d in rrf_fuse(
            [{"chunk_id": c} for c in v_docs],
            [{"chunk_id": c} for c in b_docs])]
        rows_v512.append(_metrics(v_docs, expected, ks))
        rows_vfull.append(_metrics(vfull_docs, expected, ks))
        rows_bm25.append(_metrics(b_docs, expected, ks))
        rows_fused.append(_metrics(fused, expected, ks))

    av, avf, ab, af = _avg(rows_v512, ks), _avg(rows_vfull, ks), _avg(rows_bm25, ks), _avg(rows_fused, ks)
    print("\n=== 检索三路诊断 (n=%d) ===" % len(dataset))
    print(f"{'指标':<10} {'向量@512':>9} {'向量@全文':>9} {'BM25':>9} {'融合':>9}")
    for k in ks:
        print(f"{'Recall@'+str(k):<10} {av[f'recall@{k}']*100:>8.1f}% {avf[f'recall@{k}']*100:>8.1f}% {ab[f'recall@{k}']*100:>8.1f}% {af[f'recall@{k}']*100:>8.1f}%")
    for k in ks:
        print(f"{'Hit@'+str(k):<10} {av[f'hit@{k}']*100:>8.1f}% {avf[f'hit@{k}']*100:>8.1f}% {ab[f'hit@{k}']*100:>8.1f}% {af[f'hit@{k}']*100:>8.1f}%")
    print(f"{'MRR':<10} {av['mrr']:>9.3f} {avf['mrr']:>9.3f} {ab['mrr']:>9.3f} {af['mrr']:>9.3f}")

    print("\n=== 关键结论 ===")
    d5 = (avf["recall@5"] - av["recall@5"]) * 100
    print(f"嵌入 512 截断损失 (向量@全文 - 向量@512, Recall@5): {d5:+.1f}pp")
    print(f"BM25 相对向量@512 的提升 (Recall@5): {(ab['recall@5']-av['recall@5'])*100:+.1f}pp")
    print(f"融合相对向量@512 的提升 (Recall@5): {(af['recall@5']-av['recall@5'])*100:+.1f}pp")


if __name__ == "__main__":
    main()
