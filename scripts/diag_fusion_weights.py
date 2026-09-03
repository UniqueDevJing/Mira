"""融合权重扫描 — RRF(排名融合) vs 分数插值融合, 测试 BM25 主导权重能否拉回 BM25 单跑的召回。

用法: python scripts/diag_fusion_weights.py [--limit 0]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--chunks", default="corpus_chunks.json")
    ap.add_argument("--dataset", default="eval_dataset.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    ks = [1, 3, 5, 10]

    chunks = json.load(open(os.path.join(args.eval_dir, args.chunks), encoding="utf-8"))
    dataset = json.load(open(os.path.join(args.eval_dir, args.dataset), encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]
    cid_to_i = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    emb = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_n = emb / np.clip(n, 1e-9, None)
    bm25 = Bm25Index()
    bm25.add_documents([{"id": c["chunk_id"], "chunk_id": c["chunk_id"],
                        "doc_id": c["doc_id"], "content": c["content"]} for c in chunks])

    svc = EmbeddingService()

    def _metrics(order_ids, expected):
        hit_rank = None
        for r, c in enumerate(order_ids, 1):
            if c in expected:
                hit_rank = r
                break
        res = {}
        for k in ks:
            res[f"r@{k}"] = len(set(order_ids[:k]) & expected) / len(expected) if expected else 0.0
            res[f"h@{k}"] = 1.0 if (set(order_ids[:k]) & expected) else 0.0
        res["mrr"] = 1.0 / hit_rank if hit_rank else 0.0
        return res

    def _avg(rows):
        return {m: round(np.mean([r[m] for r in rows]).item(), 4)
                for m in ["r@1", "r@3", "r@5", "r@10", "h@5", "mrr"]}

    print(f"[sweep] chunks={len(chunks)} questions={len(dataset)}  (向量余弦 + BM25 分数插值)")
    print(f"{'权重w(BM25)':<12} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'H@5':>8} {'MRR':>7}")
    for w in [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]:
        rows = []
        for it in dataset:
            q = it["question"]
            expected = set(it["expected_chunk_ids"])
            qe = np.array(svc.embed_query(q), dtype=np.float32)
            cos = emb_n @ qe
            # BM25 raw 0-1
            bscores = {d["chunk_id"]: d["score"] for d in bm25.search(q, 40)}
            # 候选并集
            cand = set(np.argsort(-cos)[:40].tolist()) | set(cid_to_i[c] for c in bscores)
            scored = {}
            for i in cand:
                cid = chunks[i]["chunk_id"]
                b = bscores.get(cid, 0.0)
                v = (cos[i] + 1.0) / 2.0  # 余弦归一 0-1
                scored[cid] = w * b + (1 - w) * v
            order = sorted(scored, key=scored.get, reverse=True)
            rows.append(_metrics(order, expected))
        a = _avg(rows)
        print(f"{w:<12} {a['r@1']*100:>7.1f}% {a['r@3']*100:>7.1f}% {a['r@5']*100:>7.1f}% "
              f"{a['r@10']*100:>7.1f}% {a['h@5']*100:>7.1f}% {a['mrr']:>7.3f}")


if __name__ == "__main__":
    main()
