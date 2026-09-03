"""P1-1a RRF_K 扫描 — 不重排(RRF 融合)的 top5 golden 覆盖对 RRF_K 的敏感度。

纯离线(无 CE), 向量+BM25 各取 40 候选, 用不同 RRF_K 做 rrf_fuse, 测 top5 golden 命中。
若某 RRF_K 显著优于当前 30, 则属零成本配置优化(改 RAG_RRF_K 环境变量)。

同时顺带测 "向量相似度加权融合" 替代纯 RRF: 把向量余弦分也并入融合,
看能否把 RRF 第 6-10 位的 golden 抬进 top5。
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
from engines.retrieval.fusion import rrf_fuse


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
        out.append({"chunk_id": c["chunk_id"], "id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"], "score": float(sims[i])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.eval_dir, "corpus_chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(args.eval_dir, "eval_dataset.json"), encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[:args.limit]

    embs_n, bm25 = build_index(chunks)
    emb = EmbeddingService()

    # 预计算向量/BM25 候选 (带余弦分)
    V, B, EXP = [], [], []
    for it in dataset:
        q = it["question"]
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)
        v = _vtopk(q_emb, embs_n, chunks, 40)
        b = bm25.search(q, 40)
        V.append(v); B.append(b); EXP.append(set(it["expected_chunk_ids"]))

    def hit5(order, expected):
        ids = [d.get("chunk_id") or d.get("id") for d in order[:5]]
        return len(set(ids) & expected)

    n = len(dataset)
    print(f"=== RRF_K 扫描 (n={n}, 向量/BM25 各 40 候选) ===")
    print(f"{'RRF_K':<8} {'hit5':>8} {'cp5':>8}")
    for K in [10, 20, 30, 50, 60]:
        tot = sum(hit5(rrf_fuse(V[i], B[i], k=K), EXP[i]) for i in range(n))
        print(f"{K:<8} {tot/n:>8.3f} {tot/n/5:>8.3f}")

    # 向量余弦加权融合 (替代纯 RRF): score = rrf + w*cos_norm
    print("\n=== 向量余弦加权融合 (w=权重) ===")
    print(f"{'w':<8} {'hit5':>8} {'cp5':>8}")
    K = 30
    for w in [0.0, 0.3, 0.5, 1.0, 2.0]:
        tot = 0
        for i in range(n):
            fused = rrf_fuse(V[i], B[i], k=K)
            # 取融合序, 叠加向量余弦
            cos_map = {d.get("chunk_id") or d.get("id"): d.get("score", 0.0) for d in V[i]}
            scored = []
            for rank, d in enumerate(fused):
                key = d.get("chunk_id") or d.get("id")
                rrf_s = 1.0 / (K + rank + 1)
                cos_s = cos_map.get(key, 0.0)
                scored.append((key, rrf_s + w * cos_s))
            scored.sort(key=lambda x: x[1], reverse=True)
            ids = [s[0] for s in scored[:5]]
            tot += len(set(ids) & EXP[i])
        print(f"{w:<8.1f} {tot/n:>8.3f} {tot/n/5:>8.3f}")


if __name__ == "__main__":
    main()
