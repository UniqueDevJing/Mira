"""rerank 候选深度 sweep — 攻 context_precision / R@1。

思路: 检索融合池放大到 25, CE 对 25 候选一次性打分并缓存到磁盘;
对 candidate_k ∈ {5,10,15,20,25} 各自取前 k 候选做 alpha=0.9 融合重排,
统计 R@1/3/5/10/MRR, 找出 context_precision 的最优候选深度。

用法: python scripts/tune_rerank_depth.py [--limit 0] [--use-cache]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np  # noqa: E402

import eval_retrieval as er  # noqa: E402
from engines.retrieval.reranker import Reranker  # noqa: E402
from api.state import resolve_model_path  # noqa: E402
from api.config import settings  # noqa: E402

KS = [1, 3, 5, 10]
CANDIDATE_KS = [5, 10, 15, 20, 25]
ALPHA = 0.9  # 上一轮 alpha sweep 确认的最优值
CACHE = "data/eval/_ce_cache_25.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    dataset = json.load(open("data/eval/eval_dataset.json", encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]
    chunks = json.load(open("data/eval/corpus_chunks.json", encoding="utf-8"))
    print(f"[tune] 题数 {len(dataset)}")

    from engines.embedding.embedder import EmbeddingService

    embedder = EmbeddingService()
    embs_n, bm25 = er.build_index(chunks)
    reranker = Reranker(embedder=embedder, ce_model_name=resolve_model_path(settings.reranker_model),
                        max_length=settings.reranker_max_length)
    ce_model = reranker._get_ce_model()

    cache = {}
    if args.use_cache and os.path.exists(CACHE):
        cache = json.load(open(CACHE, encoding="utf-8"))
        print(f"[tune] 缓存命中 {len(cache)} 题")

    scored = []
    for i, item in enumerate(dataset, 1):
        q = item["question"]
        qid = item["id"]
        q_vec = np.array(embedder.embed_query(q), dtype=np.float32)
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-9)
        vec_hits = er.vector_topk(q_vec, embs_n, chunks, 25)
        bm25_hits = bm25.search(q, top_k=25)
        fused = er.fuse(vec_hits, bm25_hits, method="rrf", w_bm25=0.6)
        pool = fused[:25]
        if qid in cache:
            ce_scores = np.array(cache[qid], dtype=np.float64)
        else:
            pairs = [(q, d.get("content", "")[:settings.reranker_max_length]) for d in pool]
            ce_scores = np.array([float(s) for s in ce_model.predict(pairs)], dtype=np.float64)
            cache[qid] = ce_scores.tolist()
        rrf_scores = {d.get("chunk_id") or d.get("id"): d.get("rrf", d.get("score", 0.0)) for d in pool}
        expected = set(item.get("expected_chunk_ids") or [])
        scored.append((q, expected, pool, ce_scores, rrf_scores))
        if i % 100 == 0:
            print(f"[tune] CE 打分 {i}/{len(dataset)}")

    json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    print("[tune] CE 缓存已写入", CACHE)

    def metrics_for_k(pool, ce_arr, rrf_scores, expected, k):
        sub_pool = pool[:k]
        sub_ce = ce_arr[:k]
        sub_rrf = {d.get("chunk_id") or d.get("id"): rrf_scores.get(d.get("chunk_id") or d.get("id"), 0.0) for d in sub_pool}
        cmin, cmax = float(sub_ce.min()), float(sub_ce.max())
        ce_norm = (sub_ce - cmin) / (cmax - cmin + 1e-9)
        rrf_arr = np.array([float(sub_rrf.get(d.get("chunk_id") or d.get("id"), 0.0)) for d in sub_pool], dtype=np.float64)
        rmax = float(rrf_arr.max()) if rrf_arr.size else 1.0
        rrf_norm = rrf_arr / (rmax + 1e-9)
        final = (1.0 - ALPHA) * rrf_norm + ALPHA * ce_norm
        order = np.argsort(-final, kind="stable")
        ids = [(sub_pool[int(i)].get("chunk_id") or sub_pool[int(i)].get("id")) for i in order]
        m = {}
        for kk in KS:
            top = set(ids[:kk])
            m[f"recall@{kk}"] = len(top & expected) / len(expected) if expected else 0.0
        hit_rank = next((r for r, cid in enumerate(ids, 1) if cid in expected), None)
        m["mrr"] = (1.0 / hit_rank) if hit_rank else 0.0
        return m

    agg = {k: {f"recall@{kk}": [] for kk in KS} | {"mrr": []} for k in CANDIDATE_KS}
    for _, expected, pool, ce_arr, rrf_scores in scored:
        for k in CANDIDATE_KS:
            m = metrics_for_k(pool, ce_arr, rrf_scores, expected, k)
            for key, v in m.items():
                agg[k][key].append(v)

    print(f"\n{'candidate_k':<12} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7}")
    out_all = {}
    for k in CANDIDATE_KS:
        a = agg[k]
        row = {kk: round(sum(v) / len(v), 4) for kk, v in a.items()}
        out_all[k] = row
        print(f"{k:<12} {row['recall@1']:>7.4f} {row['recall@3']:>7.4f} {row['recall@5']:>7.4f} {row['recall@10']:>7.4f} {row['mrr']:>7.4f}")

    json.dump({"alpha": ALPHA, "candidate_ks": CANDIDATE_KS, "metrics": out_all,
               "sample_count": len(dataset)}, open("data/eval/depth_sweep.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("已写入 data/eval/depth_sweep.json")


if __name__ == "__main__":
    main()
