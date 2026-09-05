"""rerank 融合 alpha 调优 sweep — 攻 Recall@3。

CE 分数每题只算一次, alpha 网格仅做 numpy 重排 (快 5 倍)。
输出各配置的 R@1/3/5/10 + MRR, 给出最优 alpha。

用法: python scripts/tune_rerank_alpha.py [--eval-dir data/eval] [--limit 0]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np

import eval_retrieval as er  # noqa: E402
from engines.retrieval.dedup import adaptive_alpha  # noqa: E402
from engines.retrieval.reranker import Reranker  # noqa: E402

ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
KS = [1, 3, 5, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dataset = json.load(open(os.path.join(args.eval_dir, "eval_dataset.json"), encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]
    chunks = json.load(open(os.path.join(args.eval_dir, "corpus_chunks.json"), encoding="utf-8"))
    print(f"[tune] 题数 {len(dataset)}, 语料块 {len(chunks)}")

    embs_n, bm25 = er.build_index(chunks)
    from engines.embedding.embedder import EmbeddingService

    embedder = EmbeddingService()
    from api.state import resolve_model_path
    from api.config import settings as _settings
    reranker = Reranker(embedder=embedder, ce_model_name=resolve_model_path(_settings.reranker_model), max_length=_settings.reranker_max_length)
    ce_model = reranker._get_ce_model()

    # 逐题: 融合候选池 + CE 打分一次 (缓存)
    scored = []  # (question, expected_set, pool_docs, ce_scores, rrf_scores)
    for i, item in enumerate(dataset, 1):
        q = item["question"]
        q_vec = np.array(embedder.embed_query(q), dtype=np.float32)
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-9)
        vec_hits = er.vector_topk(q_vec, embs_n, chunks, 20)
        bm25_hits = bm25.search(q, top_k=20)
        fused = er.fuse(vec_hits, bm25_hits, method="rrf", w_bm25=0.6)
        pool = fused[:20]
        rrf_scores = {d.get("chunk_id") or d.get("id"): d.get("rrf", d.get("score", 0.0)) for d in pool}
        pairs = [(q, d.get("content", "")[:512]) for d in pool]
        ce_scores = ce_model.predict(pairs) if pairs else np.array([])
        expected = set(item.get("expected_chunk_ids") or [])
        scored.append((q, expected, pool, np.array([float(s) for s in ce_scores], dtype=np.float64), rrf_scores))
        if i % 100 == 0:
            print(f"[tune] CE 打分 {i}/{len(dataset)}")

    def rank_with_alpha(q, expected, pool, ce_arr, rrf_scores, alpha: float, adaptive: bool):
        if adaptive:
            embs_map = {c["chunk_id"]: c["embedding"] for c in chunks}
            pool_embs = [embs_map.get(d.get("chunk_id") or d.get("id")) for d in pool]
            pool_embs = [e for e in pool_embs if e is not None]
            alpha = adaptive_alpha([{"embedding": e} for e in pool_embs]) if pool_embs else alpha
        a = float(np.clip(alpha, 0.0, 1.0))
        cmin, cmax = float(ce_arr.min()), float(ce_arr.max())
        ce_norm = (ce_arr - cmin) / (cmax - cmin + 1e-9)
        rrf_arr = np.array([float(rrf_scores.get(d.get("chunk_id") or d.get("id"), 0.0)) for d in pool], dtype=np.float64)
        rmax = float(rrf_arr.max()) if rrf_arr.size else 1.0
        rrf_norm = rrf_arr / (rmax + 1e-9)
        final = (1.0 - a) * rrf_norm + a * ce_norm
        order = np.argsort(-final, kind="stable")
        ids = [(pool[int(i)].get("chunk_id") or pool[int(i)].get("id")) for i in order]
        metrics = {}
        for k in KS:
            top = set(ids[:k])
            metrics[f"recall@{k}"] = len(top & expected) / len(expected) if expected else 0.0
            metrics[f"hit@{k}"] = 1.0 if (top & expected) else 0.0
        hit_rank = next((r for r, cid in enumerate(ids, 1) if cid in expected), None)
        metrics["mrr"] = (1.0 / hit_rank) if hit_rank else 0.0
        return metrics

    configs = [(f"alpha={a}", a, False) for a in ALPHAS] + [("adaptive", None, True)]
    table = {name: {f"{m}@": 0.0 for m in []} for name, _, _ in configs}
    agg = {name: {f"{k}": [] for k in [f"recall@{k}" for k in KS] + [f"hit@{k}" for k in KS] + ["mrr"]} for name, _, _ in configs}
    per_alpha_hits = {name: [] for name, _, _ in configs}

    for q, expected, pool, ce_arr, rrf_scores in scored:
        for name, alpha, adaptive in configs:
            m = rank_with_alpha(q, expected, pool, ce_arr, rrf_scores, alpha, adaptive)
            for k, v in m.items():
                agg[name][k].append(v)

    print(f"\n{'配置':<12} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7}")
    best = None
    for name, _, _ in configs:
        a = agg[name]
        row = {k: sum(v) / len(v) for k, v in a.items()}
        print(f"{name:<12} {row['recall@1']:>7.4f} {row['recall@3']:>7.4f} {row['recall@5']:>7.4f} {row['recall@10']:>7.4f} {row['mrr']:>7.4f}")
        score = row["recall@3"] * 2 + row["recall@1"] + row["mrr"]  # R@3 双倍权重
        if best is None or score > best[1]:
            best = (name, score, row)

    print(f"\n最优配置: {best[0]} (R@3 加权分 {best[1]:.4f})")
    out = {"best": best[0], "best_metrics": best[2],
           "all": {name: {k: round(sum(v) / len(v), 4) for k, v in agg[name].items()} for name, _, _ in configs},
           "sample_count": len(dataset)}
    json.dump(out, open(os.path.join(args.eval_dir, "alpha_tuning_sweep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("已写入 data/eval/alpha_tuning_sweep.json")


if __name__ == "__main__":
    main()
