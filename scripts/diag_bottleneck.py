"""P1 瓶颈诊断 — 定位 context_precision 短板在检索候选池还是重排器。

不复用 LLM, 纯离线。对每个评测问题计算:
  - pool_coverage: golden 是否进入重排候选池 (RRF 融合序前 rerank_candidate_k 条)
  - baseline@5  : 不重排 (RRF 融合序 top5) 的 golden 命中
  - fused@5     : 生产路径 rerank_fused(CE+RRF, 自适应 alpha) top5 的 golden 命中
  - pure_ce@5   : 纯 Cross-Encoder rerank top5 的 golden 命中
  - n_golden    : 该问题 golden chunk 总数 (context_precision 的分母)

结论判读:
  pool_coverage 低  -> 检索层漏召 golden, 重排无从挽救 (瓶颈在检索)
  pool_coverage 高但 fused@5 低 -> 重排器把 golden 推下去了 (瓶颈在重排)
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from api.config import settings
from api.state import resolve_model_path
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index
from engines.retrieval.dedup import adaptive_alpha
from engines.retrieval.fusion import rrf_fuse
from engines.retrieval.reranker import Reranker


def build_index(chunks):
    embs = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.clip(norms, 1e-9, None)
    bm25 = Bm25Index()
    bm25.add_documents([{"id": c["chunk_id"], "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"]} for c in chunks])
    return embs_n, bm25


def _ids(order):
    return [d.get("chunk_id") or d.get("id") for d in order]


def _golden_in_top(order, expected, k):
    return len(set(_ids(order)[:k]) & expected)


def main():
    ap = __import__("argparse").ArgumentParser()
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
    reranker = Reranker(embedder=emb, ce_model_name=resolve_model_path(settings.reranker_model),
                        max_length=settings.reranker_max_length or None, backend=settings.reranker_backend)
    reranker.warmup()

    cap = settings.rerank_candidate_k
    pool_cov = []       # golden 是否进候选池
    base_hit5 = []      # 不重排 top5 命中数
    fused_hit5 = []     # 融合重排 top5 命中数
    pure_hit5 = []      # 纯 CE top5 命中数
    n_golden_total = []  # 每问 golden 数
    # 进入池但融合重排漏掉 golden 的样本 (真·重排器失误)
    pool_in_but_rerank_miss = 0

    for it in dataset:
        q = it["question"]
        expected = set(it["expected_chunk_ids"])
        n_golden_total.append(len(expected))
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)
        v = _vtopk(q_emb, embs_n, chunks, 40)
        b = bm25.search(q, 40)
        fused = rrf_fuse(v, b)

        pool = fused[:cap]
        in_pool = bool(set(_ids(pool)) & expected)
        pool_cov.append(1.0 if in_pool else 0.0)

        base_hit5.append(_golden_in_top(fused, expected, 5))

        rrf_map = {d.get("chunk_id") or d.get("id"): d.get("_rrf", 0.0) for d in fused}
        alpha = adaptive_alpha(pool, threshold=settings.rerank_density_threshold, mode=settings.rerank_density_mode,
                               alpha_max=settings.rerank_alpha_max, alpha_min=settings.rerank_alpha_min,
                               density_full=settings.rerank_density_full)
        fused_order = reranker.rerank_fused(q, pool, rrf_map, top_k=5, alpha=alpha)
        fused_hit5.append(_golden_in_top(fused_order, expected, 5))

        pure_order = reranker.rerank(q, pool, top_k=5)
        pure_hit5.append(_golden_in_top(pure_order, expected, 5))

        if in_pool and _golden_in_top(fused_order, expected, 5) < _golden_in_top(pool, expected, 5):
            pool_in_but_rerank_miss += 1

    n = len(dataset)
    avg = lambda xs: sum(xs) / n if n else 0.0
    # context_precision 近似 = 融合重排 top5 命中数 / min(5, golden数) 的均值
    cp = avg([min(5, g) and (h / min(5, g)) for h, g in zip(fused_hit5, n_golden_total)])
    print(f"=== P1 瓶颈诊断 (n={n}, rerank_candidate_k={cap}) ===")
    print(f"候选池覆盖 pool_coverage@k={cap} : {avg(pool_cov)*100:.1f}%  (golden 进重排候选池的比例)")
    print(f"不重排 baseline  top5 命中 golden 均值 : {avg(base_hit5):.3f}")
    print(f"融合重排 fused    top5 命中 golden 均值 : {avg(fused_hit5):.3f}")
    print(f"纯CE重排 pure_ce  top5 命中 golden 均值 : {avg(pure_hit5):.3f}")
    print(f"每问 golden 数均值 : {avg(n_golden_total):.2f}")
    print(f"context_precision 近似 : {cp:.3f}")
    print(f"进池却被融合重排推掉 golden 的样本数 : {pool_in_but_rerank_miss}")

    if avg(pool_cov) < 0.6:
        print("\n>>> 结论: 瓶颈在【检索层】— golden 常未进入候选池, 重排无法挽救。优先 P1 检索优化。")
    else:
        print("\n>>> 结论: 瓶颈在【重排器】— golden 已在池中却被推下。优先重排 α/融合策略调优。")


def _vtopk(q_emb, embs_n, chunks, k):
    sims = embs_n @ q_emb
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        c = chunks[int(i)]
        out.append({"chunk_id": c["chunk_id"], "id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"], "embedding": c["embedding"], "score": float(sims[i])})
    return out


if __name__ == "__main__":
    main()
