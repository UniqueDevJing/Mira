"""P1-1 Rerank 融合 α 调优 — 找使 top5 golden 覆盖最优的 α。

关键优化: CE 分数每问只算一次, α 网格扫描是纯 numpy 线性组合 (final=(1-α)·rrf + α·ce),
因此可在全量 390 问上秒级扫完多个 α, 无需重复 CE 推理。

指标:
  hit5   = top5 中 golden 命中数均值 (越高越好, 直接决定 context_precision)
  cp5    = hit5 / 5  (与 QA 评测口径一致的 context_precision)
同时报告: 自适应 α(当前生产) / 纯 RRF 基线 / 纯 CE。
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

from api.config import settings
from api.state import resolve_model_path
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index
from engines.retrieval.fusion import rrf_fuse
from engines.retrieval.reranker import Reranker


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
        out.append({"chunk_id": c["chunk_id"], "id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"], "embedding": c["embedding"]})
    return out


def _ids(order):
    return [d.get("chunk_id") or d.get("id") for d in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha-grid", default="0.0,0.2,0.4,0.6,0.8,1.0")
    args = ap.parse_args()
    grid = [float(x) for x in args.alpha_grid.split(",")]

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
    ce = reranker._get_ce_model()
    cap = settings.rerank_candidate_k

    # 预计算每问: rrf(归一) + ce(归一) + expected
    rrf_list, ce_list, exp_list = [], [], []
    t0 = time.time()
    for it in dataset:
        q = it["question"]
        expected = set(it["expected_chunk_ids"])
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)
        v = _vtopk(q_emb, embs_n, chunks, 40)
        b = bm25.search(q, 40)
        fused = rrf_fuse(v, b)
        pool = fused[:cap]
        rrf_map = {d.get("chunk_id") or d.get("id"): d.get("_rrf", 0.0) for d in fused}
        pairs = [(q, d.get("content", "")[:512]) for d in pool]
        ce_scores = np.array([float(s) for s in ce.predict(pairs)], dtype=np.float64)
        cmin, cmax = ce_scores.min(), ce_scores.max()
        ce_norm = (ce_scores - cmin) / (cmax - cmin + 1e-9)
        rrf_arr = np.array([rrf_map.get(d.get("chunk_id") or d.get("id"), 0.0) for d in pool], dtype=np.float64)
        rmax = rrf_arr.max() if rrf_arr.size else 1.0
        rrf_norm = rrf_arr / (rmax + 1e-9)
        rrf_list.append(rrf_norm)
        ce_list.append(ce_norm)
        exp_list.append(expected)
    print(f"[tune] 预计算 {len(dataset)} 问 耗时 {time.time()-t0:.1f}s")

    # 重建 pool id 列表 (与上面 pool 顺序一致) 供 α 扫描时定位 golden
    pool_ids_list = []
    for it in dataset:
        q = it["question"]
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)
        v = _vtopk(q_emb, embs_n, chunks, 40)
        b = bm25.search(q, 40)
        fused = rrf_fuse(v, b)
        pool = fused[:cap]
        pool_ids_list.append([d.get("chunk_id") or d.get("id") for d in pool])

    agg = {a: [] for a in grid}
    for a in grid:
        for i, expected in enumerate(exp_list):
            final = (1.0 - a) * rrf_list[i] + a * ce_list[i]
            order = np.argsort(-final, kind="stable")[:5]
            ids = [pool_ids_list[i][j] for j in order]
            agg[a].append(len(set(ids) & expected))

    n = len(dataset)
    print(f"\n=== Rerank 融合 α 调优 (n={n}, 候选池 k={cap}) ===")
    print(f"{'α':<8} {'hit5':>8} {'cp5=hit5/5':>12}")
    best_a, best_hit = None, -1
    for a in grid:
        hit = sum(agg[a]) / n
        cp5 = hit / 5
        print(f"{a:<8.1f} {hit:>8.3f} {cp5:>12.3f}")
        if hit > best_hit:
            best_hit, best_a = hit, a
    # 基线
    base_hit = sum(len(set(pool_ids_list[i][:5]) & exp_list[i]) for i in range(n)) / n
    print(f"{'RRF基线':<8} {base_hit:>8.3f} {base_hit/5:>12.3f}")
    print(f"\n>>> 最佳固定 α={best_a} (hit5={best_hit:.3f}, 较 RRF 基线 {'+' if best_hit>=base_hit else ''}{(best_hit-base_hit)*100:.2f}%)")
    if best_hit > base_hit:
        print(">>> 建议: 将该 α 设为 rerank_fusion_alpha 固定值(关闭自适应) 或重新校准自适应区间。")
    else:
        print(">>> 固定 α 未超越 RRF 基线 — 瓶颈在 CE 排序质量, 需 query 改写 / 更大候选池 / 更强 CE。")


if __name__ == "__main__":
    main()
