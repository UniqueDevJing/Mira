"""rerank CE 输入长度 sweep — 256 vs 384 (candidate_k=10, alpha=0.9 固定)。

CE 对相同候选池用不同 max_length 截断重新打分, 对比排序质量 (R@1/3/10/MRR)。
缓存按 max_length 区分 key。

用法: python scripts/tune_rerank_maxlen.py [--limit 0]
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
MAXLENS = [256, 384]
ALPHA = 0.9
CANDIDATE_K = 10
CACHE = "data/eval/_ce_cache_maxlen.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dataset = json.load(open("data/eval/eval_dataset.json", encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]
    chunks = json.load(open("data/eval/corpus_chunks.json", encoding="utf-8"))

    from engines.embedding.embedder import EmbeddingService

    embedder = EmbeddingService()
    embs_n, bm25 = er.build_index(chunks)
    reranker = Reranker(embedder=embedder, ce_model_name=resolve_model_path(settings.reranker_model),
                        max_length=max(MAXLENS))
    ce_model = reranker._get_ce_model()

    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE, encoding="utf-8"))

    # 每题检索一次 (candidate_k=10), CE 按 max_length 分别打分
    pool_by_qid = {}
    rrf_by_qid = {}
    expected_by_qid = {}
    for item in dataset:
        q = item["question"]
        q_vec = np.array(embedder.embed_query(q), dtype=np.float32)
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-9)
        vec_hits = er.vector_topk(q_vec, embs_n, chunks, CANDIDATE_K)
        bm25_hits = bm25.search(q, top_k=CANDIDATE_K)
        fused = er.fuse(vec_hits, bm25_hits, method="rrf", w_bm25=0.6)[:CANDIDATE_K]
        pool_by_qid[item["id"]] = fused
        rrf_by_qid[item["id"]] = {d.get("chunk_id") or d.get("id"): d.get("rrf", d.get("score", 0.0)) for d in fused}
        expected_by_qid[item["id"]] = set(item.get("expected_chunk_ids") or [])

    ce_by_ml = {ml: {} for ml in MAXLENS}
    for ml in MAXLENS:
        key_ml = str(ml)
        cache.setdefault(key_ml, {})
        for item in dataset:
            qid = item["id"]
            if qid in cache[key_ml]:
                ce_by_ml[ml][qid] = np.array(cache[key_ml][qid], dtype=np.float64)
                continue
            pool = pool_by_qid[qid]
            pairs = [(item["question"], d.get("content", "")[:ml]) for d in pool]
            ce_by_ml[ml][qid] = np.array([float(s) for s in ce_model.predict(pairs)], dtype=np.float64)
            cache[key_ml][qid] = ce_by_ml[ml][qid].tolist()
        print(f"[tune] max_length={ml} CE 打分完成")
    json.dump(cache, open(CACHE, "w", encoding="utf-8"))

    def metrics(qid, ml):
        pool = pool_by_qid[qid]
        ce_arr = ce_by_ml[ml][qid]
        rrf_scores = rrf_by_qid[qid]
        expected = expected_by_qid[qid]
        cmin, cmax = float(ce_arr.min()), float(ce_arr.max())
        ce_norm = (ce_arr - cmin) / (cmax - cmin + 1e-9)
        rrf_arr = np.array([float(rrf_scores.get(d.get("chunk_id") or d.get("id"), 0.0)) for d in pool], dtype=np.float64)
        rmax = float(rrf_arr.max()) if rrf_arr.size else 1.0
        rrf_norm = rrf_arr / (rmax + 1e-9)
        final = (1.0 - ALPHA) * rrf_norm + ALPHA * ce_norm
        order = np.argsort(-final, kind="stable")
        ids = [(pool[int(i)].get("chunk_id") or pool[int(i)].get("id")) for i in order]
        m = {}
        for k in KS:
            top = set(ids[:k])
            m[f"recall@{k}"] = len(top & expected) / len(expected) if expected else 0.0
        hr = next((r for r, cid in enumerate(ids, 1) if cid in expected), None)
        m["mrr"] = (1.0 / hr) if hr else 0.0
        return m

    print(f"\n{'max_length':<12} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7}")
    out_all = {}
    for ml in MAXLENS:
        agg = {f"recall@{k}": [] for k in KS} | {"mrr": []}
        for item in dataset:
            m = metrics(item["id"], ml)
            for k, v in m.items():
                agg[k].append(v)
        row = {k: round(sum(v) / len(v), 4) for k, v in agg.items()}
        out_all[ml] = row
        print(f"{ml:<12} {row['recall@1']:>7.4f} {row['recall@3']:>7.4f} {row['recall@5']:>7.4f} {row['recall@10']:>7.4f} {row['mrr']:>7.4f}")

    json.dump({"alpha": ALPHA, "candidate_k": CANDIDATE_K, "maxlens": MAXLENS, "metrics": out_all,
               "sample_count": len(dataset)}, open("data/eval/maxlen_sweep.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("已写入 data/eval/maxlen_sweep.json")


if __name__ == "__main__":
    main()
