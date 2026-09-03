"""P2 本地改写验证 — 用本地资源(embedding + BM25 索引)做伪相关反馈(PRF)式查询改写。

完全离线, 不依赖外部 LLM key (DeepSeek 当前 402 余额耗尽)。
验证假设: "改写查询能否把 golden 推进 top5" —— 与 P1-1 的 LLM 改写是同一假设,
只是把 LLM 换成离线可跑的检索增强改写(PRF)。待 LLM key 恢复后再验 LLM 版。

用法:
  python scripts/eval_query_rewrite_local.py [--limit 0] [--n-terms 5]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jieba
import numpy as np

from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index
from engines.retrieval.fusion import rrf_fuse
from engines.retrieval.query_preprocessor import preprocess_query

STOP = set(
    ["的", "了", "和", "与", "及", "在", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那", "这个", "那个", "这些", "那些", "一个", "一些", "这种", "哪种", "哪些", "如何", "怎么", "什么", "多少", "几", "哪", "何时", "哪里", "为什么", "因为", "所以", "但是", "并且", "以及", "等", "该", "各", "个", "名", "位", "项", "次", "种", "类", "中", "上", "下", "内", "外", "前", "后", "时", "日", "月", "年"]
)


def build_index(chunks):
    embs = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.clip(norms, 1e-9, None)
    bm25 = Bm25Index()
    bm25.add_documents(
        [
            {"id": c["chunk_id"], "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"]}
            for c in chunks
        ]
    )
    return embs_n, bm25


def _vtopk(q_emb, embs_n, chunks, k):
    sims = embs_n @ q_emb
    idx = np.argsort(-sims)[:k]
    return [chunks[int(i)] for i in idx]


def _ids(order):
    return [d.get("chunk_id") or d.get("id") for d in order]


def prf_rewrite(q, emb, embs_n, chunks, bm25, n_terms=5):
    """伪相关反馈: 用原查询检索 top 文档, 取高权重术语回扩查询。"""
    vq, bq = preprocess_query(q)
    q_emb = np.array(emb.embed_query(vq), dtype=np.float32)
    vec = _vtopk(q_emb, embs_n, chunks, 12)
    bm = bm25.search(bq, 12)
    cand = [(r, c) for r, c in enumerate(vec[:8])]
    seen = set()
    for r, d in enumerate(bm[:8]):
        cid = d.get("chunk_id")
        if cid in seen:
            continue
        seen.add(cid)
        c = next((x for x in chunks if x["chunk_id"] == cid), None)
        if c:
            cand.append((r, c))
    term_scores = {}
    for rank, c in cand:
        w = 1.0 / (rank + 1)
        for t in jieba.cut(c["content"]):
            t = t.strip()
            if len(t) >= 2 and t not in STOP and not t.isdigit():
                term_scores[t] = term_scores.get(t, 0.0) + w
    top = sorted(term_scores.items(), key=lambda kv: kv[1], reverse=True)[:n_terms]
    terms = [t for t, _ in top]
    return q + " " + " ".join(terms), terms


def main(chunks, dataset, limit=0, n_terms=5):
    if limit:
        dataset = dataset[:limit]
    embs_n, bm25 = build_index(chunks)
    emb = EmbeddingService()
    base_hit, rw_hit = [], []
    ex = []
    t0 = time.time()
    for it in dataset:
        q = it["question"]
        exp = set(it["expected_chunk_ids"])
        vq, bq = preprocess_query(q)
        q_emb = np.array(emb.embed_query(vq), dtype=np.float32)
        fused_b = rrf_fuse(_vtopk(q_emb, embs_n, chunks, 40), bm25.search(bq, 40))
        base_hit.append(len(set(_ids(fused_b)[:5]) & exp))

        rq, terms = prf_rewrite(q, emb, embs_n, chunks, bm25, n_terms)
        vq2, bq2 = preprocess_query(rq)
        q_emb2 = np.array(emb.embed_query(vq2), dtype=np.float32)
        fused_r = rrf_fuse(_vtopk(q_emb2, embs_n, chunks, 40), bm25.search(bq2, 40))
        rw = len(set(_ids(fused_r)[:5]) & exp)
        rw_hit.append(rw)
        if len(ex) < 8:
            ex.append((q, rq, base_hit[-1], rw, terms))
    n = len(dataset)
    b = sum(base_hit) / n
    r = sum(rw_hit) / n
    print(f"\n=== 本地 PRF 查询改写验证 (n={n}, 耗时 {time.time() - t0:.1f}s) ===")
    print(f"基线(原查询)   hit5={b:.3f}  cp5={b / 5:.3f}")
    print(f"PRF改写后      hit5={r:.3f}  cp5={r / 5:.3f}")
    print(f"提升           {(r - b) * 100:+.2f}% hit5   ({(r - b) / 5 * 100:+.2f}% cp5)")
    print(
        ">>> "
        + (
            "PRF 改写有效, 查询改写假设成立 — 待 LLM key 恢复后验 LLM 版以确认上限"
            if r > b
            else "PRF 改写未提升 — 需换更强嵌入模型或 LLM 改写"
        )
    )
    print("\n--- 样例 (原 -> PRF改写 | 基线/改写 | 追加词) ---")
    for q, rq, bh, rh, terms in ex:
        print(f"  · {q}\n    -> {rq}   [{bh}/{rh}]  +{terms}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-terms", type=int, default=5)
    a = ap.parse_args()
    chunks = json.load(open(os.path.join(a.eval_dir, "corpus_chunks.json"), encoding="utf-8"))
    dataset = json.load(open(os.path.join(a.eval_dir, "eval_dataset.json"), encoding="utf-8"))
    main(chunks, dataset, a.limit, a.n_terms)
