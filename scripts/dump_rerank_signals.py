"""转储 rerank 打分信号 — 供 scripts/tune_alpha.py 离线网格搜索 alpha, 不重复跑 CE 推理。

对每个问题只做一次 Cross-Encoder 推理, 把 (CE 分, RRF 分, embedding, golden mask) 落盘;
之后 tune_alpha.py 用纯 numpy 秒级评估任意 alpha / 自适应映射, 无需重新跑模型。

用法:
  python scripts/dump_rerank_signals.py [--eval-dir data/eval] [--out data/eval/signals_hard] [--limit 0]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import numpy as np
from eval_retrieval import build_index, vector_topk

from api.config import settings
from api.state import resolve_model_path
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.fusion import rrf_fuse
from engines.retrieval.reranker import Reranker


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--out", required=True, help="输出前缀, 生成 <out>.npz + <out>.meta.json")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条问题(0=全部)")
    args = ap.parse_args()

    with open(os.path.join(args.eval_dir, "corpus_chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(args.eval_dir, "eval_dataset.json"), encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"[dump] chunks={len(chunks)} questions={len(dataset)}", flush=True)

    embs_n, bm25 = build_index(chunks)
    emb = EmbeddingService()
    reranker = Reranker(embedder=emb, ce_model_name=resolve_model_path(settings.reranker_model))
    print(f"[dump] reranker loaded={reranker.warmup()}", flush=True)
    ce_model = reranker._get_ce_model()
    if ce_model is None:
        print("[dump] FATAL: Cross-Encoder 未加载, 无法转储 CE 分")
        sys.exit(1)

    cap = settings.rerank_candidate_k
    k = cap if cap > 0 else 10

    ce_list, rrf_list, emb_list, gold_list = [], [], [], []
    qids = []
    t0 = time.time()
    for idx, it in enumerate(dataset):
        q = it["question"]
        expected = set(it["expected_chunk_ids"])
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)
        v_docs = vector_topk(_norm(q_emb), embs_n, chunks, 40)
        b_docs = bm25.search(q, 40)
        fused = rrf_fuse(v_docs, b_docs)
        pool = fused[:k]

        pairs = [(q, d.get("content", "")[:512]) for d in pool]
        ce = np.asarray(ce_model.predict(pairs), dtype=np.float64)
        rrf = np.asarray([float(d.get("_rrf", 0.0)) for d in pool], dtype=np.float64)
        pool_embs = np.stack([
            _norm(np.asarray(d.get("embedding"), dtype=np.float32))
            if d.get("embedding") is not None else np.zeros(embs_n.shape[1], dtype=np.float32)
            for d in pool
        ])
        gold = np.asarray([
            1 if (d.get("chunk_id") or d.get("id")) in expected else 0 for d in pool
        ], dtype=np.int8)

        ce_list.append(ce)
        rrf_list.append(rrf)
        emb_list.append(pool_embs)
        gold_list.append(gold)
        qids.append(q)

        if (idx + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[dump] {idx+1}/{len(dataset)}  用时 {el:.0f}s", flush=True)

    np.savez_compressed(
        args.out + ".npz",
        ce=np.stack(ce_list),
        rrf=np.stack(rrf_list),
        embs=np.stack(emb_list),
        gold=np.stack(gold_list),
    )
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"questions": qids, "count": len(qids), "pool_k": k}, f, ensure_ascii=False)
    print(f"[dump] done -> {args.out}.npz + .meta.json  ({len(qids)} 问, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
