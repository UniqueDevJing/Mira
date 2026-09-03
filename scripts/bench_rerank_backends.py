"""对比 rerank 各后端 (PyTorch / ONNX-fp32 / ONNX-int8) 的延迟与排序一致性。

用途: 决定生产该用哪个后端。INT8 是否可用不能只看"最大绝对误差", 更要看
**它是否改变了最终排序** —— rerank 只关心相对顺序, 近似并列时的翻转无实际影响。

判据:
  - 延迟: 相对 PyTorch 的加速比
  - 排序: 与 PyTorch 的 top-1 一致率 / top-3 集合一致率 / Kendall tau
  - 分数: 最大绝对误差

用法:
  python scripts/bench_rerank_backends.py [--eval-dir data/eval] [--limit 50] [--repeat 3]
"""
from __future__ import annotations

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


def _order(scores):
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def _kendall(a, b):
    """Kendall tau-b 简化版 (无并列时即 tau)。"""
    n = len(a)
    if n < 2:
        return 1.0
    oa, ob = _order(a), _order(b)
    rank_b = {int(idx): r for r, idx in enumerate(ob)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (rank_b[int(oa[i])] - rank_b[int(oa[j])])
            if s < 0:
                conc += 1
            elif s > 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--limit", type=int, default=50, help="评测问题数")
    ap.add_argument("--pool", type=int, default=10, help="重排候选池大小")
    ap.add_argument("--repeat", type=int, default=3, help="每组延迟测量重复次数")
    ap.add_argument("--onnx-dir", default="", help="ONNX 目录, 默认 <reranker_model>-onnx")
    args = ap.parse_args()

    with open(os.path.join(args.eval_dir, "corpus_chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(args.eval_dir, "eval_dataset.json"), encoding="utf-8") as f:
        dataset = json.load(f)[: args.limit]
    print(f"[bench] chunks={len(chunks)} questions={len(dataset)} pool={args.pool}")

    embs_n, bm25 = build_index(chunks)
    emb = EmbeddingService()

    # ---- 构造候选池 (与线上一致的 向量+BM25+RRF) ----
    pools: list[list[str]] = []
    for it in dataset:
        q = it["question"]
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)
        v_docs = vector_topk(q_emb, embs_n, chunks, 40)
        b_docs = bm25.search(q, 40)
        fused = rrf_fuse(v_docs, b_docs)[: args.pool]
        pools.append([q, *(d.get("content", "")[:512] for d in fused)])
    # pools[i] = [query, doc1, doc2, ...]

    # ---- 加载后端 ----
    from sentence_transformers import CrossEncoder

    src = resolve_model_path(settings.reranker_model)
    ce = CrossEncoder(src)
    backends: dict[str, object] = {"pytorch": ce}

    onnx_dir = args.onnx_dir or (src.rstrip("/\\") + "-onnx")
    from engines.retrieval.onnx_scorer import OnnxCrossEncoder

    fp32_p = os.path.join(onnx_dir, "model.onnx")
    int8_p = os.path.join(onnx_dir, "model_int8.onnx")
    if os.path.exists(fp32_p):
        backends["onnx-fp32"] = OnnxCrossEncoder(onnx_dir, prefer_int8=False)
    else:
        print(f"[bench] ⚠️ 未找到 {fp32_p}, 跳过 onnx-fp32")
    if os.path.exists(int8_p):
        backends["onnx-int8"] = OnnxCrossEncoder(onnx_dir, prefer_int8=True)
    else:
        print(f"[bench] ⚠️ 未找到 {int8_p}, 跳过 onnx-int8")

    def pairs_of(pool):
        return [(pool[0], t) for t in pool[1:]]

    # ---- 打分 + 延迟 ----
    results: dict[str, list] = {k: [] for k in backends}
    lat: dict[str, list[float]] = {k: [] for k in backends}

    for name, be in backends.items():
        for _ in range(2):  # warmup
            be.predict(pairs_of(pools[0]))
        for pool in pools:
            prs = pairs_of(pool)
            t = time.time()
            sc = None
            for _ in range(args.repeat):
                sc = be.predict(prs)
            lat[name].append((time.time() - t) / args.repeat * 1000)
            results[name].append(np.asarray(sc, dtype=np.float64))

    # ---- 报告 ----
    ref = results["pytorch"]
    print(f"\n=== 延迟 (ms/查询, 池={args.pool}) ===")
    base = float(np.mean(lat["pytorch"]))
    for name in backends:
        m = float(np.mean(lat[name]))
        p95 = float(np.percentile(lat[name], 95))
        print(f"  {name:<12} 均值 {m:7.1f}ms   p95 {p95:7.1f}ms   加速 {base/m:5.2f}×")

    print("\n=== 与 PyTorch 的一致性 ===")
    print(f"{'后端':<12} {'top1一致':>9} {'top3集合一致':>12} {'Kendallτ':>9} {'最大|Δ分|':>10}")
    for name in backends:
        if name == "pytorch":
            continue
        top1 = top3 = 0
        taus, maxd = [], []
        for a, b in zip(ref, results[name]):
            oa, ob = _order(a), _order(b)
            top1 += int(oa[0] == ob[0])
            top3 += int(set(oa[:3].tolist()) == set(ob[:3].tolist()))
            taus.append(_kendall(a, b))
            maxd.append(float(np.abs(a - b).max()))
        n = len(ref)
        print(f"  {name:<12} {top1/n*100:>8.1f}% {top3/n*100:>11.1f}% "
              f"{np.mean(taus):>9.4f} {np.max(maxd):>10.5f}")

    print("\n判读: top1 一致率 ≥99% 且 Kendallτ ≥0.98 视为可安全替换;")
    print("      否则该后端会改变 rerank 结果, 不建议启用。")


if __name__ == "__main__":
    main()
