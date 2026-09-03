"""残余风险实证: low_relevance 是否在真实检索链路下新增误拒 (vs 已被 low_confidence 覆盖)。

背景: 离线评估(G1, 假设"检索完美命中")显示 entity 歧义集 2/221 会触发 low_relevance。
但 guard 只在 top1_score >= answer_confidence_floor(0.5) 时才进入 low_relevance 判定;
那几例 question↔gold 直接余弦仅 0.44-0.50, 真实检索分大概率 <0.5 → 会先命中 low_confidence。

本脚本对评测集跑真实离线检索(向量+BM25+interp 融合, 与线上 fusion_method 一致),
按 guard 真实判定路径重放, 统计:
  R1  gold 命中且 top1>=0.5(应作答) 中触发 low_relevance 的数量 = low_relevance **独有新增误拒**
  R2  gold 命中但 top1<0.5 的数量 = 分数下限(low_confidence)拒答域, 与 low_relevance 无关
  R3  全部触发 low_relevance 的样本(gold 命中与否), 供人工核验是否属坏题

用法: python scripts/eval_preguard_realretrieval.py
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from api.config import settings
from api.core.qa_metrics import _cosine, _word_overlap_faithfulness
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index
from engines.retrieval.fusion import fuse

FLOOR = settings.answer_confidence_floor  # 0.50
MIN_SIM = settings.low_relevance_min_sim  # 0.50


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    t0 = time.time()
    corpus = _load("data/eval/corpus_chunks.json")
    cmap = {c["chunk_id"]: c for c in corpus}
    embs = np.array([c["embedding"] for c in corpus], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.clip(norms, 1e-9, None)
    bm25 = Bm25Index()
    bm25.add_documents([
        {"id": c["chunk_id"], "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"]}
        for c in corpus
    ])
    emb = EmbeddingService()
    print(f"[load] corpus={len(corpus)} emb_dim={embs_n.shape[1]} floor={FLOOR} min_sim={MIN_SIM}", flush=True)

    def vector_topk(q_emb, k=40):
        sims = embs_n @ q_emb
        idx = np.argsort(-sims)[:k]
        return [
            {"chunk_id": corpus[int(i)]["chunk_id"], "id": corpus[int(i)]["chunk_id"],
             "doc_id": corpus[int(i)]["doc_id"], "content": corpus[int(i)]["content"],
             "score": float(sims[i])} for i in idx
        ]

    datasets = [("eval390", "data/eval/eval_dataset.json"),
                ("entity221", "data/eval/entity_ambig_dataset.json")]
    all_new_refusal = []
    for ds_name, ds_path in datasets:
        ds = _load(ds_path)
        gold_hit_ge_floor = 0
        gold_hit_lt_floor = 0
        new_refusal = []
        low_conf_cases = []
        for it in ds:
            q = it["question"]
            expected = set(it["expected_chunk_ids"])
            q_emb = np.asarray(emb.embed_query(q), dtype=np.float32)
            v = vector_topk(q_emb)
            b = bm25.search(q, 40)
            fused = fuse(v, b, method=settings.fusion_method, w_bm25=settings.fusion_bm25_weight)
            if not fused:
                continue
            top1 = fused[0]
            t1_score = float(top1.get("score", 0.0))
            docs = fused[:5]
            if not (expected & {d.get("chunk_id") for d in docs}):
                continue  # 检索失败本身是召回问题, 非护栏误拒
            if t1_score < FLOOR:
                gold_hit_lt_floor += 1
                if len(low_conf_cases) < 5:
                    low_conf_cases.append({"q": q[:44], "top1": round(t1_score, 3)})
                continue
            gold_hit_ge_floor += 1
            ov = _word_overlap_faithfulness(q, [d.get("content") or "" for d in docs])
            if ov == 0.0:
                # guard 语义兜底同款: q 与 top-docs 的最大余弦(corpus 自带向量)
                ctx_embs = [np.asarray(cmap[cid]["embedding"], dtype=np.float32)
                            for cid in (d.get("chunk_id") for d in docs) if cid in cmap]
                best = max((_cosine(q_emb, ce) for ce in ctx_embs), default=0.0) if ctx_embs else 0.0
                if best < MIN_SIM:
                    new_refusal.append({"id": it.get("id"), "q": q[:50], "top1": round(t1_score, 3),
                                        "sim": round(best, 3)})
        print(f"\n=== {ds_name} n={len(ds)} ===", flush=True)
        print(f"gold 命中且 top1>={FLOOR} (进入 low_relevance 判定域)      : {gold_hit_ge_floor}", flush=True)
        print(f"gold 命中但 top1< {FLOOR} (被 low_confidence 覆盖, 与此守卫无关): {gold_hit_lt_floor}", flush=True)
        print(f"low_relevance 独有新增拒答(零重合 且 语义<{MIN_SIM}): {len(new_refusal)} 条", flush=True)
        for c in new_refusal:
            print("   ", c, flush=True)
        if low_conf_cases:
            print("  low_confidence 覆盖样本示例:", low_conf_cases, flush=True)
        all_new_refusal += new_refusal

    print(f"\n>> 真实检索下 low_relevance 独有新增误拒合计 = {len(all_new_refusal)} 条"
          f"(耗时 {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
