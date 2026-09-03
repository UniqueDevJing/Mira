"""#3 跨库 rerank 端到端 A/B — 真实产品语料 12 道路由题 (无 LLM)。

对每题按「选择性扇出(3 候选库全出)」取各库融合候选 → 三种排序策略对比 gold chunk 命中:

  OFF      : 跨库去重合并后不重排(融合序) — #3 的降级兜底
  GLOBAL   : 跨库合并池做一次全局 rerank(生产 _rerank_safe 原样调用) = #3 现状
  PER-KB   : 各库先各自 rerank top-N, 再按 CE 分跨库拼接 — #3 之前的旧逻辑(基线)

语料 = data/bm25_{kb}.json 快照(service42/policy12/tech41), 向量现场嵌入(bge-small),
检索与线上一致: 向量 top40 + BM25 top40 + interp 融合(w=settings.fusion_bm25_weight)。

指标: gold chunk hit@1/3/5。用法: python scripts/eval_fanout_rerank_ab.py
"""
import argparse
import asyncio
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from api.config import settings
from api.core import orchestrator
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index
from engines.retrieval.fusion import fuse

KBS = ("service", "policy", "tech")


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_kb_indexes():
    """每库: (归一化向量矩阵, corpus list, BM25)。现场 embed 快照正文。"""
    emb = EmbeddingService()
    out = {}
    for kb in KBS:
        docs = _load(f"data/bm25_{kb}.json")["docs"]
        texts = [(d.get("content") or "") for d in docs]
        vecs = np.array([np.asarray(emb.embed_query(t[:2000]), dtype=np.float32) for t in texts])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_n = vecs / np.clip(norms, 1e-9, None)
        bm = Bm25Index()
        bm.add_documents([
            {"id": d["chunk_id"], "chunk_id": d["chunk_id"], "doc_id": d["doc_id"], "content": d.get("content", "")}
            for d in docs
        ])
        out[kb] = {"vecs": vecs_n, "docs": docs, "bm25": bm}
    return emb, out


def retrieve_fused(emb, idx, kb, q, k=20):
    """单库融合候选 (同线上 _retrieve_context: 向量+BM25+interp)。"""
    v = idx["vecs"] @ np.asarray(emb.embed_query(q), dtype=np.float32)
    topv = np.argsort(-v)[:40]
    v_docs = [
        {"chunk_id": idx["docs"][int(i)]["chunk_id"], "id": idx["docs"][int(i)]["chunk_id"],
         "doc_id": idx["docs"][int(i)]["doc_id"], "content": idx["docs"][int(i)]["content"],
         "kb": kb, "score": float(v[i])} for i in topv
    ]
    b_docs = idx["bm25"].search(q, 40)
    for d in b_docs:
        d["kb"] = kb
    fused = fuse(v_docs, b_docs, method=settings.fusion_method, w_bm25=settings.fusion_bm25_weight)
    return fused[:k]


def interleave_pool(per_kb: list[list[dict]], cap: int) -> list[dict]:
    """与 orchestrator._retrieve_fanout 修复后一致: 跨库去重(保最高 _rrf) + 轮转交织成候选池。"""
    best = {}
    for docs in per_kb:
        for d in docs:
            cid = d.get("chunk_id") or d.get("id")
            if cid is None:
                continue
            sc = d.get("_rrf", d.get("score", 0.0))
            if cid not in best or sc > best[cid].get("_rrf", best[cid].get("score", 0.0)):
                best[cid] = d
    pool, emitted = [], set()
    max_rank = max((len(f) for f in per_kb), default=0)
    for rank in range(max_rank):
        for docs in per_kb:
            if rank >= len(docs):
                continue
            d = docs[rank]
            cid = d.get("chunk_id") or d.get("id")
            if cid is None or cid in emitted:
                continue
            emitted.add(cid)
            pool.append(best.get(cid, d))
            if cap > 0 and len(pool) >= cap:
                break
        if cap > 0 and len(pool) >= cap:
            break
    return pool


async def rank_global(pool, q, top_k):
    """#3 现状: 全局一次 rerank(生产函数, 降级时返回融合序)。返回 (docs, ran_flag)。"""
    t = time.time()
    docs, _ms, _deg = await orchestrator._rerank_safe("", q, pool, top_k, t, 0)
    ran = bool(settings.rerank_enabled and settings.reranker_model and docs)
    return docs, ran


async def rank_per_kb(per_kb_fused, q, top_k):
    """旧逻辑: 各库各自 rerank top_k, 再按分跨库拼接 (baseline)。返回 docs(顺序=排序后)。"""
    t = time.time()
    all_r = []
    for kb, docs in per_kb_fused:
        if not docs:
            continue
        r, _ms, _deg = await orchestrator._rerank_safe(kb, q, docs, top_k, t, 0)
        all_r.extend(r)
    # 重排后按分降序(旧实现按 CE/融合分跨库合并排序)
    all_r.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return all_r[:top_k]


def _norm(s):
    return "".join((s or "").split())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=0,
                    help="覆盖 rerank_candidate_k(候选池预算), 对照池大小对 recall 的影响(0=用 settings)")
    args = ap.parse_args()
    if args.cap > 0:
        settings.rerank_candidate_k = args.cap  # interleave_pool 与 _rerank_safe 同读 settings
    t0 = time.time()
    emb, idxs = build_kb_indexes()
    sub = _load("data/_eval_p1_subset.json")
    # gold 内容集: 语料存在"同内容重复 doc"(1e379a96/48513be2 等), chunk_id 精确匹配会
    # 因保了另一副本而误判 miss —— 同时用内容口径评估(内容进 top-k 即算命中)。
    gold_ctx = {}
    for kb in KBS:
        docs = _load(f"data/bm25_{kb}.json")["docs"]
        gold_ctx[kb] = {d["chunk_id"]: _norm(d.get("content", "")) for d in docs}
    print(f"[load] {len(sub)} 题, rerank_model={settings.reranker_model} enabled={settings.rerank_enabled} "
          f"fusion={settings.fusion_method} w_bm25={settings.fusion_bm25_weight}", flush=True)

    rows = []
    agg = {m: {"OFF": [], "GLOBAL": [], "PERKB": []} for m in ("hit@1", "hit@3", "hit@5")}
    for it in sub:
        q = it["question"]
        gold_ids = set(it["expected_chunk_ids"])
        gold_texts = {gold_ctx[it["kb"]].get(c, "") for c in gold_ids}
        gold_texts.discard("")
        per_kb_fused = []
        for kb in KBS:
            fd = retrieve_fused(emb, idxs[kb], kb, q)
            per_kb_fused.append(fd)
        pool = interleave_pool(per_kb_fused, settings.rerank_candidate_k)
        # OFF: rerank 关闭的降级路径 = 交织池前 top_k(修复后跨库不再只剩主库)
        off = pool[:5]
        # GLOBAL(#3): 生产 _rerank_safe 原样(开则 CE, 关则同样取池前 top_k)
        g_docs, g_ran = await rank_global(pool, q, 5)
        # PER-KB(旧逻辑基线)
        p_docs = await rank_per_kb(list(zip(KBS, per_kb_fused)), q, 5)

        def hits(docs, g_ids=gold_ids, g_texts=gold_texts):
            ids = [d.get("chunk_id") or d.get("id") for d in docs]
            contents = [_norm(d.get("content", "")) for d in docs]
            return {
                f"hit@{k}": (1.0 if ((g_ids & set(ids[:k])) or (g_texts & set(contents[:k]))) else 0.0)
                for k in (1, 3, 5)
            }

        h = {"q": q[:34], "gold_kb": it["kb"], "pool": len(pool),
             "off": hits(off), "global": hits(g_docs), "perkb": hits(p_docs), "global_reranked": g_ran}
        rows.append(h)
        for m in ("hit@1", "hit@3", "hit@5"):
            agg[m]["OFF"].append(h["off"][m])
            agg[m]["GLOBAL"].append(h["global"][m])
            agg[m]["PERKB"].append(h["perkb"][m])

    print(f"\n{'题目(gold_kb)':<38}{'pool':>5} {'OFF':^16} {'GLOBAL(#3)':^16} {'PER-KB(旧)':^16}")
    for h in rows:
        def fmt(d):
            return f"{d['hit@1']:.0f}/{d['hit@3']:.0f}/{d['hit@5']:.0f}"
        print(f"{h['q']:<38}{h['pool']:>5} {fmt(h['off']):>16} {fmt(h['global']):>16} {fmt(h['perkb']):>16}")
    print("\n=== 汇总 hit@k (0-1) ===")
    print(f"{'指标':<8}{'OFF':>10}{'GLOBAL(#3)':>14}{'PER-KB(旧)':>14}")
    for m in ("hit@1", "hit@3", "hit@5"):
        print(f"{m:<8}" + "".join(f"{np.mean(agg[m][k]):>14.3f}" for k in ("OFF", "GLOBAL", "PERKB")))
    n_reran = sum(1 for h in rows if h["global_reranked"])
    print(f"\nGLOBAL 实际执行重排: {n_reran}/{len(rows)} 题 (其余因 rerank 关/空池走融合序降级)")
    print(f"耗时 {time.time() - t0:.0f}s", flush=True)

    out = {"dataset": "_eval_p1_subset", "n": len(rows), "rows": rows,
           "agg": {m: {k: round(float(np.mean(agg[m][k])), 3) for k in ("OFF", "GLOBAL", "PERKB")} for m in agg}}
    with open("data/eval/fanout_rerank_ab.json", "w", encoding="utf-8") as f:  # noqa: ASYNC230 — 收尾写盘
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("明细: data/eval/fanout_rerank_ab.json")


if __name__ == "__main__":
    asyncio.run(main())
