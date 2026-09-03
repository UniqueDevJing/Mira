"""闭环质量验证探针 (离线/自托管均可跑)。

不依赖外部标注集, 用「自检索召回」作为生产 KB 可检索性的可量化代理, 并覆盖:
  1. 自检索召回: 每个生产 KB 抽 chunk, 以其片段/标题作 query, 测 _retrieve_context 能否召回自身
  2. 语义缓存闭环: 重复/近重复 query 的 exact/semantic 命中率 + 实测相似度(校验 0.92 阈值)
  3. 韧性: PRF/rerank 超时回退, embedding 失败降级
  4. 端到端生成: ask() 采样, 延迟 + faithfulness 启发式(答案 vs 检索上下文词重叠)

输出 JSON 报告到 --out (默认 data/eval/closed_loop_report.json)。
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time

# 允许以 scripts/xxx.py 直接运行 (venv 未 editable 安装时, 需把项目根加入 sys.path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from api.config import settings
from api.core.degradation import _deg_reset
from api.core.qa_cache import QACache
from api.core.retrieval import _retrieve_context
from api.core.skills import ask
from engines.router.intent_router import RoutingResult


def _tok(s: str) -> set:
    # 中文按字 + 英文按词, 极简重叠度量
    s = (s or "").lower()
    s = re.sub(r"\s+", "", s)
    eng = set(re.findall(r"[a-z0-9]+", s))
    cjk = set(re.findall(r"[\u4e00-\u9fff]", s))
    return eng | cjk


def _overlap(a: str, b: str) -> float:
    ta, tb = _tok(a), _tok(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


async def self_retrieval(kbs, sample_per_kb, top_k):
    import lancedb

    db = lancedb.connect(settings.vector_uri)
    out = {}
    per_kb_detail = {}
    for kb in kbs:
        try:
            tbl = db.open_table(kb)
        except Exception as e:  # noqa: BLE001
            out[kb] = {"error": f"open_table failed: {e}"}
            continue
        rows = tbl.search().limit(sample_per_kb).to_list()
        if not rows:
            out[kb] = {"error": "empty table"}
            continue
        hits = {1: 0, 3: 0, 5: 0, 10: 0}
        lat = []
        rerank_lat = []
        deg = 0
        detail = []
        for r in rows:
            cid = r.get("id") or r.get("chunk_id")
            content = r.get("content") or ""
            if not cid or len(content) < 20:
                continue
            # 两种探针: verbatim 片段(易) + 标题/首句(难)
            q_verb = content[:60]
            q_hard = (r.get("doc_title") or content[:30]).strip()
            if not q_hard:
                continue
            for q, kind in ((q_verb, "verbatim"), (q_hard, "title")):
                start = time.time()
                try:
                    _deg_reset()
                    res = await _retrieve_context(
                        q, RoutingResult(skill="rag", kb=kb, confidence=1.0, source="manual"),
                        top_k=top_k, start=start, mode="hybrid",
                    )
                except Exception as e:  # noqa: BLE001
                    detail.append({"cid": cid, "kind": kind, "error": str(e)[:120]})
                    continue
                ids = [d.get("chunk_id") for d in res.get("docs", [])]
                lat.append(res.get("retrieval_ms", 0.0))
                rerank_lat.append(res.get("rerank_ms", 0.0))
                if res.get("degradation"):
                    deg += 1
                for k in hits:
                    if cid in ids[:k]:
                        hits[k] += 1
                        break
                detail.append({"cid": cid, "kind": kind, "top1": ids[0] if ids else None,
                               "hit": cid in ids[:top_k], "retrieval_ms": round(res.get("retrieval_ms", 0), 1)})
        n = max(1, len(detail))
        out[kb] = {
            "sample": n,
            "recall@1": round(hits[1] / n, 3),
            "recall@3": round(hits[3] / n, 3),
            "recall@5": round(hits[5] / n, 3),
            "recall@10": round(hits[10] / n, 3),
            "retrieval_ms_p50": round(sorted(lat)[len(lat) // 2] if lat else 0, 1),
            "rerank_ms_p50": round(sorted(rerank_lat)[len(rerank_lat) // 2] if rerank_lat else 0, 1),
            "degradation_count": deg,
            "rerank_enabled": settings.rerank_enabled,
        }
        per_kb_detail[kb] = detail
    return out, per_kb_detail


async def semantic_cache_loop():
    cache = QACache()
    # 取一批真实短语作 query 源
    import lancedb
    db = lancedb.connect(settings.vector_uri)
    qs = []
    for kb in ["rag_tech", "rag_policy", "rag_service"]:
        try:
            rows = db.open_table(kb).search().limit(8).to_list()
        except Exception:  # noqa: BLE001
            continue
        for r in rows:
            c = (r.get("content") or "")[:50]
            if len(c) > 15:
                qs.append(c)
    if not qs:
        return {"error": "no queries sampled"}
    ttl = 3600
    exact_hit = 0
    sem_hit = 0
    miss = 0
    sims = []
    # 第一轮写入
    for q in qs:
        key = cache.make_key(q, None, 10, False, 0.1, "hybrid", None, None)
        scope = cache.make_scope(q, None, 10, False, 0.1, "hybrid", None, None)
        stats = {}
        hit = cache.get(key, question=q, scope=scope, stats=stats)
        if hit is not None:
            exact_hit += 1
        else:
            miss += 1
            cache.set(key, {"answer": "CACHED_" + q[:10], "kb_id": "documents"}, ttl, question=q, scope=scope)
    # 第二轮: 重复(精确) + 轻改写(语义)
    for q in qs:
        key = cache.make_key(q, None, 10, False, 0.1, "hybrid", None, None)
        scope = cache.make_scope(q, None, 10, False, 0.1, "hybrid", None, None)
        stats = {}
        hit = cache.get(key, question=q, scope=scope, stats=stats)
        if hit is not None and stats.get("kind") == "exact":
            exact_hit += 1
        else:
            miss += 1
    # 轻改写: 加"请问"前缀 / 改标点, 测语义命中 + 实测相似度
    near_dup_pairs = 0
    for q in qs[: min(20, len(qs))]:
        q2 = "请问" + q
        key2 = cache.make_key(q2, None, 10, False, 0.1, "hybrid", None, None)
        scope2 = cache.make_scope(q2, None, 10, False, 0.1, "hybrid", None, None)
        stats = {}
        hit = cache.get(key2, question=q2, scope=scope2, stats=stats)
        if hit is not None:
            sem_hit += 1
            sims.append(stats.get("sim"))
            if stats.get("kind") == "semantic":
                near_dup_pairs += 1
    return {
        "unique_queries": len(qs),
        "exact_hits_round2": exact_hit,
        "semantic_hits_round3": sem_hit,
        "misses": miss,
        "near_dup_semantic_hit": near_dup_pairs,
        "semantic_sim_samples": [round(s, 3) for s in sims if s is not None][:10],
        "threshold": settings.qa_cache_semantic_threshold,
    }


async def resilience():
    res = {}
    # PRF 超时回退
    try:
        old = settings.query_augmentation_prf_timeout_s
        settings.query_augmentation_prf_timeout_s = 1e-4
        start = time.time()
        _deg_reset()
        r = await _retrieve_context("如何申请退款", RoutingResult(skill="rag", kb="rag_service", confidence=1.0, source="manual"), top_k=5, start=start)
        res["prf_timeout"] = {"ok": bool(r.get("docs")), "docs": len(r.get("docs", [])), "degradation": r.get("degradation")}
    except Exception as e:  # noqa: BLE001
        res["prf_timeout"] = {"ok": False, "error": str(e)[:120]}
    finally:
        settings.query_augmentation_prf_timeout_s = old
    # Rerank 超时回退
    try:
        old = settings.rerank_timeout_s
        settings.rerank_timeout_s = 1e-4
        start = time.time()
        _deg_reset()
        r = await _retrieve_context("发票怎么开", RoutingResult(skill="rag", kb="rag_finance", confidence=1.0, source="manual"), top_k=5, start=start)
        res["rerank_timeout"] = {"ok": bool(r.get("docs")), "docs": len(r.get("docs", [])), "degradation": r.get("degradation"), "rerank_ms": r.get("rerank_ms")}
    except Exception as e:  # noqa: BLE001
        res["rerank_timeout"] = {"ok": False, "error": str(e)[:120]}
    finally:
        settings.rerank_timeout_s = old
    # Embedding 失败降级 (monkeypatch _embed_query)
    try:
        import api.core.retrieval as R
        orig = R._embed_query
        async def _boom(emb, vq):
            raise RuntimeError("simulated embed failure")
        R._embed_query = _boom
        start = time.time()
        _deg_reset()
        r = await _retrieve_context("退货流程", RoutingResult(skill="rag", kb="rag_service", confidence=1.0, source="manual"), top_k=5, start=start)
        res["embed_fail"] = {"ok": bool(r.get("docs")), "docs": len(r.get("docs", [])), "degradation": r.get("degradation"),
                             "note": "期望 BM25 兜底, 非崩溃"}
    except Exception as e:  # noqa: BLE001
        res["embed_fail"] = {"ok": False, "error": str(e)[:120]}
    finally:
        R._embed_query = orig
    return res


async def e2e_generation(kbs, per_kb):
    import lancedb
    db = lancedb.connect(settings.vector_uri)
    out = []
    for kb in kbs:
        try:
            rows = db.open_table(kb).search().limit(per_kb).to_list()
        except Exception:  # noqa: BLE001
            continue
        for r in rows:
            content = r.get("content") or ""
            q = (r.get("doc_title") or content[:40]).strip()
            if len(q) < 8:
                continue
            t0 = time.time()
            try:
                ans = await ask(q, allowed_kbs=[kb], session_id=None)
            except Exception as e:  # noqa: BLE001
                out.append({"kb": kb, "q": q, "error": str(e)[:120]})
                continue
            total_ms = (time.time() - t0) * 1000
            answer = ans.get("answer", "")
            # faithfulness 启发式: 答案 vs 检索上下文(同 query 经 _retrieve_context)
            try:
                _deg_reset()
                ctx = await _retrieve_context(q, RoutingResult(skill="rag", kb=kb, confidence=1.0, source="manual"), top_k=5, start=time.time())
                ctx_text = " ".join(d.get("content", "") for d in ctx.get("docs", []))
            except Exception:  # noqa: BLE001
                ctx_text = ""
            out.append({
                "kb": kb, "q": q[:40], "cache_hit": ans.get("cache_hit", False),
                "total_ms": round(total_ms, 0), "answer_len": len(answer),
                "faithfulness_proxy": round(_overlap(answer, ctx_text), 3),
                "has_answer": bool(answer),
            })
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kbs", default="rag_tech,rag_policy,rag_service,rag_finance,rag_hr,rag_marketing,rag_meeting,rag_product")
    ap.add_argument("--sample-per-kb", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--e2e-per-kb", type=int, default=2)
    ap.add_argument("--out", default="data/eval/closed_loop_report.json")
    args = ap.parse_args()
    kbs = [k.strip() for k in args.kbs.split(",") if k.strip()]

    print(f"[1/4] 自检索召回 ({len(kbs)} KB x {args.sample_per_kb}) ...", flush=True)
    sr, _ = await self_retrieval(kbs, args.sample_per_kb, args.top_k)

    print("[2/4] 语义缓存闭环 ...", flush=True)
    cache = await semantic_cache_loop()

    print("[3/4] 韧性 (PRF/rerank 超时, embedding 失败) ...", flush=True)
    res = await resilience()

    print("[4/4] 端到端生成采样 ...", flush=True)
    e2e = await e2e_generation(kbs, args.e2e_per_kb)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "rerank_enabled": settings.rerank_enabled,
            "rerank_backend": settings.reranker_backend,
            "aug_enabled": settings.query_augmentation_enabled,
            "aug_strategy": settings.query_augmentation_strategy,
            "semantic_cache": settings.qa_cache_semantic_enabled,
            "semantic_threshold": settings.qa_cache_semantic_threshold,
        },
        "self_retrieval": sr,
        "semantic_cache": cache,
        "resilience": res,
        "e2e_generation": e2e,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已写入 {args.out}")

    # 摘要
    print("\n=== 自检索召回 (生产 KB 可检索性代理) ===")
    for kb, v in sr.items():
        if "error" in v:
            print(f"  {kb}: ERROR {v['error']}")
        else:
            print(f"  {kb}: R@1={v['recall@1']} R@5={v['recall@5']} R@10={v['recall@10']} "
                  f"ret={v['retrieval_ms_p50']}ms rerank={v['rerank_ms_p50']}ms deg={v['degradation_count']}/{v['sample']}")
    print(f"\n=== 语义缓存 ===\n  {json.dumps(cache, ensure_ascii=False)}")
    print(f"\n=== 韧性 ===\n  {json.dumps(res, ensure_ascii=False)}")
    print(f"\n=== 端到端生成样本数={len(e2e)} ===")
    faith = [e['faithfulness_proxy'] for e in e2e if 'faithfulness_proxy' in e]
    if faith:
        print(f"  faithfulness_proxy 均值={round(sum(faith)/len(faith),3)} 范围=[{min(faith)}, {max(faith)}]")


if __name__ == "__main__":
    asyncio.run(main())
