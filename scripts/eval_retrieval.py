"""离线检索评测 — 量化 Rerank Quick Win 的真实提升（rerank 关 vs 开）。

走与线上一致的真实链路：向量余弦召回 + BM25 + RRF 融合 + Cross-Encoder 重排。
全部离线（bge-small-zh-v1.5 向量 / bge-reranker-base 重排，均为本地缓存），无需起服务或调 LLM。

计算指标（按 golden_chunk_ids 精确匹配）：
  Recall@K / Hit@K / MRR  —— 在 rerank 关 与 开 两种条件下分别统计，对比提升。

用法:
  python scripts/eval_retrieval.py [--eval-dir data/eval] [--k 10] [--limit 0]
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
from engines.retrieval.dedup import adaptive_alpha, redundancy_reduce
from engines.retrieval.fusion import fuse
from engines.retrieval.query_augmentation import blend_vectors, prf_augment_vector
from engines.retrieval.reranker import Reranker


def build_index(chunks: list[dict]):
    embs = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.clip(norms, 1e-9, None)
    bm25 = Bm25Index()
    bm25.add_documents([{"id": c["chunk_id"], "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"]} for c in chunks])
    return embs_n, bm25


def vector_topk(q_emb: np.ndarray, embs_n: np.ndarray, chunks: list[dict], k: int) -> list[dict]:
    sims = embs_n @ q_emb
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        c = chunks[int(i)]
        out.append({"chunk_id": c["chunk_id"], "id": c["chunk_id"], "doc_id": c["doc_id"], "content": c["content"], "embedding": c["embedding"], "score": float(sims[i])})
    return out


def _metrics(order: list[dict], expected: set, ks: list[int]) -> dict:
    ids = [d.get("chunk_id") or d.get("id") for d in order]
    hit_rank = None
    for rank, cid in enumerate(ids, 1):
        if cid in expected:
            hit_rank = rank
            break
    res = {}
    for k in ks:
        top = set(ids[:k])
        res[f"recall@{k}"] = len(top & expected) / len(expected) if expected else 0.0
        res[f"hit@{k}"] = 1.0 if (top & expected) else 0.0
    res["mrr"] = (1.0 / hit_rank) if hit_rank else 0.0
    return res


def _rank(order: list[dict], expected: set) -> int | None:
    """第一个 golden chunk 的排名（1-based），未进入列表返回 None。"""
    ids = [d.get("chunk_id") or d.get("id") for d in order]
    for r, cid in enumerate(ids, 1):
        if cid in expected:
            return r
    return None


_HYDE_SYS = (
    "你是问答知识库的检索增强器。给定用户问题, 写一段可能出现在知识库中的『标准答案文档』"
    "(包含具体术语、步骤、要点), 用于提升向量检索召回。只输出文档正文, 不超过 80 字, 不要解释。"
)


def _hyde_doc_eval(query: str, emb) -> str | None:
    """HyDE 假设文档生成(同步, 离线评测用)。无 LLM/失败 → 返回 None (上层回退 none)。"""
    try:
        from api.core.llm_client import get_sync_llm_client

        client = get_sync_llm_client()
        if not getattr(client, "api_key", None):
            return None
        resp = client.chat(
            [{"role": "system", "content": _HYDE_SYS}, {"role": "user", "content": query}],
            temperature=0.0,
            max_tokens=96,
        )
        return (getattr(resp, "content", "") or "").strip() or None
    except Exception as e:  # noqa: BLE001 — 评测中 HyDE 为可选增强, 失败不阻断
        print(f"[eval] HyDE 跳过(无 LLM 或调用失败): {str(e)[:100]}")
        return None


_STOP = set(["的", "了", "在", "是", "和", "与", "及", "也", "都", "就", "而", "等", "中", "个", "上", "下", "里", "这", "那", "有", "为", "对", "从", "到", "把", "被", "给", "让", "向", "以", "于", "之", "其", "该", "此", "我", "你", "他", "她", "它", "们", "不", "没", "很", "更", "最", "并", "或", "即", "且", "一个", "多少", "哪些", "什么", "如何", "怎么", "是否"])


def _answer_signals(text: str) -> list[str]:
    """从标准答案抽取『答案信号词』(名词/专名/数字, 长度>=2, 去停用词), 作为答案支撑度量的锚。"""
    import jieba.posseg as pseg
    out = []
    for w, flag in pseg.cut(text):
        if len(w) < 2:
            continue
        if flag.startswith(("n", "m")) or flag in ("PER", "LOC", "ORG"):
            if w not in _STOP:
                out.append(w)
    return out


def _answer_support(order: list[dict], signals: list[str]) -> float:
    """top-k 检索块文本对答案信号词的覆盖率 (命中信号词数 / 总信号词去重数)。

    命中采用子串匹配: golden 块含该专名即算支撑 (如『天津』命中『天津滨海新区』)。
    作为 faithfulness 的本地 proxy — 不经过 LLM 生成/评判。
    """
    if not signals:
        return 1.0
    text = " ".join((d.get("content") or "") for d in order)
    uniq = set(signals)
    hit = sum(1 for s in uniq if s in text)
    return hit / len(uniq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--chunks", default="corpus_chunks.json",
                    help="语料块嵌入文件(默认 corpus_chunks.json; 重嵌入对比可用独立文件如 corpus_chunks_bgem3.json)")
    ap.add_argument("--embedding-model", default="",
                    help="覆盖 config 的 embedding 模型(仅本次评测, 不改生产配置)。如 BAAI/bge-m3")
    ap.add_argument("--embedding-backend", default="",
                    help="覆盖 embedding 后端(local/api)。api=商用兼容 OpenAI 的 Embedding API(需另配 --embedding-api-base/--embedding-api-key)")
    ap.add_argument("--dataset", default="eval_dataset.json", help="评测集文件名(默认 eval_dataset.json; 实体歧义集用 entity_ambig_dataset.json)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条问题(0=全部)")
    ap.add_argument("--no-rerank", action="store_true", help="跳过 rerank 开 分支(仅看基线, 省时)")
    ap.add_argument("--fusion-method", default="rrf", choices=["rrf", "interp"],
                    help="融合方法(默认 rrf, 与历史基线可比; interp=分数插值放大 BM25)")
    ap.add_argument("--bm25-weight", type=float, default=0.6, help="interp 融合的 BM25 权重(默认 0.6)")
    ap.add_argument("--dedup", action="store_true", help="(对照用)rerank 前相似度去重, 证明该路径有害(Recall@3 暴跌)")
    ap.add_argument("--dedup-threshold", type=float, default=0.0,
                    help="相似度去重阈值(默认 0.0=关闭, 仅同文档归并; 仅 --dedup 时有效)")
    ap.add_argument("--fusion", action="store_true", help="rerank 分与 RRF 检索分融合重排(正确修复密集近重复语料 rerank 伤召回), 对比 Recall 恢复")
    ap.add_argument("--adaptive", action="store_true", help="配合 --fusion: alpha 按候选池近重复密度自适应(而非固定 0.7)")
    ap.add_argument("--max-length", type=int, default=0,
                    help="CE 输入最大 token 数(0=用 settings.reranker_max_length 或模型默认)。用于验证截断对质量的影响")
    ap.add_argument("--backend", default="", help="覆盖 reranker 后端: torch/onnx/auto (默认 settings.reranker_backend)")
    ap.add_argument("--answer-support", action="store_true",
                    help="用 reference_answer 做『答案支撑充分性』对比(rerank 关 vs 开): "
                         "top-k 检索块对标准答案信号词的覆盖率, 作为 faithfulness 的本地 proxy, 不依赖 LLM")
    ap.add_argument("--augment", default="none", choices=["none", "prf", "hyde"],
                    help="查询嵌入增广(打开首阶段召回天花板): none / prf(伪相关反馈,离线) / hyde(假设文档,需LLM)")
    ap.add_argument("--augment-weight", type=float, default=0.5, help="增广向量融合权重(默认 0.5)")
    ap.add_argument("--augment-prf-k", type=int, default=10, help="PRF 首轮反馈候选数(默认 10)")
    ap.add_argument(
        "--out",
        default="",
        help="报告输出路径(默认 <eval-dir>/retrieval_summary_<dataset-stem>.json); "
        "显式指定可避免覆盖已提交的基线 summary",
    )
    args = ap.parse_args()

    ks = [1, 3, 5, 10]
    with open(os.path.join(args.eval_dir, args.chunks), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(args.eval_dir, args.dataset), encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"[eval] chunks={len(chunks)} questions={len(dataset)} chunks_file={args.chunks}")
    print(f"[eval] augment={args.augment} weight={args.augment_weight}"
          + (f" prf_k={args.augment_prf_k}" if args.augment == "prf" else ""))

    embs_n, bm25 = build_index(chunks)
    emb_model = args.embedding_model or settings.embedding_model
    emb_backend = args.embedding_backend or settings.embedding_backend
    emb = EmbeddingService(model_name=emb_model, backend=emb_backend,
                           api_base=settings.embedding_api_base, api_key=settings.embedding_api_key,
                           api_model=settings.embedding_api_model, api_dims=settings.embedding_api_dims)
    max_length = args.max_length or settings.reranker_max_length or None
    backend = args.backend or settings.reranker_backend
    reranker = Reranker(embedder=emb, ce_model_name=resolve_model_path(settings.reranker_model),
                        max_length=max_length, backend=backend) if settings.reranker_model else None
    ce_name = settings.reranker_model
    print(f"[eval] reranker model={ce_name} backend={backend} max_length={max_length or 'model-default'} "
          f"loaded={reranker is not None and reranker.warmup()}")

    agg_off, agg_on = {f"recall@{k}": [] for k in ks} | {f"hit@{k}": [] for k in ks} | {"mrr": []}, \
                      {f"recall@{k}": [] for k in ks} | {f"hit@{k}": [] for k in ks} | {"mrr": []}
    agg_on_dedup = {f"recall@{k}": [] for k in ks} | {f"hit@{k}": [] for k in ks} | {"mrr": []}
    agg_on_fusion = {f"recall@{k}": [] for k in ks} | {f"hit@{k}": [] for k in ks} | {"mrr": []}
    rerank_total = 0.0
    rerank_calls = 0
    dedup_total = 0.0
    dedup_calls = 0
    alphas: list[float] = []

    # P0-1: 难样本 / rerank 翻盘率 统计
    agg_off_hard = {f"recall@{k}": [] for k in ks} | {f"hit@{k}": [] for k in ks} | {"mrr": []}
    agg_on_hard = {f"recall@{k}": [] for k in ks} | {f"hit@{k}": [] for k in ks} | {"mrr": []}
    hard_items_total = 0

    # 答案支撑充分性 (faithfulness 本地 proxy)
    if args.answer_support:
        sig_cache = {it["id"]: _answer_signals(it.get("reference_answer", "")) for it in dataset}
        agg_off_as, agg_on_as = [], []
    else:
        sig_cache = None
    rrf_missed = 0
    recovered = 0

    for it in dataset:
        q = it["question"]
        expected = set(it["expected_chunk_ids"])
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)
        # 查询嵌入增广: 在向量腿对 q_emb 做 PRF/HyDE, BM25 腿仍用原查询。
        if args.augment == "prf":
            # 首轮 top-k 反馈向量取自已归一化的全量语料矩阵 embs_n
            sims = embs_n @ q_emb
            fb_idx = np.argsort(-sims)[: args.augment_prf_k]
            fb_embs = [embs_n[int(i)] for i in fb_idx]
            q_emb = prf_augment_vector(q_emb, fb_embs, weight=args.augment_weight)
        elif args.augment == "hyde":
            hyp = _hyde_doc_eval(q, emb)
            if hyp is not None:
                hyp_emb = np.array(emb.embed_batch([hyp[:512]])[0], dtype=np.float32)
                q_emb = blend_vectors(q_emb, hyp_emb, weight=args.augment_weight)
        v_docs = vector_topk(q_emb, embs_n, chunks, 40)
        b_docs = bm25.search(q, 40)
        fused = fuse(v_docs, b_docs, method=args.fusion_method, w_bm25=args.bm25_weight)

        off_order = fused[: args.k]
        m_off = _metrics(off_order, expected, ks)
        off_rank = _rank(off_order, expected)
        for key in m_off:
            agg_off[key].append(m_off[key])

        on_order = None
        on_order_dedup = None
        if not args.no_rerank and reranker:
            pool = fused[: settings.rerank_candidate_k] if settings.rerank_candidate_k > 0 else fused
            t = time.time()
            on_order = reranker.rerank(q, pool, top_k=args.k)
            rerank_total += time.time() - t
            rerank_calls += 1
            m_on = _metrics(on_order, expected, ks)
            for key in m_on:
                agg_on[key].append(m_on[key])

            if args.dedup:
                t = time.time()
                dedup_pool = redundancy_reduce(pool, embedder=emb,
                                               threshold=args.dedup_threshold,
                                               max_per_doc=1)
                dedup_total += time.time() - t
                dedup_calls += 1
                on_order_dedup = reranker.rerank(q, dedup_pool, top_k=args.k)
                m_dd = _metrics(on_order_dedup, expected, ks)
                for key in m_dd:
                    agg_on_dedup[key].append(m_dd[key])

            if args.fusion:
                rrf_map = {d.get("chunk_id") or d.get("id"): d.get("_rrf", 0.0) for d in fused}
                alpha = settings.rerank_fusion_alpha
                if args.adaptive:
                    alpha = adaptive_alpha(
                        pool,
                        threshold=settings.rerank_density_threshold,
                        mode=settings.rerank_density_mode,
                        alpha_max=settings.rerank_alpha_max,
                        alpha_min=settings.rerank_alpha_min,
                        density_full=settings.rerank_density_full,
                    )
                    alphas.append(alpha)
                on_order_fusion = reranker.rerank_fused(q, pool, rrf_map, top_k=args.k, alpha=alpha)
                m_fu = _metrics(on_order_fusion, expected, ks)
                for key in m_fu:
                    agg_on_fusion[key].append(m_fu[key])

        on_rank = _rank(on_order, expected) if on_order is not None else None

        # 答案支撑充分性: rerank 关 vs 开 的 top-k 对标准答案信号词覆盖
        if args.answer_support:
            sig = sig_cache.get(it["id"], [])
            agg_off_as.append(_answer_support(off_order, sig))
            if on_order is not None:
                agg_on_as.append(_answer_support(on_order, sig))

        # 难样本：BM25 漏召 golden 的问题，单独统计 rerank 关/开（rerank 价值主战场）
        if it.get("hard"):
            hard_items_total += 1
            for key in m_off:
                agg_off_hard[key].append(m_off[key])
            if on_rank is not None:
                for key in m_on:
                    agg_on_hard[key].append(m_on[key])

        # 翻盘率：RRF（rerank 关）漏召 golden，rerank 开 是否捞回 top-K
        if off_rank is None or off_rank > args.k:
            rrf_missed += 1
            if on_rank is not None and on_rank <= args.k:
                recovered += 1

    def _avg(d):
        return {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in d.items()}

    print("\n=== 检索评测 (golden_chunk_ids 精确匹配) ===")
    off_a, on_a = _avg(agg_off), _avg(agg_on)
    dd_a = _avg(agg_on_dedup) if args.dedup else None
    fu_a = _avg(agg_on_fusion) if args.fusion else None
    if args.fusion:
        print(f"{'指标':<12} {'rerank 关':>10} {'rerank 开':>10} {'开+融合':>10}")
    elif args.dedup:
        print(f"{'指标':<12} {'rerank 关':>10} {'rerank 开':>10} {'开+去重':>10}")
    else:
        print(f"{'指标':<12} {'rerank 关':>10} {'rerank 开':>10} {'提升':>10}")
    for k in ks:
        o = off_a[f"recall@{k}"]; n = on_a[f"recall@{k}"]
        if args.fusion:
            f = fu_a[f"recall@{k}"]
            print(f"{'Recall@'+str(k):<12} {o*100:>9.1f}% {n*100:>9.1f}% {f*100:>9.1f}%")
        elif args.dedup:
            d = dd_a[f"recall@{k}"]
            print(f"{'Recall@'+str(k):<12} {o*100:>9.1f}% {n*100:>9.1f}% {d*100:>9.1f}%")
        else:
            print(f"{'Recall@'+str(k):<12} {o*100:>9.1f}% {n*100:>9.1f}% {(n-o)*100:>+9.1f}%")
    for k in ks:
        o = off_a[f"hit@{k}"]; n = on_a[f"hit@{k}"]
        if args.fusion:
            f = fu_a[f"hit@{k}"]
            print(f"{'Hit@'+str(k):<12} {o*100:>9.1f}% {n*100:>9.1f}% {f*100:>9.1f}%")
        elif args.dedup:
            d = dd_a[f"hit@{k}"]
            print(f"{'Hit@'+str(k):<12} {o*100:>9.1f}% {n*100:>9.1f}% {d*100:>9.1f}%")
        else:
            print(f"{'Hit@'+str(k):<12} {o*100:>9.1f}% {n*100:>9.1f}% {(n-o)*100:>+9.1f}%")
    o = off_a["mrr"]; n = on_a["mrr"]
    if args.fusion:
        f = fu_a["mrr"]
        print(f"{'MRR':<12} {o:>10.3f} {n:>10.3f} {f:>10.3f}")
    elif args.dedup:
        d = dd_a["mrr"]
        print(f"{'MRR':<12} {o:>10.3f} {n:>10.3f} {d:>10.3f}")
    else:
        print(f"{'MRR':<12} {o:>10.3f} {n:>10.3f} {(n-o):>+10.3f}")
    if rerank_calls:
        print(f"\nrerank 平均耗时: {rerank_total/rerank_calls*1000:.0f}ms/查询 (n={rerank_calls})")
    if alphas:
        import statistics
        print(f"自适应 alpha: 均值 {statistics.mean(alphas):.3f}  "
              f"min {min(alphas):.3f}  max {max(alphas):.3f}  (n={len(alphas)})")
    if args.dedup and dedup_calls:
        print(f"去重平均耗时(复用已有 embedding): {dedup_total/dedup_calls*1000:.0f}ms/查询 (n={dedup_calls})")

    # 答案支撑充分性 (faithfulness 本地 proxy)
    if args.answer_support:
        off_as = sum(agg_off_as) / len(agg_off_as) if agg_off_as else 0.0
        on_as = sum(agg_on_as) / len(agg_on_as) if agg_on_as else 0.0
        print(f"\n=== 答案支撑充分性 (reference_answer 信号词覆盖, top-{args.k}) ===")
        print(f"{'条件':<12} {'支撑率':>10}")
        print(f"{'rerank 关':<12} {off_as*100:>9.1f}%")
        print(f"{'rerank 开':<12} {on_as*100:>9.1f}%")
        print(f"{'差异':<12} {(on_as-off_as)*100:>+9.1f}pp")
        print("(支撑率 = top-k 检索块文本对标准答案专名/数字信号词的覆盖比例; 越高越可能产出忠实答案)")

    # P0-1: 难样本检索指标 + rerank 翻盘率
    if hard_items_total:
        ho = _avg(agg_off_hard)
        hn = _avg(agg_on_hard)
        print(f"\n=== 难样本 (BM25 漏召 golden, n={hard_items_total}) ===")
        print(f"{'指标':<12} {'rerank 关':>10} {'rerank 开':>10} {'提升':>10}")
        for k in ks:
            o = ho[f"recall@{k}"]; n = hn.get(f"recall@{k}", 0.0)
            print(f"{'Recall@'+str(k):<12} {o*100:>9.1f}% {n*100:>9.1f}% {(n-o)*100:>+9.1f}%")
        o = ho["mrr"]; n = hn.get("mrr", 0.0)
        print(f"{'MRR':<12} {o:>10.3f} {n:>10.3f} {(n-o):>+10.3f}")
    if rrf_missed:
        rate = recovered / rrf_missed * 100 if rrf_missed else 0.0
        print(f"\n=== Rerank 翻盘率 (RRF 漏召@{args.k}) ===")
        print(f"RRF 漏召样本: {rrf_missed}/{len(dataset)}   rerank 捞回: {recovered}   翻盘率: {rate:.1f}%")

    summary = {
        "sample_count": len(dataset),
        "rerank_off": off_a,
        "rerank_on": on_a,
        "rerank_avg_ms": round(rerank_total / rerank_calls * 1000, 1) if rerank_calls else None,
        "reranker_model": ce_name,
        "embedding_model": emb_model,
        "hard_query_count": hard_items_total,
        "rrf_missed_count": rrf_missed,
        "rerank_recovery_rate": round(recovered / rrf_missed, 4) if rrf_missed else None,
        "rerank_on_dedup": dd_a if args.dedup else None,
        "rerank_on_fusion": fu_a if args.fusion else None,
        "rerank_fusion_alpha": settings.rerank_fusion_alpha if args.fusion else None,
        "answer_support": {"rerank_off": round(off_as, 4), "rerank_on": round(on_as, 4)} if args.answer_support else None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    stem = os.path.splitext(os.path.basename(args.dataset))[0]
    out = args.out or os.path.join(args.eval_dir, f"retrieval_summary_{stem}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
