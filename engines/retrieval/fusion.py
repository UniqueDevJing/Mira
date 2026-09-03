"""检索融合 — RRF (Reciprocal Rank Fusion) + 分数插值融合。

从 orchestrator 下沉为独立模块: 主检索 / 自检索超时降级 / 跨库兜底统一调用,
避免融合语义分散在编排层。带 _rrf 分数标注, 供重排前比较。

分数插值融合 (interp) 用于修正 RRF 的系统性缺陷: RRF 是排名融合、无法对任一路加权,
在"专名/关键词驱动"语料上 BM25 远强于向量(diag_fusion_weights.py: w=0.5 时
Recall@1 39.6% vs RRF 34.7%), 但 RRF 把 BM25 强信号稀释到接近向量水平。
interp 用 final = w*BM25_norm + (1-w)*向量_norm 直接放大 BM25 腿。
"""

import os

RRF_K = int(os.environ.get("RAG_RRF_K", "30"))


def rrf_fuse(vector_docs: list[dict], bm25_docs: list[dict], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: 按排名加权融合两路结果。"""
    if k <= 0:  # 防御: k 必须为正, 否则 1/(k+rank+1) 语义错误或除零 (rank=0 时)
        k = RRF_K
    if not vector_docs:
        return _tag_rrf(bm25_docs, k)
    if not bm25_docs:
        return _tag_rrf(vector_docs, k)

    rrf: dict[str, float] = {}
    merged: dict[str, dict] = {}

    def _feed(docs):
        for rank, d in enumerate(docs):
            key = d.get("chunk_id") or d.get("id")
            if not key:
                continue
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in merged:
                merged[key] = d

    _feed(vector_docs)
    _feed(bm25_docs)

    out = []
    for key in sorted(rrf, key=rrf.get, reverse=True):
        d = dict(merged[key])
        d["_rrf"] = round(rrf[key], 6)
        out.append(d)
    return out


def _tag_rrf(docs: list[dict], k: int = RRF_K) -> list[dict]:
    """单路透传时也补 _rrf 字段, 对齐输出形状 (排名加权语义, 与双路融合可比)。

    与 _feed 一致: 跳过无 chunk_id/id 的 doc, 避免泄漏无标识文档到下游。
    """
    out = []
    for rank, d in enumerate(docs):
        if not (d.get("chunk_id") or d.get("id")):
            continue
        nd = dict(d)
        nd["_rrf"] = round(1.0 / (k + rank + 1), 6)
        out.append(nd)
    return out


def score_interpolate_fuse(vector_docs: list[dict], bm25_docs: list[dict], w_bm25: float = 0.6) -> list[dict]:
    """分数插值融合: final = w_bm25*BM25_norm + (1-w_bm25)*向量_norm。

    与 rrf_fuse 同构 (输出 dict 带 _rrf 字段, 可直接喂 _rerank_safe / rerank_fused),
    但能按权重放大 BM25 腿信号。BM25 score 已是 per-query 归一化 0-1;
    向量余弦 [-1,1] 用 (cos+1)/2 映射到 0-1, 与 diag_fusion_weights.py 验证口径一致。
    任一路缺失时退化为另一路 (w=1 或 w=0 等效)。
    """
    w_bm25 = max(0.0, min(1.0, w_bm25))
    vmap = {d.get("chunk_id") or d.get("id"): d for d in vector_docs if (d.get("chunk_id") or d.get("id"))}
    bmap = {d.get("chunk_id") or d.get("id"): d for d in bm25_docs if (d.get("chunk_id") or d.get("id"))}
    ids = set(vmap) | set(bmap)
    scored = []
    for cid in ids:
        vd = vmap.get(cid)
        bd = bmap.get(cid)
        vscore = (float(vd.get("score", 0.0)) + 1.0) / 2.0 if vd else 0.0
        bscore = float(bd.get("score", 0.0)) if bd else 0.0
        final = w_bm25 * bscore + (1.0 - w_bm25) * vscore
        src = vd or bd
        out = dict(src)
        out["_rrf"] = round(final, 6)
        out["_fusion"] = "interp"
        scored.append(out)
    scored.sort(key=lambda d: d["_rrf"], reverse=True)
    return scored


def fuse(vector_docs: list[dict], bm25_docs: list[dict],
         method: str = "rrf", w_bm25: float = 0.6, k: int = RRF_K) -> list[dict]:
    """融合调度器: method='rrf' 走 RRF(默认, 向后兼容); 'interp' 走分数插值融合。"""
    if method == "interp":
        return score_interpolate_fuse(vector_docs, bm25_docs, w_bm25=w_bm25)
    return rrf_fuse(vector_docs, bm25_docs, k=k)
