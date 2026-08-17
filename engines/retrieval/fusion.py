"""检索融合 — RRF (Reciprocal Rank Fusion)。

从 orchestrator 下沉为独立模块: 主检索 / 自检索超时降级 / 跨库兜底统一调用,
避免融合语义分散在编排层。带 _rrf 分数标注, 供重排前比较。
"""

RRF_K = 60


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
