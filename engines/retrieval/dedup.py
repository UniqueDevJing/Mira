"""候选池冗余度量与削减 — 为 rerank 融合提供"语料密度"信号。

历史 (P0-1 实测结论, 勿重蹈):
  在 rerank **之前**按相似度"删除"候选是有害的 —— 近重复干扰项靠共享原词在 RRF 里常排在
  golden 之前, 贪心去重"保先到者删相似者"会把 golden 提前删掉 (Recall@3 92.5%→46.2%)。
  相似度信号的正确用法是**加权**(见 adaptive_alpha), 不是剪枝。

本模块现在提供两件事:
  1. pool_density()      —— 候选池近重复密度度量 (零模型调用, 复用已有 embedding)
  2. adaptive_alpha()    —— 按密度动态给出 rerank 融合权重
  (redundancy_reduce 保留, 仅作评测对照, 未接入生产链路)
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _embeddings_of(docs: list[dict], embedder, truncate: int = 512) -> list:
    """返回每篇文档的归一化 embedding; 缺失且 embedder 可用则批量补齐。

    无 embedding 且无 embedder 的文档返回 None (后续不参与相似度判定, 直接保留)。
    """
    embs: list = []
    missing_idx: list[int] = []
    for i, d in enumerate(docs):
        e = d.get("embedding")
        embs.append(np.asarray(e, dtype=np.float32) if e is not None else None)
        if e is None and embedder is not None:
            missing_idx.append(i)
    if missing_idx and embedder is not None:
        texts = [docs[i].get("content", "")[:truncate] for i in missing_idx]
        try:
            vecs = embedder.embed_batch(texts)
        except Exception as exc:  # noqa: BLE001 — 去重降级: 补齐失败不强依赖, 缺失者直接保留
            logger.warning("redundancy_reduce 补齐 embedding 失败, 跳过相似度判定: %s", str(exc)[:120])
            return embs
        for slot, v in zip(missing_idx, vecs):
            embs[slot] = np.asarray(v, dtype=np.float32)
    out = []
    for e in embs:
        if e is None:
            out.append(None)
        else:
            n = float(np.linalg.norm(e))
            out.append(e / n if n > 1e-9 else e)
    return out


def redundancy_reduce(docs: list[dict], embedder=None, threshold: float = 0.92,
                      max_per_doc: int = 1) -> list[dict]:
    """贪心去重: 按输入顺序 (已 RRF 排序, 降序) 保留首个, 剔除与其
    同 doc_id 已达上限 或 余弦相似度 >= threshold 的后续候选。

    返回保留列表 (保持原顺序, 长度 <= len(docs))。阈值对中文 base 嵌入取 0.92:
    既能剔除"实体替换"型近重复 (cos≈0.9), 又不误伤语义不同的相关文档。
    """
    if not docs:
        return []
    embs = _embeddings_of(docs, embedder)
    kept: list[dict] = []
    kept_embs: list[np.ndarray] = []
    kept_doc_ids: dict[str, int] = {}

    for d, e in zip(docs, embs):
        did = d.get("doc_id")
        if did is not None and kept_doc_ids.get(did, 0) >= max_per_doc:
            continue
        # threshold<=0: 关闭相似度去重 (仅靠同文档归并)。原因 (P0-1 实测):
        # 在 rerank 之前做相似度去重会"保 RRF 先到者、删其后相似者"; 而近重复干扰项
        # 往往靠共享原词在 RRF 里排在 golden 之前, 导致 golden 被提前删掉 → Recall 暴跌。
        # 相似度去重仅在确认你的语料近重复不会反超 golden 时才启用 (见 eval --dedup-threshold)。
        if e is not None and threshold > 0:
            dup = False
            for ke in kept_embs:
                if float(e @ ke) >= threshold:
                    dup = True
                    break
            if dup:
                continue
        kept.append(d)
        if e is not None:
            kept_embs.append(e)
        if did is not None:
            kept_doc_ids[did] = kept_doc_ids.get(did, 0) + 1
    return kept


def pool_density(documents: list[dict], threshold: float = 0.90, mode: str = "docs") -> float:
    """候选池近重复密度 (0=候选彼此语义各异, 1=全是近重复)。

    mode:
      docs  —— 有至少一个近重复伙伴的候选占比 (对"少数几个近重复簇"更敏感)
      pairs —— cos>=threshold 的候选对占比 (对"整体相似性"更敏感, 实测更优, 默认)

    零模型调用: 复用候选自带的 embedding (向量库检索结果本就带)。

    ⚠️ 缺 embedding 的候选 (仅被 BM25 召回、不在向量 top-k 里的文档, 实测约占 15% 的查询) 一律
    **填零参与分母**, 不丢弃。理由: 分母必须等于候选池真实规模 k*(k-1), 丢弃会系统性抬高密度、
    压低 alpha, 导致生产行为偏离 scripts/tune_alpha.py 网格搜索出的参数 (两者口径必须一致)。
    填零 = "无法判定相似, 视为不相似", 是保守方向。
    """
    n = len(documents)
    if n < 2:
        return 0.0
    embs = _embeddings_of(documents, None)
    dim = next((e.shape[0] for e in embs if e is not None), 0)
    if dim == 0:
        return 0.0
    mat = np.stack([e if e is not None else np.zeros(dim, dtype=np.float32) for e in embs])
    ge = (mat @ mat.T) >= threshold
    if mode == "pairs":
        cnt = int(ge.sum() - n)  # 去掉对角线自相似
        return cnt / float(n * (n - 1))
    has_partner = (ge.sum(axis=1) - 1) > 0
    return float(has_partner.mean())


def adaptive_alpha(documents: list[dict], threshold: float = 0.90, mode: str = "docs",
                   alpha_max: float = 0.90, alpha_min: float = 0.50, density_full: float = 0.20) -> float:
    """按候选池密度给出 rerank 融合权重 alpha。

        alpha = alpha_max - (alpha_max - alpha_min) * clamp(density / density_full, 0, 1)

    直觉: 池子越"干净" -> alpha 越高, 越信任 Cross-Encoder 的语义精排 (吃满精度上限);
          池子越"近重复密集" -> alpha 越低, 越依赖检索分托底 (避免 CE 被实体替换项迷惑)。

    该映射由 scripts/tune_alpha.py 在 clean / hard 双语料上网格搜索得出 (见 alpha_tuning.json)。
    """
    d = pool_density(documents, threshold=threshold, mode=mode)
    scale = min(max(d / density_full, 0.0), 1.0) if density_full > 0 else 1.0
    return alpha_max - (alpha_max - alpha_min) * scale
