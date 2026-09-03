"""查询嵌入增广 — 打开首阶段召回天花板 (HyDE / PRF)。

背景: 离线评测显示 Recall@10 触顶 ~97.9% 是首阶段召回天花板, 重排救不了; query 侧增广
是突破该天花板的唯一杠杆 (改写查询已用 LLM, 此处进一步用"更好的查询向量")。

两种策略 (均作用于**向量检索腿**; BM25 腿始终用原始查询, 不污染关键词信号):
  - prf : Pseudo-Relevance Feedback (伪相关反馈)。离线可用, 评测可量化。首轮向量检索 top-k,
          取其块向量与查询向量加权平均作为新查询向量, 再做正式检索。
  - hyde: Hypothetical Document Embeddings。用 LLM 生成"假设答案文档"并嵌入, 与查询向量
          加权融合。需 LLM; 任何失败上层回退原查询向量 (绝不阻断检索)。

本模块只做纯 numpy 向量运算, 不依赖 embedding/LLM, 便于单测与评测/生产复用。
"""

from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def prf_augment_vector(
    q_emb: np.ndarray,
    feedback_embs: list[np.ndarray] | np.ndarray,
    weight: float = 0.5,
) -> np.ndarray:
    """PRF: 用首轮 top-k 反馈向量与查询向量加权融合。

    q_emb: 归一化 1D (dim,)。feedback_embs: 归一化向量列表 [(dim,)] 或 (M, dim) 矩阵。
    weight=0 或空反馈 → 原样返回 (无操作, 零回归)。返回归一化 1D。
    """
    if weight <= 0.0:
        return q_emb
    if feedback_embs is None:
        return q_emb
    if isinstance(feedback_embs, list):
        if not feedback_embs:
            return q_emb
        mat = np.stack([np.asarray(e, dtype=float) for e in feedback_embs])
    else:
        mat = np.asarray(feedback_embs, dtype=float)
        if mat.size == 0:
            return q_emb
    fb = _normalize(mat.mean(axis=0))
    q = np.asarray(q_emb, dtype=float)
    new = (1.0 - weight) * q + weight * fb
    return _normalize(new)


def blend_vectors(q_emb: np.ndarray, other_emb: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """HyDE: 将假设文档向量 (other_emb, 须先归一化) 与查询向量加权融合。

    等价于单条反馈的 PRF。返回归一化 1D。
    """
    return prf_augment_vector(q_emb, [np.asarray(other_emb, dtype=float)], weight)
