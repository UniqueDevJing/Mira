"""QA 质量近似指标 — 纯函数, 与编排/状态解耦, 可独立单测。

- faithfulness: 答案 vs 检索上下文的忠实度 (幻觉护栏信号)
  - 主信号: 词重合率 (jieba, 便宜, 同步)
  - 辅信号: embedding 余弦相似度 (可注入 embed_fn, 默认 None 不加载模型)
  - 仅当 embedding 相似度达 EMBED_FAITHFUL_MIN 才抬升分数, 避免弱相关文本绕过护栏
- calc_qa_metrics: 综合近似质量指标 (无 ground truth, 见 scripts/evaluate.py 离线评估)

从 orchestrator 下沉, 使其脱离 async/state 可被纯单测覆盖。
"""

import logging
import math

import jieba

logger = logging.getLogger(__name__)

# embedding 相似度低于此值时不参与抬升忠实度: 弱相关/纯话题相关文本不会绕过词重合护栏
EMBED_FAITHFUL_MIN = 0.6


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度, 输入为已归一化向量时等价于点积。空向量返回 0.0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _word_overlap_faithfulness(answer: str, contexts: list[str]) -> float:
    """答案词与检索上下文词的重合率 (0-1, 越高越忠实)。"""
    if not answer:
        return 0.0
    ans_tokens = {t for t in jieba.cut(answer) if len(t.strip()) >= 2}
    if not ans_tokens:
        return 0.0
    ctx_text = " ".join(c for c in contexts if c)
    ctx_tokens = {t for t in jieba.cut(ctx_text) if len(t.strip()) >= 2} if ctx_text else set()
    if not ctx_tokens:
        return 0.0
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)


def _embedding_similarity(answer: str, contexts: list[str], embed_fn) -> float:
    """答案与各上下文的 embedding 余弦相似度最大值 (取最匹配片段)。

    embed_fn 失败 (模型不可用/异常) 返回 0.0, 让护栏回退到词重合信号。
    """
    try:
        a = embed_fn(answer)
    except Exception:  # noqa: BLE001 — 嵌入不可用时降级, 不阻断护栏
        return 0.0
    if not a:
        return 0.0
    best = 0.0
    for c in contexts:
        if not c:
            continue
        try:
            ce = embed_fn(c)
        except Exception as e:  # noqa: BLE001 — 单片段失败不影响其余
            logger.debug("上下文片段嵌入失败, 跳过: %s", str(e)[:80])
            continue
        if not ce:
            continue
        sim = _cosine(a, ce)
        best = max(best, sim)
    return max(0.0, best)


def _faithfulness(answer: str, contexts: list[str], embed_fn=None) -> float:
    """答案与检索上下文的忠实度 (0-1, 越高幻觉风险越低)。

    词重合近似对同义改写偏保守 (易被误判拒答)。当注入 embed_fn 且 embedding 相似度
    达 EMBED_FAITHFUL_MIN 时, 取两者较大值抬升分数, 避免语义一致但措辞不同的答案被误拒;
    弱相关文本相似度低于阈值, 不覆盖词重合护栏 (仍可能判为无依据)。
    """
    word_faith = _word_overlap_faithfulness(answer, contexts)
    if embed_fn is None:
        return word_faith
    emb_faith = _embedding_similarity(answer, contexts, embed_fn)
    if emb_faith >= EMBED_FAITHFUL_MIN:
        return max(word_faith, emb_faith)
    return word_faith


def _calc_qa_metrics(answer: str, contexts: list[str], top1: float, retrieval_rounds: int = 1) -> dict:
    """运行时评估近似指标 (无 ground truth, 标注为近似质量):
    - faithfulness: 词重合 + 可选 embedding 语义信号 (越高幻觉风险越低)
    - retrieval_relevance: 检索相关性 (top1 归一化近似)
    - confidence: 综合置信度
    真实准确率/召回率需离线标注评估 (见 scripts/evaluate.py)。
    """
    m = {"top1_score": round(top1, 4), "retrieval_rounds": retrieval_rounds}
    if not answer:
        m.update({"faithfulness": 0.0, "retrieval_relevance": 0.0, "confidence": 0.0})
        return m
    faith = _faithfulness(answer, contexts)
    rel = min(1.0, max(0.0, top1))
    m["faithfulness"] = round(faith, 4)
    m["retrieval_relevance"] = round(rel, 4)
    m["confidence"] = round(0.6 * faith + 0.4 * rel, 4)
    return m
