"""QA 质量共享原语 — 离线评估与运行时指标的单一实现 (OPT-C4)。

此前拒答识别 regex / 余弦相似度 / jieba 词重合近似在 scripts/evaluate.py 与
api/core/qa_metrics.py 各有一份, 且已出现漂移: 评估侧的词重合漏了单字否定词
保留, "不支持七天无理由"与"支持七天无理由"的重合率完全相同。抽到这里统一维护,
双方只调用、不再自带实现。

纯函数、零 api 层依赖, 可独立单测。
"""

import math
import re

import jieba

# 拒答识别 (护栏/置信度下限触发的标准拒答话术 + LLM 主动声明无法回答)。
# 拒答不是幻觉: 把它混进 hallucination_rate 会虚高 (8 条拒答可把幻觉率从 1.5% 抬到 9.3%),
# 且掩盖真正的问题 —— 过度拒答 (检索已命中却拒答) 与路由失败 (检索根本没命中才拒答)。
REJECT_RE = re.compile(r"未找到|无法回答|相关度过低|不匹配|换一种表述|无法确认|未提及")

# 单字否定词: 长度 1 但语义权重极高, 必须参与重合计算
NEGATIONS = frozenset({"不", "无", "未", "非", "没", "否"})


def is_refusal(answer: str) -> bool:
    """答案是否命中标准拒答话术。"""
    return bool(REJECT_RE.search(answer or ""))


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度, 输入为已归一化向量时等价于点积。空/不等长向量返回 0.0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def word_overlap_faithfulness(answer: str, contexts: list[str]) -> float:
    """答案词与检索上下文词的重合率 (0-1, 越高越忠实)。

    单字否定词 (不/无/未/非/没/否) 虽长度为 1 但语义权重极高, 显式保留,
    否则"支持"与"不支持"的重合率完全相同。
    """
    if not answer:
        return 0.0
    ans_tokens = {t for t in jieba.cut(answer) if len(t.strip()) >= 2 or t.strip() in NEGATIONS}
    if not ans_tokens:
        return 0.0
    ctx_text = " ".join(c for c in contexts if c)
    ctx_tokens = (
        {t for t in jieba.cut(ctx_text) if len(t.strip()) >= 2 or t.strip() in NEGATIONS} if ctx_text else set()
    )
    if not ctx_tokens:
        return 0.0
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)
