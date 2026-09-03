"""QA 质量近似指标 — 纯函数, 与编排/状态解耦, 可独立单测。

- faithfulness: 答案 vs 检索上下文的忠实度 (幻觉护栏信号)
  - 主信号: 词重合率 (jieba, 便宜, 同步)
  - 辅信号: embedding 余弦相似度 (可注入 embed_fn, 默认 None 不加载模型)
  - 仅当 embedding 相似度达 EMBED_FAITHFUL_MIN 才抬升分数, 避免弱相关文本绕过护栏
- calc_qa_metrics: 综合近似质量指标 (无 ground truth, 见 scripts/evaluate.py 离线评估)

从 orchestrator 下沉, 使其脱离 async/state 可被纯单测覆盖。
"""

import logging
import re

from engines.common.quality import (
    cosine as _cosine_impl,
)
from engines.common.quality import (
    word_overlap_faithfulness as _word_overlap_impl,
)

logger = logging.getLogger(__name__)

# embedding 相似度低于此值时不参与抬升忠实度: 弱相关/纯话题相关文本不会绕过词重合护栏
EMBED_FAITHFUL_MIN = 0.6

# ───────────────── 数字型幻觉护栏 ─────────────────
# 原 _word_overlap_faithfulness 用 len(token) >= 2 过滤, 把所有单字符 token 全部丢弃,
# 于是:
#   · 所有阿拉伯数字被丢弃 → "退款需 7 天" 与 "退款需 3 天" 词重合完全相同(满分)
#   · 单字否定词被丢弃     → "不支持七天无理由" 与 "支持七天无理由" 忠实度都是 1.0
# 而 finance 类型的 prompt_hint 恰恰要求"数字须与原文一致"。
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# 单字否定词统一走 engines/common/quality.NEGATIONS (OPT-C4)
# 中文字数字 + 量词才认定为数字, 避免把"一定""一起"里的"一"误当数字
_MEASURE_WORDS = "个天周月年小时分钟秒元角分次条款项种人件台套份倍成点%"
_CN_NUM_RE = re.compile(f"[零〇一两二三四五六七八九十]{{1,4}}(?=[{_MEASURE_WORDS}])")
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
              "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(s: str) -> int | None:
    """解析简单中文数字 (0~99, 形如 三 / 十五 / 二十三); 歧义/非数字片段返回 None。

    仅接受"十"两侧各至多 1 位数字的简单形式。像"十三五"(规划名)、"五六十"(约数)、
    "一百三十五"这类复杂/歧义片段一律返回 None —— 宁可少抽一个中文数字(保守),
    也不冒崩溃或误拒风险。阿拉伯数字走 _NUMBER_RE, 不受此限制。
    """
    if not s:
        return None
    if "十" in s:
        hi_s, _, lo_s = s.partition("十")
        if len(hi_s) > 1 or len(lo_s) > 1:
            return None
        if any(c not in _CN_DIGITS for c in hi_s + lo_s):
            return None
        return (_CN_DIGITS[hi_s] if hi_s else 1) * 10 + (_CN_DIGITS[lo_s] if lo_s else 0)
    # 无"十": 仅接受单位数中文数字 (多位数如"三五"歧义, 跳过)
    if len(s) != 1 or s not in _CN_DIGITS:
        return None
    return _CN_DIGITS[s]


def _extract_numbers(text: str) -> set[str]:
    """抽取数字 (阿拉伯数字 + 带量词的中文数字), 统一成字符串集合便于子集比较。"""
    if not text:
        return set()
    nums = set(_NUMBER_RE.findall(text))
    for m in _CN_NUM_RE.findall(text):
        try:
            v = _cn_to_int(m)
        except Exception:  # noqa: BLE001 — 单片段解析异常不影响其余数字提取
            v = None
        if v is not None:
            nums.add(str(v))
    return nums


def _number_supported(answer: str, contexts: list[str]) -> bool:
    """答案中的数字必须是上下文数字的子集, 否则判定为数字型幻觉。

    两种情形主动放行 (不判幻觉), 避免误拒:
      1. 答案无数字 — 无数字可比对
      2. 上下文无数字 — 无法建立基准。典型场景: 上下文写"共有三种方式"、
         答案写"共 3 种方式", 此时强行比对只会误伤。

    剥离 LLM 自动追加的引用标记 (如 [来源1][来源2]), 防止把源标签当数字提取。
    """
    # 剥除 [来源N] / [来源 N] 等引用标记 (LLM 在答案末尾追加)
    import re as _re
    answer = _re.sub(r'\[\s*来源\s*\d+\s*\]', '', answer)
    ans_nums = _extract_numbers(answer)
    if not ans_nums:
        return True
    ctx_nums = _extract_numbers(" ".join(c for c in contexts if c))
    if not ctx_nums:
        return True
    return ans_nums <= ctx_nums


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度 — 统一实现见 engines/common/quality.py (OPT-C4)。"""
    return _cosine_impl(a, b)


def _word_overlap_faithfulness(answer: str, contexts: list[str]) -> float:
    """答案词与检索上下文词的重合率 — 统一实现见 engines/common/quality.py (OPT-C4)。"""
    return _word_overlap_impl(answer, contexts)


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


def _faithfulness(answer: str, contexts: list[str], embed_fn=None, check_numbers: bool = True) -> float:
    """答案与检索上下文的忠实度 (0-1, 越高幻觉风险越低)。

    词重合近似对同义改写偏保守 (易被误判拒答)。当注入 embed_fn 且 embedding 相似度
    达 EMBED_FAITHFUL_MIN 时, 取两者较大值抬升分数, 避免语义一致但措辞不同的答案被误拒;
    弱相关文本相似度低于阈值, 不覆盖词重合护栏 (仍可能判为无依据)。

    check_numbers: 先过一道数字硬校验 —— 答案出现上下文中不存在的数字时直接判 0.0。
    词重合看不见数字 (len<2 被过滤), 这道校验专门堵"退款 7 天 vs 3 天"这类数字型幻觉。
    可用 RAG_FIDELITY_CHECK_NUMBERS=false 关闭 (若误拒过多)。
    """
    if check_numbers and not _number_supported(answer, contexts):
        logger.debug("忠实度护栏: 答案含上下文中不存在的数字, 判定数字型幻觉")
        return 0.0
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
