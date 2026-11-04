"""M1 — 错误账本 + 幻觉归因。

把每次 RAG 问答的可疑样本(badcase)自动归入 5 类, 给出确定性归因规则 + 修复动作,
形成"改进闭环"。分类结果写入三处:
  1) 进程内环形缓冲 (最近 N 条, 供 summary()/trend() 与单测断言)
  2) 本地 JSONL 文件 (settings.m1_ledger_path, 可持久化、可离线分析)
  3) Prometheus 计数器 rag_m1_badcase_total{cls}

五类定义与优先级(互斥, 取首个命中):
  no_recall          未召回      —— 检索 0 命中(docs 为空)或拒答原因 no_docs。
  false_refusal      误拒答      —— 触发了拒答, 但召回其实相关(top1 高 / 问答词面重合达 gate), 疑似误杀。
  number_rewritten   数字被改写  —— 金融校验 fail, 或答案数字不是上下文数字子集(数字型幻觉)。
  low_rank           召回排序低  —— docs 非空、未拒答, 但 top1_score < m1_low_rank_threshold。
  context_truncated  上下文被截  —— 喂给 LLM 的某段上下文被 800 字截断(Truncated 标记), 且非上述更严重的类。

返回 None 表示本次问答未被判定为 badcase(不入账)。归因准确率由离线标注集(390 问)回测,
不在运行时断言; summary()/trend() 提供每类占比与趋势, 供"badcase 分类归因表"导出。
"""
from __future__ import annotations

import json
import logging
import threading
from collections import Counter, deque
from datetime import UTC, date, datetime

from api.config import settings
from prometheus_client import Counter as PromCounter

logger = logging.getLogger(__name__)

# ── 五类英文键 ──
NO_RECALL = "no_recall"
LOW_RANK = "low_rank"
CONTEXT_TRUNCATED = "context_truncated"
NUMBER_REWRITTEN = "number_rewritten"
FALSE_REFUSAL = "false_refusal"

CLASS_LABELS = {
    NO_RECALL: "未召回",
    LOW_RANK: "召回但排序低",
    CONTEXT_TRUNCATED: "上下文被截",
    NUMBER_REWRITTEN: "数字被改写",
    FALSE_REFUSAL: "误拒答",
}

# 每类的(自动归因规则, 修复动作)。写入账本记录 + 归因表。
CLASS_FIX = {
    NO_RECALL: (
        "检索/索引缺口：问题相关文档未被召回（0 命中或仅 no_docs 拒答）。",
        "① 检查 chunk 切分粒度与 embedding 是否覆盖该问法；② 补 BM25/同义词召回；③ 确认知识点已入库。",
    ),
    LOW_RANK: (
        "召回存在但 top1 相关性低：rerank 未把正确片段顶到首位，或 embedding 对该领域区分度不足。",
        "① 调 rerank 阈值/权重；② 提高 top_k 召回深度；③ 评估该领域 embedding 质量。",
    ),
    CONTEXT_TRUNCATED: (
        "喂给 LLM 的上下文被 800 字/段截断，答案所在片段被切掉（Truncated 标记）。",
        "① 提高单段截断上限或从父块取更完整上下文；② 关键字段前置（attention 重排已做）；③ 评估是否扩 context window。",
    ),
    NUMBER_REWRITTEN: (
        "答案出现上下文中不存在的数字（数字型幻觉），或金融数值校验 fail。",
        "① 强化金融数值校验护栏（已 fail 拦截）；② 答案数字强制回链来源；③ 数字走确定性比对而非依赖 LLM。",
    ),
    FALSE_REFUSAL: (
        "护栏误杀：召回其实相关却因零词重合/低分被拒答（low_relevance/low_confidence/low_fidelity 触发）。",
        "① 放宽 low_relevance 语义下限或加 synonym 兜底；② 高 top1 直接放行；③ 用离线集回测拒答率。",
    ),
}

_badcase_total = PromCounter(
    "rag_m1_badcase_total",
    "M1 错误账本 badcase 总数（按类别）",
    ["cls"],
)

# 进程内环形缓冲：默认上限, 可被 settings.m1_max_buffer 运行时调整(下次重启生效)。
_ledger: deque = deque(maxlen=2000)
_lock = threading.Lock()


def _contexts_of(docs) -> list[str]:
    return [(d.get("context_full") or d.get("content") or "") for d in (docs or [])]


def _max_question_overlap(question: str, contexts: list[str]) -> float:
    """问答词面重合的最大值（复用 qa_metrics 的词重合实现, 懒加载避免重依赖）。"""
    if not contexts or not question:
        return 0.0
    from api.core.qa_metrics import _word_overlap_faithfulness

    best = 0.0
    for c in contexts:
        if c:
            best = max(best, _word_overlap_faithfulness(question, [c]))
    return best


def classify(
    question: str,
    docs: list[dict],
    top1_score: float,
    answer: str,
    refusal_reason: str | None,
    finance_verdict: str | None,
    truncated: bool,
) -> tuple[str | None, str, str]:
    """返回 (class_key|None, reason, fix_action)。纯函数, 便于单测。

    优先级: no_recall > false_refusal > number_rewritten > low_rank > context_truncated。
    """
    contexts = _contexts_of(docs)

    # 1) 未召回
    if not docs or refusal_reason == "no_docs":
        return NO_RECALL, CLASS_FIX[NO_RECALL][0], CLASS_FIX[NO_RECALL][1]

    # 2) 误拒答：拒答但召回其实相关
    if refusal_reason in ("low_confidence", "low_relevance", "low_fidelity"):
        if (
            top1_score >= settings.m1_false_refusal_top1_gate
            or _max_question_overlap(question, contexts) >= settings.m1_false_refusal_overlap_gate
        ):
            return FALSE_REFUSAL, CLASS_FIX[FALSE_REFUSAL][0], CLASS_FIX[FALSE_REFUSAL][1]

    # 3) 数字被改写
    if finance_verdict == "fail":
        return NUMBER_REWRITTEN, CLASS_FIX[NUMBER_REWRITTEN][0], CLASS_FIX[NUMBER_REWRITTEN][1]
    if answer:
        from api.core.qa_metrics import _number_supported

        if not _number_supported(answer, contexts):
            return NUMBER_REWRITTEN, CLASS_FIX[NUMBER_REWRITTEN][0], CLASS_FIX[NUMBER_REWRITTEN][1]

    # 4) 召回但排序低
    if top1_score < settings.m1_low_rank_threshold:
        return LOW_RANK, CLASS_FIX[LOW_RANK][0], CLASS_FIX[LOW_RANK][1]

    # 5) 上下文被截
    if truncated:
        return CONTEXT_TRUNCATED, CLASS_FIX[CONTEXT_TRUNCATED][0], CLASS_FIX[CONTEXT_TRUNCATED][1]

    return None, "", ""


def record(question: str, kb: str, cls: str, reason: str, fix: str, meta: dict | None = None) -> dict | None:
    """写入账本：内存环形缓冲 + JSONL 文件 + Prometheus。返回记录 dict。"""
    if not cls:
        return None
    rec = {
        "ts": datetime.now(UTC).isoformat(),
        "date": date.today().isoformat(),
        "class": cls,
        "label": CLASS_LABELS.get(cls, cls),
        "reason": reason,
        "fix": fix,
        "kb": kb,
        "question": (question or "")[:300],
        "meta": meta or {},
    }
    try:
        _badcase_total.labels(cls=cls).inc()
    except Exception:  # noqa: BLE001 — 指标失败不影响主流程
        pass
    global _ledger
    with _lock:
        if _ledger.maxlen != settings.m1_max_buffer:
            _ledger = deque(_ledger, maxlen=settings.m1_max_buffer)
        _ledger.append(rec)
    if settings.m1_ledger_path:
        try:
            with open(settings.m1_ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.debug("M1 账本写文件失败: %s", str(e)[:80])
    return rec


def record_from_rag(
    question: str,
    kb: str,
    *,
    docs: list[dict],
    top1_score: float,
    answer: str,
    refusal_reason: str | None = None,
    finance_verdict: str | None = None,
    truncated: bool = False,
    extra: dict | None = None,
) -> dict | None:
    """从 RAG 两条链路(_skill_rag / _stream_rag)统一调用：分类 → 入账。返回记录或 None。"""
    if not settings.m1_enabled:
        return None
    cls, reason, fix = classify(
        question, docs, top1_score, answer, refusal_reason, finance_verdict, truncated
    )
    if not cls:
        return None
    meta = {
        "top1_score": round(float(top1_score), 4),
        "refusal_reason": refusal_reason,
        "finance_verdict": finance_verdict,
        "truncated": truncated,
    }
    if extra:
        meta.update(extra)
    return record(question, kb, cls, reason, fix, meta)


def summary() -> dict:
    """返回每类计数 + 总 badcase + 修复提示 + 最近样本, 供 /debug 与单测。"""
    with _lock:
        cnt = Counter(r["class"] for r in _ledger)
        recent = [r for r in list(_ledger)[-50:]]
    return {
        "total": sum(cnt.values()),
        "by_class": {k: cnt.get(k, 0) for k in CLASS_LABELS},
        "labels": CLASS_LABELS,
        "fix_hints": {k: CLASS_FIX[k][1] for k in CLASS_LABELS},
        "recent": recent,
    }


def trend() -> dict:
    """按日期 × 类别聚合, 供"每类占比趋势"。键为 date, 值为 {class: count}。"""
    with _lock:
        rows = list(_ledger)
    agg: dict[str, Counter] = {}
    for r in rows:
        agg.setdefault(r["date"], Counter())[r["class"]] += 1
    return {d: dict(c) for d, c in sorted(agg.items())}
