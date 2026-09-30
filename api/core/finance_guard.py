"""金融数值校验护栏 —— 把 engines.finance 的确定性校验接进 RAG 出口。

为什么需要第二道出口护栏
------------------------
已有的忠实度护栏回答的是「答案有没有检索依据」（词重合 + 可选语义信号），
偏统计、有容错；但金融场景的典型故障是**数字被改写**——"1.2 亿元" 写成
"1.2 万元"、"营收" 说成 "利润"、把去年数当今年报。这类错误的词重合度很高，
忠实度护栏抓不到，而财务场景错一个数就是真金白银。

所以这里补一道纯规则的数值对账：答案里的每个数字都必须能由来源数字
（含量级归一、币种/百分比语义区分）支撑或复算得出，否则判 fail。

三重闸门（避免拖累非金融问答）
------------------------------
1. 开关打开（settings.finance_guard_enabled，默认开）；
2. 问题被规则判定属于金融/财务域（零 LLM 成本、微秒级）；
3. 检索到了文档（无来源则无从对账）。

裁决映射
--------
fail → 拒答（附原因）。金融场景给错数字比拒答代价高得多。
warn → 放行 + 附加提示（时效过期 / 同源分部差异），宁提示勿误拒。
pass → 放行，零打扰。

故障策略
--------
任何异常一律放行（返回 None），护栏自身故障不得阻断回答 —— 与忠实度护栏一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from api.config import settings
from engines.finance import Source, is_finance_question, verify_finance_answer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinanceGuardOutcome:
    """一次金融校验的结果（只保留调用方需要的字段）。"""

    verdict: str  # pass / warn / fail
    summary: str
    findings: tuple[str, ...] = ()


def _as_of_from_doc(doc: dict) -> datetime | None:
    """从文档元数据取「数据时效点」。

    优先级：显式 as_of（文档自带的报告期/发布日）→ update_time（入库/更新时刻）
    → created_at。取不到返回 None —— 校验引擎会把「未知时效」记为 warn 级提示，
    不会因此拒答（保守：不因缺元数据误伤正常回答）。

    注意口径差异：as_of 表示「该来源陈述的事实成立于何时」（如 2025 年报的报告期），
    而 update_time 只表示「文档何时入库」——前者才真正决定时效裁决。若入库时
    能解析出报告期，建议写入 as_of。
    """
    explicit = doc.get("as_of")
    if explicit:
        if isinstance(explicit, datetime):
            return explicit
        try:
            return datetime.fromisoformat(str(explicit))
        except ValueError:
            pass
    raw = doc.get("update_time") or doc.get("created_at")
    if not raw:
        return None
    try:
        ts = float(raw)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    if ts > 1e11:  # 毫秒时间戳
        ts /= 1000.0
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def docs_to_sources(docs: list[dict]) -> list[Source]:
    """把检索结果映射成校验引擎的来源列表。

    两个字段的口径很关键（引擎的跨源矛盾分级依赖它们）：
      name  —— 必须**每个 chunk 唯一**，否则同一文件切成多块后，
               块间差异会被「同名跳过」规则整体漏检；
      group —— 文件级身份（标题/文件名），用来区分
               「同文件不同块」（warn）与「不同文件真矛盾」（fail）。
    """
    sources: list[Source] = []
    for idx, d in enumerate(docs or []):
        title = (d.get("doc_title") or d.get("file_name") or d.get("name") or "来源").strip()
        text = d.get("context_full") or d.get("content") or ""
        if not text:
            continue
        sources.append(
            Source(
                name=f"{title}#{idx}",
                text=text,
                as_of=_as_of_from_doc(d),
                group=d.get("file_name") or d.get("doc_title") or None,
            )
        )
    return sources


def should_verify(question: str, docs: list[dict]) -> bool:
    """三重闸门：开关 / 金融域 / 有来源。"""
    if not settings.finance_guard_enabled:
        return False
    if not docs:
        return False
    try:
        return is_finance_question(question or "")
    except Exception as e:  # noqa: BLE001 — 意图检测故障时保守不启用
        logger.warning("金融意图检测失败, 跳过金融校验: %s", str(e)[:100])
        return False


def run_finance_guard(question: str, answer: str, docs: list[dict]) -> FinanceGuardOutcome | None:
    """执行金融数值校验。不满足启用条件或校验异常时返回 None（放行）。"""
    if not should_verify(question, docs):
        return None
    try:
        sources = docs_to_sources(docs)
        if not sources:
            return None
        verdict = verify_finance_answer(
            answer or "",
            sources,
            max_age=__import__("datetime").timedelta(days=settings.finance_guard_max_age_days),
        )
        outcome = FinanceGuardOutcome(
            verdict=verdict.verdict,
            summary=verdict.summary(),
            findings=tuple(f.message for f in verdict.findings),
        )
        if outcome.verdict != "pass":
            logger.info(
                "金融校验护栏: verdict=%s findings=%d", outcome.verdict, len(outcome.findings)
            )
        return outcome
    except Exception as e:  # noqa: BLE001 — 护栏故障一律放行
        logger.warning("金融校验护栏执行失败, 放行: %s", str(e)[:120])
        return None


def refusal_text(outcome: FinanceGuardOutcome) -> str:
    """fail 时的对外文案（不用模型的自由文本，避免二次幻觉）。"""
    detail = "\n".join(f"· {m}" for m in outcome.findings[:3])
    body = (
        "该问题涉及数值，校验未通过 —— 为避免给出错误数字，暂不直接回答。\n"
        f"校验结论：{outcome.summary}\n"
    )
    if detail:
        body += f"{detail}\n"
    return body + "请核对知识库中的原始文档，或补充来源后重试。"


def notice_text(outcome: FinanceGuardOutcome) -> str:
    """warn 时的附加提示。"""
    return f"\n\n（数值校验提示：{outcome.summary}）"
