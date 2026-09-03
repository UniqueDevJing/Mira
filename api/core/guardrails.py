"""生成前幻觉护栏 / 忠实度护栏 (从 orchestrator 拆分)。"""

import asyncio
import logging

from api.config import settings
from api.core.qa_metrics import (
    _cosine,
    _faithfulness,
    _word_overlap_faithfulness,
)
from api.state import get_embedder

logger = logging.getLogger(__name__)


def _faithfulness_guard(answer: str, docs: list[dict]) -> float:
    """忠实度护栏求值: 词重合 + 可选 embedding 语义信号。

    embed_fn 在 settings.fidelity_use_embedding 为真时由全局 embedder 单例注入;
    单例加载/推理失败则退回纯词重合 (与旧行为一致), 不阻断护栏。
    """
    # context_full = LLM 实际看到的完整文本(800字); content 只是 200 字显示片段,
    # 用片段判忠实度会低估 → 误拒答 (与 evaluate.py 判分器同源问题)
    contexts = [d.get("context_full") or d.get("content", "") for d in docs]
    embed_fn = None
    if settings.fidelity_use_embedding:
        try:
            embed_fn = get_embedder().embed_query
        except Exception:  # noqa: BLE001 — 嵌入不可用时降级为词重合护栏
            embed_fn = None
    return _faithfulness(
        answer, contexts, embed_fn=embed_fn, check_numbers=settings.fidelity_check_numbers
    )


async def _guard_faithfulness(answer: str, docs: list[dict]) -> float:
    """护栏求值卸载到线程池 (embedding 推理可能阻塞事件循环), wait_for 限时。

    超时/异常视为放行 (返回 1.0), 不因护栏自身故障阻断回答。
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(_faithfulness_guard, answer, docs), timeout=2.0)
    except TimeoutError:
        logger.warning("忠实度护栏求值超时, 放行")
        return 1.0
    except Exception as e:  # noqa: BLE001
        logger.warning("忠实度护栏求值失败, 放行: %s", str(e)[:100])
        return 1.0


def _preguard_embed_fn():
    """low_relevance 语义判定的 embed 工厂(独立函数便于测试 monkeypatch)。

    加载失败返回 None → 纯词重合判定(与 embed 依赖同源, 系统级故障时维持原逻辑)。
    """
    try:
        return get_embedder().embed_query
    except Exception as e:  # noqa: BLE001
        logger.warning("低重合语义判定 embed 不可用, 回退纯词重合判定: %s", str(e)[:100])
        return None


def _pregeneration_hallucination_guard(
    question: str, docs: list[dict], top1_score: float, embed_fn=None
) -> str | None:
    """生成前幻觉护栏 (幻觉前置 #4): 返回拒答 reason 或 None。

    两条生成链路 (_skill_rag / _stream_rag) 共用, 保证行为一致、不分化。

    - low_confidence: 最佳来源分数低于 answer_confidence_floor → 无关上下文, 不生成(杜绝答非所问)。
      始终生效, 是 #4「检索置信度下限」的核心前置护栏。
    - low_relevance (受 answerability_preguard_enabled 控制, 默认开启): 分数达标但上下文与问题
      **零内容词重合**(检索到的是无关/关键词堆砌文档, 高分局外片段), 生成必然幻觉 → 前置拒答。

      零词重合对同义改写("取消订单" vs 上下文"退订")敏感, 会误拒语义等价改写 —— 因此启用时
      必须注入 embed_fn 做语义兜底: 零重合 且 语义相似度 < settings.low_relevance_min_sim 才判定
      真无关并拒答; 相似度达标视为语义改写, 放行。embed 失败 → 保守放行(零重合信号不可靠),
      不因护栏自身故障误伤。离线评估依据见 api/config.py answerability_preguard_enabled 注释。
    """
    if docs and top1_score < settings.answer_confidence_floor:
        return "low_confidence"
    if settings.answerability_preguard_enabled and docs:
        contexts = [d.get("content", "") or d.get("parent_content", "") or "" for d in docs]
        if contexts and _word_overlap_faithfulness(question, contexts) == 0.0:
            if embed_fn is not None:
                # 语义兜底: 零词重合 ≠ 无关。question 嵌入失败 → 保守放行(词重合信号不可靠,
                # 宁让后置护栏兜底也不因护栏自身故障误伤); 片段失败仅跳过该片段。
                try:
                    a = embed_fn(question)
                except Exception as e:  # noqa: BLE001
                    logger.warning("低重合语义判定 embed 失败, 放行: %s", str(e)[:100])
                    return None
                best = 0.0
                if a:
                    for c in contexts:
                        if not c:
                            continue
                        try:
                            ce = embed_fn(c)
                        except Exception as e:  # noqa: BLE001 — 单片段 embed 失败跳过, 不影响其余
                            logger.debug("低重合语义判定: 片段 embed 失败, 跳过: %s", str(e)[:80])
                            continue
                        if ce:
                            best = max(best, _cosine(a, ce))
                if best >= settings.low_relevance_min_sim:
                    return None  # 语义等价改写, 非无关
            return "low_relevance"
    return None


async def _pregeneration_hallucination_guard_async(question: str, docs: list[dict], top1_score: float) -> str | None:
    """生成前护栏异步封装: embedding 推理卸载到线程池(防阻塞事件循环), wait_for 限时。

    超时/异常视为放行, 不因护栏自身故障阻断回答。仅 low_relevance 分支需要 embed,
    且只在零词重合的罕见情形下才会触发实际推理(常规路径零额外延迟)。
    """
    embed_fn = _preguard_embed_fn() if settings.answerability_preguard_enabled else None
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_pregeneration_hallucination_guard, question, docs, top1_score, embed_fn),
            timeout=2.0,
        )
    except TimeoutError:
        logger.warning("生成前护栏求值超时, 放行")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("生成前护栏求值失败, 放行: %s", str(e)[:100])
        return None
