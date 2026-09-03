"""Skill 执行 + 编排入口 (从 orchestrator 拆分)。

编排层 — Router 分发 + Skill 执行 + 分阶段超时/降级。

用户方案:
  全局 5s 计时 → Router(规则≥0.85直通 / LLM 分类 1.5s / fallback)
  → RAG Skill: Embedding(缓存+重试) + 向量|BM25 并行(0.8s) → RRF 融合
               → top1<阈值 跨库兜底(并行, 预算内) → Rerank(0.5s, 超时跳过=L1)
               → LLM 生成(2s, 失败→检索摘要=L3)
  → 直接回答 Skill: 仅 LLM

降级等级: 0 正常 / 1 rerank 跳过 / 2 向量失败仅 BM25 / 3 LLM 失败返回摘要。
各阶段降级另由 rag_degradation_stage_total{stage} 独立计数 (一次查询可同时命中多阶段,
等级维度只记最高级, 会丢失中间阶段; 阶段维度与等级维度互补, 见 degradation 模块)。
所有阈值/超时读 Settings (env RAG_* 可调)。
"""

import asyncio
import copy
import logging
import time

from api.config import settings
from api.core.auth import KBForbiddenError
from api.core.degradation import (
    _deg_bump,
    _deg_record_level,
    _deg_record_stage_only,
    _deg_reset,
)
from api.core.guardrails import (
    _guard_faithfulness,
    _pregeneration_hallucination_guard_async,
)
from api.core.llm_client import (
    CircuitBreakerOpenError,
    LLMClient,
    SyncLLMClient,
    get_llm_client,
)
from api.core.metrics import (
    qa_cache_hits_total,
    qa_cache_misses_total,
    qa_cache_semantic_hits_total,
    qa_faithfulness,
    qa_latency_seconds,
    qa_requests_total,
    qa_top1_score,
    retrieved_docs_count,
    track_llm_latency,
)
from api.core.metrics import (
    retrieval_rounds as retrieval_rounds_hist,
)
from api.core.qa_cache import get_qa_cache
from api.core.response import (
    build_done_event,
    build_qa_result,
    build_refusal_info,
    build_sources_event,
    calc_qa_metrics,
    public_sources,
)
from api.core.retrieval import (
    _apply_deferred_rerank,
    _dedupe_docs,
    _report_embed_cache,
    _retrieve_context,
    _retrieve_fanout,
)
from api.core.routing import (
    _candidate_kbs,
    _chat_messages,
    _direct_messages,
    _remaining,
    _route,
    _should_fanout,
)
from api.core.session_store import load_session, save_session
from api.schemas.qa import ChatTurn
from engines.router.intent_router import RoutingResult

logger = logging.getLogger(__name__)


def _record_qa_quality(result: dict) -> None:
    """把 QA 质量信号上报为 Prometheus 指标 + 结构化质量日志。

    填补 /metrics 看不到"回答靠不靠谱"的空白; 质量日志经 JsonFormatter 自动带 trace_id,
    可按 trace_id 串联一次请求的全链路。上报失败仅 debug, 不阻断回答。
    """
    try:
        m = result.get("qa_metrics") or {}
        if "faithfulness" in m:
            qa_faithfulness.observe(float(m["faithfulness"]))
        tm = result.get("retrieval_meta") or {}
        if "top1_score" in tm:
            qa_top1_score.observe(float(tm["top1_score"]))
        logger.info(
            "qa_quality",
            extra={
                "structured": {
                    "kb": result.get("kb_id"),
                    "skill": result.get("skill"),
                    "degradation": result.get("degradation_level", 0),
                    "faithfulness": m.get("faithfulness"),
                    "top1_score": tm.get("top1_score"),
                }
            },
        )
    except Exception as e:  # noqa: BLE001 — 质量上报失败不影响回答
        logger.debug("QA 质量上报失败: %s", str(e)[:80])


async def ask(
    question: str,
    skill: str | None = None,
    top_k: int = 10,
    enable_self_retrieval: bool = False,
    temperature: float = 0.1,
    mode: str = "hybrid",
    history=None,
    session_id=None,
    allowed_kbs=None,
    owner: str | None = None,
) -> dict:
    """编排入口: 缓存命中直接返回 → 路由 → skill 执行 → 组装响应。

    allowed_kbs: principal 可访问的知识库集合(None=不限制); 自动路由命中非授权库时抛 KBForbiddenError。
    owner: session 归属者 key_id (S6), 会话历史写入时绑定, 供 IDOR 校验。
    """
    start = time.time()

    # 会话解析: 携带 session_id 时服务端按 session 维护历史(覆盖 body.history, 刷新/换设备不丢)
    effective_history = load_session(session_id) if session_id else history

    candidate_kbs = _candidate_kbs(allowed_kbs)

    # QA 结果缓存: 相同输入指纹命中则跳过路由+检索+LLM, 直接返回缓存
    # S1 安全: 缓存键含 allowed_kbs 作用域(RBAC 分片), 命中后仍做 kb_id 复检, 防越权读
    cache = get_qa_cache() if settings.qa_cache_enabled else None
    cache_key = (
        cache.make_key(
            question, skill, top_k, enable_self_retrieval, temperature, mode, effective_history, allowed_kbs
        )
        if cache is not None
        else None
    )
    cache_scope = (
        cache.make_scope(
            question, skill, top_k, enable_self_retrieval, temperature, mode, effective_history, allowed_kbs
        )
        if cache is not None
        else None
    )
    if cache_key is not None:
        hit_stats: dict = {}
        hit = cache.get(cache_key, question=question, scope=cache_scope, stats=hit_stats)
        if hit is not None:
            hit_kb = hit.get("kb_id")
            if allowed_kbs is None or hit_kb is None or hit_kb in set(allowed_kbs):
                qa_cache_hits_total.inc()
                if hit_stats.get("kind") == "semantic":
                    qa_cache_semantic_hits_total.inc()
                qa_latency_seconds.observe(time.time() - start)
                result = copy.deepcopy(hit)  # 深拷贝: 缓存条目含嵌套 dict, 浅拷贝修改会污染缓存
                result["cache_hit"] = True
                qa_requests_total.labels(mode=mode, status="cache_hit").inc()
                # 会话持久化: 缓存命中也记录本轮(刷新/换设备不丢上下文); 否则重复问题会丢失多轮链路
                if session_id:
                    save_session(
                        session_id,
                        list(effective_history)
                        + [
                            ChatTurn(role="user", content=question),
                            ChatTurn(role="assistant", content=result.get("answer", "")),
                        ],
                        owner=owner,
                    )
                return result
        qa_cache_misses_total.inc()

    llm = get_llm_client()
    try:
        routing, candidates, router_ms = await _route(question, skill, llm, start, candidate_kbs)
    except CircuitBreakerOpenError:
        logger.warning("LLM 熔断, 降级为纯检索 (无路由)")
        routing = RoutingResult("tech", "tech", 0.0, "circuit_open")
        candidates = [routing]
        router_ms = 0.0

    # RBAC: 自动/兜底路由命中了 principal 无权访问的库 → 拒绝 (路由层转 403)
    if allowed_kbs is not None and routing.kb is not None and routing.kb not in set(allowed_kbs):
        raise KBForbiddenError(f"无权访问知识库: {routing.kb}")

    if routing.skill == "direct":
        result = await _skill_direct(question, llm, routing, start, temperature, effective_history)
    else:
        # P1' 选择性扇出: 判据见 _should_fanout —— 主候选模糊, 或次选与主候选旗鼓相当
        # (如退换货同时命中 policy/service) 时扇出; 归属明确时单路, 不白增一倍检索延迟。
        fanout_routings = [
            r for r in candidates[: settings.route_fanout_top_n] if r.kb and r.kb != routing.kb
        ]
        if _should_fanout(routing, fanout_routings):
            logger.info(
                "[%s] 扇出 %d 个候选 KB (主候选 conf=%.2f, 次选 %s): %s",
                routing.kb, len(fanout_routings) + 1, routing.confidence,
                [r.confidence for r in fanout_routings],
                [r.kb for r in ([routing] + fanout_routings)],
            )
            precomputed = await _retrieve_fanout(
                question, [routing] + fanout_routings, top_k, start,
                enable_self_retrieval, mode, candidate_kbs,
            )
            result = await _skill_rag(
                question, routing, llm, top_k, start,
                enable_self_retrieval, temperature, mode, effective_history,
                allowed_kbs=allowed_kbs, precomputed_retr=precomputed,
            )
        else:
            result = await _skill_rag(
                question,
                routing,
                llm,
                top_k,
                start,
                enable_self_retrieval,
                temperature,
                mode,
                effective_history,
                allowed_kbs=allowed_kbs,
            )

    # 先组装完整 result (router_ms/qa_metrics) 再写缓存 — 原 cache.set 在 qa_metrics 计算前执行,
    # 缓存条目永久缺 qa_metrics, 命中路径与 miss 路径契约不一致
    # router_ms / qa_metrics 在入口统一填充: _skill_rag 不知道路由耗时(占位 0.0);
    # qa_metrics 也在这里算, 因为 answer 可能在降级分支被改写, 入口处才是最终值
    result.setdefault("latency_breakdown", {})["router_ms"] = round(router_ms, 1)
    result["qa_metrics"] = calc_qa_metrics(
        answer=result.get("answer", ""),
        docs=result.get("sources", []),
        top1_score=result.get("retrieval_meta", {}).get("top1_score", 0.0),
        retrieval_rounds=result.get("retrieval_rounds", 1),
    )
    qa_latency_seconds.observe(time.time() - start)
    qa_requests_total.labels(
        mode=mode, status="fallback" if result.get("degradation_level", 0) >= 3 else "success"
    ).inc()

    if cache_key is not None:
        cached = copy.deepcopy(result)
        cached["cache_hit"] = False
        cache.set(cache_key, cached, settings.qa_cache_ttl_s, question=question, scope=cache_scope)

    # 会话持久化: 服务端按 session 维护多轮历史(覆盖 body.history)
    if session_id:
        save_session(
            session_id,
            list(effective_history)
            + [ChatTurn(role="user", content=question), ChatTurn(role="assistant", content=result.get("answer", ""))],
            owner=owner,
        )

    _record_qa_quality(result)
    _report_embed_cache()
    return result


async def _skill_rag(
    question: str,
    routing: RoutingResult,
    llm: "LLMClient | SyncLLMClient",
    top_k: int,
    start: float,
    enable_self_retrieval: bool = False,
    temperature: float = 0.1,
    mode: str = "hybrid",
    history=None,
    allowed_kbs=None,
    precomputed_retr: dict | None = None,
) -> dict:
    _deg_reset()
    kb = routing.kb
    candidate_kbs = _candidate_kbs(allowed_kbs)
    # P1' 扇出: 命中 precomputed_retr 时跳过内部检索(检索已在 _retrieve_fanout 完成并合并)
    if precomputed_retr is not None:
        retr = precomputed_retr
    else:
        retr = await _retrieve_context(
            question, routing, top_k, start, enable_self_retrieval, mode=mode, candidate_kbs=candidate_kbs
        )
    docs = retr["docs"]
    context = retr["context"]
    degradation = retr["degradation"]
    retrieval_ms = retr["retrieval_ms"]
    rerank_ms = retr["rerank_ms"]
    top1_score = retr["top1_score"]
    cross_kb_kbs = retr["cross_kb_kbs"]
    retrieval_rounds = retr.get("retrieval_rounds", 1)
    rewritten_queries = retr.get("rewritten_queries", [])
    graph_ctx = retr.get("graph_context")

    # 图谱实体关系文本拼入 LLM prompt — 补充检索文档外的多跳语义 (原实现丢弃, 多跳结果不进 context)
    if graph_ctx and graph_ctx.get("graph_context"):
        graph_txt = "\n".join(f"- {g}" for g in graph_ctx["graph_context"][:8])
        context = f"{context}\n\n图谱实体关系:\n{graph_txt}"

    # 生成前幻觉护栏 (幻觉前置 #4): 置信度下限 / 上下文零重合 → 直接拒答, 不基于无关上下文生成
    pre_reason = await _pregeneration_hallucination_guard_async(question, docs, top1_score)
    if pre_reason:
        logger.warning("[%s] 生成前护栏触发(%s), 拒答防幻觉", kb, pre_reason)
        degradation = _deg_bump(degradation, 3, "llm")
        answer = (
            "知识库中未找到与您问题高度相关的内容（检索到的最佳片段相关度过低或与问题不匹配）。\n"
            "请确认是否已上传相关文档，或换一种表述方式后重试。"
        )
        # U1 拒答分级: 附候选来源(检索命中但被拦截) + 引导追问, 替代纯文本死胡同
        refusal_info = build_refusal_info(pre_reason, docs, question, kb)
        _deg_record_level(degradation)
        retrieval_rounds_hist.observe(retrieval_rounds)
        retrieved_docs_count.observe(len(docs))
        return build_qa_result(
            answer=answer,
            docs=docs,
            routing=routing,
            kb=kb,
            retrieval_rounds=retrieval_rounds,
            rewritten_queries=rewritten_queries,
            graph_context=graph_ctx,
            token_usage=None,
            degradation_level=degradation,
            router_ms=0.0,  # 占位: 由 ask() 覆盖为真实路由耗时
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            llm_ms=0.0,
            total_ms=(time.time() - start) * 1000,
            top1_score=top1_score,
            cross_kb_kbs=cross_kb_kbs,
            refusal=refusal_info,
        )

    answer, token_usage, llm_ms, llm_ok = await _generate(question, context, llm, start, temperature, history, kb=kb)
    track_llm_latency.observe(llm_ms / 1000)
    if not llm_ok:
        degradation = _deg_bump(degradation, 3, "llm")
        if docs:
            answer = "（LLM 暂时不可用）以下为检索到的相关内容：\n" + "\n".join(
                f"· {d.get('content', '')[:200]}" for d in _dedupe_docs(docs[:3])
            )
            refusal_info = None  # 降级成功: 已附检索内容, 非拒答
        else:
            answer = "未在知识库中找到相关信息，请先上传文档。"
            refusal_info = build_refusal_info("no_docs", [], question, kb)
    else:
        # 忠实度护栏: 生成内容与检索上下文词重合过低 → 判定无依据, 拒答防编造
        if docs and await _guard_faithfulness(answer, docs) < settings.fidelity_threshold:
            logger.warning("[%s] 忠实度护栏触发: 答案与上下文重合 < %.2f, 改为拒答", kb, settings.fidelity_threshold)
            degradation = _deg_bump(degradation, 3, "llm")
            answer = (
                "知识库中未找到足以可靠回答该问题的内容（可能与现有文档不匹配）。\n"
                "以下为检索到的相关片段供参考：\n" + "\n".join(f"· {d.get('content', '')[:150]}" for d in docs[:3])
            )
            # U1 拒答分级: 忠实度护栏拦截(检索命中但答案无依据), 附候选 + 引导
            refusal_info = build_refusal_info("low_fidelity", docs, question, kb)
        else:
            refusal_info = None

    _deg_record_level(degradation)
    retrieval_rounds_hist.observe(retrieval_rounds)
    retrieved_docs_count.observe(len(docs))

    return build_qa_result(
        answer=answer,
        docs=docs,
        routing=routing,
        kb=kb,
        retrieval_rounds=retrieval_rounds,
        rewritten_queries=rewritten_queries,
        graph_context=graph_ctx,
        token_usage=token_usage,
        degradation_level=degradation,
        router_ms=0.0,  # 占位: 由 ask() 覆盖为真实路由耗时
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        llm_ms=llm_ms,
        total_ms=(time.time() - start) * 1000,
        top1_score=top1_score,
        cross_kb_kbs=cross_kb_kbs,
        refusal=refusal_info,
    )


async def _generate(
    question: str, context: str, llm: "LLMClient | SyncLLMClient",
    start: float, temperature: float = 0.1, history=None, kb: str | None = None
):
    """调用 LLM 生成答案。返回 (answer, token_usage_dict|None, ms, ok)。"""
    t0 = time.time()
    if not context:
        return "未在知识库中找到相关信息，请先上传文档。", None, 0.0, False
    try:
        resp = await asyncio.wait_for(
            llm.chat(
                messages=_chat_messages(context, question, history, kb=kb),
                temperature=temperature,
                max_tokens=2000,
            ),
            timeout=_remaining(start, settings.llm_generate_timeout_s),
        )
        ms = (time.time() - t0) * 1000
        token_usage = {
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "total_tokens": resp.total_tokens,
            "llm_latency_ms": round(resp.latency_ms, 1),
        }
        answer = resp.content or "（推理中）增加 max_tokens 后重试"
        logger.info("LLM 生成完成: total=%d, latency=%.1fms", resp.total_tokens, ms)
        return answer, token_usage, ms, True
    except (TimeoutError, CircuitBreakerOpenError) as e:
        logger.warning("LLM 生成降级 L3: %s", type(e).__name__)
        return "", None, (time.time() - t0) * 1000, False
    except Exception as e:  # noqa: BLE001 — 降级边界: LLM 生成失败 (L3)
        logger.error("LLM 生成失败: %s", str(e)[:200])
        return "", None, (time.time() - t0) * 1000, False


# ────────────────────────── 流式编排 ──────────────────────────


async def _replay_cache_stream(hit: dict):
    """命中缓存时按流式协议重放: meta → sources → delta → done。"""
    yield {
        "type": "meta",
        "skill": hit.get("skill", ""),
        "kb_id": hit.get("kb_id"),
        "routing_source": hit.get("routing_source", ""),
        "router_ms": hit.get("latency_breakdown", {}).get("router_ms", 0.0),
    }
    yield {
        "type": "sources",
        "sources": public_sources(hit.get("sources", [])),
        "retrieval_meta": hit.get("retrieval_meta", {}),
    }
    yield {"type": "delta", "content": hit.get("answer", "")}
    yield {
        "type": "done",
        "answer": hit.get("answer", ""),
        "token_usage": hit.get("token_usage"),
        "degradation_level": hit.get("degradation_level", 0),
        "latency_breakdown": hit.get("latency_breakdown", {}),
        "retrieval_meta": hit.get("retrieval_meta", {}),
        "qa_metrics": hit.get("qa_metrics", {}),
        "cache_hit": True,
    }


async def ask_stream(
    question: str,
    skill: str | None = None,
    top_k: int = 10,
    enable_self_retrieval: bool = False,
    temperature: float = 0.1,
    mode: str = "hybrid",
    history=None,
    session_id=None,
    allowed_kbs=None,
    owner: str | None = None,
):
    """流式编排入口: 缓存命中重放 → 路由 → 检索 → LLM 逐块产出。yield SSE 事件 dict。

    事件协议:
      - {"type": "meta", skill, kb_id, routing_source}
      - {"type": "sources", sources, retrieval_meta}
      - {"type": "delta", content}
      - {"type": "done", answer, token_usage, degradation_level,
         latency_breakdown, retrieval_meta}
    """
    start = time.time()

    # 会话解析: 携带 session_id 时服务端按 session 维护历史(覆盖 body.history)
    effective_history = load_session(session_id) if session_id else history
    candidate_kbs = _candidate_kbs(allowed_kbs)

    cache = get_qa_cache() if settings.qa_cache_enabled else None
    cache_key = (
        cache.make_key(
            question, skill, top_k, enable_self_retrieval, temperature, mode, effective_history, allowed_kbs
        )
        if cache is not None
        else None
    )
    cache_scope = (
        cache.make_scope(
            question, skill, top_k, enable_self_retrieval, temperature, mode, effective_history, allowed_kbs
        )
        if cache is not None
        else None
    )
    if cache_key is not None:
        hit_stats: dict = {}
        hit = cache.get(cache_key, question=question, scope=cache_scope, stats=hit_stats)
        if hit is not None:
            hit_kb = hit.get("kb_id")
            if allowed_kbs is None or hit_kb is None or hit_kb in set(allowed_kbs):
                qa_cache_hits_total.inc()
                if hit_stats.get("kind") == "semantic":
                    qa_cache_semantic_hits_total.inc()
                qa_latency_seconds.observe(time.time() - start)
                qa_requests_total.labels(mode=mode, status="cache_hit").inc()
                async for ev in _replay_cache_stream(hit):
                    yield ev
                # 会话持久化: 缓存命中也记录本轮(与 ask() 非流式路径保持一致)
                if session_id:
                    save_session(
                        session_id,
                        list(effective_history)
                        + [
                            ChatTurn(role="user", content=question),
                            ChatTurn(role="assistant", content=hit.get("answer", "")),
                        ],
                        owner=owner,
                    )
                return
        qa_cache_misses_total.inc()

    llm = get_llm_client()
    try:
        routing, candidates, router_ms = await _route(question, skill, llm, start, candidate_kbs)
    except CircuitBreakerOpenError:
        logger.warning("LLM 熔断, 降级为纯检索 (无路由)")
        routing = RoutingResult("tech", "tech", 0.0, "circuit_open")
        candidates = [routing]
        router_ms = 0.0

    # RBAC: 自动/兜底路由命中非授权库 → 拒绝 (路由层转 403)
    if allowed_kbs is not None and routing.kb is not None and routing.kb not in set(allowed_kbs):
        raise KBForbiddenError(f"无权访问知识库: {routing.kb}")

    # P1' 选择性扇出: 判据见 _should_fanout (与 ask 非流式链路共用, 避免两处逻辑漂移)
    precomputed_retr = None
    fanout_candidates = (
        [r for r in candidates[: settings.route_fanout_top_n] if r.kb and r.kb != routing.kb]
        if routing.skill != "direct"
        else []
    )
    if _should_fanout(routing, fanout_candidates):
        logger.info(
            "[%s] 扇出 %d 个候选库检索 (主候选 conf=%.2f, 次选 %s): %s",
            routing.kb, len(fanout_candidates) + 1, routing.confidence,
            [r.confidence for r in fanout_candidates],
            [r.kb for r in ([routing] + fanout_candidates)],
        )
        precomputed_retr = await _retrieve_fanout(
            question, [routing] + fanout_candidates, top_k, start,
            enable_self_retrieval, mode, candidate_kbs=candidate_kbs,
        )

    yield {
        "type": "meta",
        "skill": routing.skill,
        "kb_id": routing.kb,
        "routing_source": routing.source,
        "router_ms": round(router_ms, 1),
    }

    cached = {
        "skill": routing.skill,
        "kb_id": routing.kb,
        "routing_source": routing.source,
        "latency_breakdown": {"router_ms": round(router_ms, 1)},
    }

    gen = (
        _stream_direct(question, llm, routing, start, router_ms, temperature, effective_history)
        if routing.skill == "direct"
        else _stream_rag(
            question,
            routing,
            llm,
            top_k,
            start,
            enable_self_retrieval,
            temperature,
            mode,
            effective_history,
            precomputed_retr=precomputed_retr,
            allowed_kbs=allowed_kbs,
        )
    )
    async for ev in gen:
        if ev["type"] == "sources":
            cached["sources"] = ev.get("sources", [])
            cached["retrieval_meta"] = ev.get("retrieval_meta", {})
        elif ev["type"] == "done":
            # router_ms 统一在入口填充: _stream_rag 不知道路由耗时(占位 0.0),
            # _stream_direct 已知真实值(幂等覆盖)。修正前此处由 _stream_rag 自行
            # 计算 (time.time()-start)*1000, 那是 total_ms 而非路由耗时。
            ev.setdefault("latency_breakdown", {})["router_ms"] = round(router_ms, 1)
            cached["answer"] = ev.get("answer", "")
            cached["token_usage"] = ev.get("token_usage")
            cached["degradation_level"] = ev.get("degradation_level", 0)
            cached["latency_breakdown"] = ev.get("latency_breakdown", cached["latency_breakdown"])
            cached["retrieval_meta"] = ev.get("retrieval_meta", cached.get("retrieval_meta", {}))
            cached["qa_metrics"] = ev.get("qa_metrics", {})
            if cache_key is not None:
                cache.set(cache_key, dict(cached), settings.qa_cache_ttl_s, question=question, scope=cache_scope)
            # 会话持久化: 服务端按 session 维护多轮历史(覆盖 body.history)
            if session_id:
                save_session(
                    session_id,
                    list(effective_history)
                    + [
                        ChatTurn(role="user", content=question),
                        ChatTurn(role="assistant", content=cached.get("answer", "")),
                    ],
                    owner=owner,
                )
        yield ev

    qa_latency_seconds.observe(time.time() - start)
    qa_requests_total.labels(
        mode=mode, status="fallback" if cached.get("degradation_level", 0) >= 3 else "success"
    ).inc()
    _record_qa_quality(cached)
    _report_embed_cache()


def _run_stream_llm(llm, messages, temperature: float, max_tokens: int, start: float):
    """构造流式生成器。返回 (gen, result); 耗尽 gen 后 result 填充
    (answer/token_usage/ok/llm_ms), 供调用方组装 done 事件。

    流式受全局预算约束 (asyncio.timeout 包整段迭代): 超时截断但已产出的 delta 保留,
    防 httpx 默认 60s 悬挂 (原实现无 wait_for, 上游卡死则用户等满 60s)。
    """
    result = {}

    async def _gen():
        answer_parts, token_usage, llm_ok = [], None, True
        t0 = time.time()
        try:
            async with asyncio.timeout(_remaining(start, settings.llm_generate_timeout_s)):
                stream = llm.stream_chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
                async for ev in stream:
                    if ev["type"] == "delta":
                        answer_parts.append(ev["content"])
                        yield ev["content"]
                    elif ev["type"] == "usage":
                        usage = ev["usage"]
                        token_usage = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                            "llm_latency_ms": round((time.time() - t0) * 1000, 1),
                        }
        except (TimeoutError, CircuitBreakerOpenError) as e:
            logger.warning("流式 LLM 生成降级 L3: %s", type(e).__name__)
            llm_ok = False
        except Exception as e:  # noqa: BLE001 — 降级边界: 流式 LLM 失败 (L3)
            logger.error("流式 LLM 生成失败: %s", str(e)[:200])
            llm_ok = False
        result["answer"] = "".join(answer_parts)
        result["token_usage"] = token_usage
        result["ok"] = llm_ok
        result["llm_ms"] = (time.time() - t0) * 1000
        track_llm_latency.observe(result["llm_ms"] / 1000)

    return _gen(), result


async def _stream_rag(
    question: str,
    routing: RoutingResult,
    llm,
    top_k: int,
    start: float,
    enable_self_retrieval: bool = False,
    temperature: float = 0.1,
    mode: str = "hybrid",
    history=None,
    precomputed_retr: dict | None = None,
    allowed_kbs=None,
):
    """流式 RAG Skill: 检索 → 发 sources → 流式 LLM 生成。"""
    _deg_reset()
    candidate_kbs = _candidate_kbs(allowed_kbs)
    # defer 必须在两个分支之前都有值: 它只在 else 分支内赋值, 而下方 `if defer` 无条件
    # 访问 —— 扇出路径 (precomputed_retr) 走 if 分支时曾命中 UnboundLocalError。
    # 扇出复用的是 ask 层已合并的多库结果, 没有"重排前先发临时来源"的延迟需要掩盖, 故恒 False。
    defer = False
    if precomputed_retr is not None:
        # P1' 扇出: 复用 ask 层已合并的多库检索结果, 跳过内部单库检索
        retr = precomputed_retr
    else:
        # P0-2: 检索一完成(~50ms)先把"重排前"的来源推给前端, 消除 rerank(~700ms)期间的白屏。
        # 答案生成仍等 rerank 完成 —— 上下文用重排后的 docs, 引用不会错。
        # 重排未生效(rerank_effective_enabled=False)时, 没有 rerank 延迟可掩盖, 直接发最终 sources (省一次无效事件)。
        from api.state import rerank_effective_enabled

        defer = settings.stream_sources_before_rerank and rerank_effective_enabled()
        retr = await _retrieve_context(
            question, routing, top_k, start, enable_self_retrieval, mode=mode,
            candidate_kbs=candidate_kbs, defer_rerank=defer,
        )
    if defer:
        yield build_sources_event(retr["docs"], retr["top1_score"], retr["cross_kb_kbs"],
                                 retr["degradation"], final=False)
        retr = await _apply_deferred_rerank(retr, routing.kb, question, top_k, start)

    docs = retr["docs"]
    context = retr["context"]
    degradation = retr["degradation"]
    cross_kb_kbs = retr["cross_kb_kbs"]
    top1_score = retr["top1_score"]

    # 重排完成, 发最终 sources (前端整体替换, 覆盖上面的临时来源)
    yield build_sources_event(docs, top1_score, cross_kb_kbs, degradation)

    # 生成前幻觉护栏 (幻觉前置 #4): 置信度下限 / 上下文零重合 → 直接拒答, 不流式生成错误答案
    pre_reason = await _pregeneration_hallucination_guard_async(question, docs, top1_score)
    if pre_reason:
        refusal = (
            "知识库中未找到与您问题高度相关的内容（检索到的最佳片段相关度过低或与问题不匹配）。\n"
            "请确认是否已上传相关文档，或换一种表述方式后重试。"
        )
        logger.warning("[%s] 生成前护栏触发(%s), 拒答防幻觉", routing.kb, pre_reason)
        degradation = _deg_bump(degradation, 3, "llm")
        # U1 拒答分级: 流式首个拒答原因, 附候选来源 + 引导追问
        refusal_info = build_refusal_info(pre_reason, docs, question, routing.kb)
        _deg_record_level(degradation)
        yield {"type": "delta", "content": refusal}
        yield build_done_event(
            answer=refusal,
            docs=docs,
            token_usage=None,
            degradation_level=degradation,
            router_ms=0.0,  # 占位: 由 ask_stream() 覆盖为真实路由耗时
            retrieval_ms=retr["retrieval_ms"],
            rerank_ms=retr["rerank_ms"],
            llm_ms=0.0,
            total_ms=(time.time() - start) * 1000,
            top1_score=top1_score,
            cross_kb_kbs=cross_kb_kbs,
            retrieval_rounds=retr.get("retrieval_rounds", 1),
            refusal=refusal_info,
        )
        return

    # 流式 LLM 生成
    gen, r = _run_stream_llm(
        llm,
        messages=_chat_messages(context, question, history, kb=routing.kb),
        temperature=temperature,
        max_tokens=2000,
        start=start,
    )
    async for delta in gen:
        yield {"type": "delta", "content": delta}
    answer, token_usage, llm_ok, llm_ms = r["answer"], r["token_usage"], r["ok"], r["llm_ms"]

    refusal_info = None
    if not llm_ok:
        degradation = _deg_bump(degradation, 3, "llm")
        if docs:
            fallback = "（LLM 暂时不可用）以下为检索到的相关内容：\n" + "\n".join(
                f"· {d.get('content', '')[:200]}" for d in _dedupe_docs(docs[:3])
            )
            refusal_info = None  # 降级成功: 已附检索内容, 非拒答
        else:
            fallback = "未在知识库中找到相关信息，请先上传文档。"
            refusal_info = build_refusal_info("no_docs", [], question, routing.kb)
        answer = fallback
        yield {"type": "delta", "content": fallback}
    elif docs and await _guard_faithfulness(answer, docs) < settings.fidelity_threshold:
        # 忠实度护栏 (流式): 答案已流式发送, 无法撤回, 追加拒答提示 + 相关片段
        logger.warning("[%s] 流式忠实度护栏触发: 答案与上下文重合 < %.2f", routing.kb, settings.fidelity_threshold)
        degradation = _deg_bump(degradation, 3, "llm")
        warning = "\n\n⚠️ 以上回答与知识库内容匹配度较低，请谨慎参考。相关片段：\n" + "\n".join(
            f"· {d.get('content', '')[:150]}" for d in docs[:3]
        )
        answer = answer + warning
        # U1 拒答分级: 流式忠实度拦截, 附候选来源 + 引导追问(前端从 done 事件取 refusal)
        refusal_info = build_refusal_info("low_fidelity", docs, question, routing.kb)
        yield {"type": "delta", "content": warning}

    _deg_record_level(degradation)
    retrieval_rounds_hist.observe(retr.get("retrieval_rounds", 1))
    retrieved_docs_count.observe(len(docs))

    yield build_done_event(
        answer=answer,
        docs=docs,
        token_usage=token_usage,
        degradation_level=degradation,
        router_ms=0.0,  # 占位: 由 ask_stream() 覆盖为真实路由耗时
        retrieval_ms=retr["retrieval_ms"],
        rerank_ms=retr["rerank_ms"],
        llm_ms=llm_ms,
        total_ms=(time.time() - start) * 1000,
        top1_score=top1_score,
        cross_kb_kbs=cross_kb_kbs,
        retrieval_rounds=retr.get("retrieval_rounds", 1),
        refusal=refusal_info,
    )


async def _stream_direct(
    question: str,
    llm,
    routing: RoutingResult,
    start: float,
    router_ms: float = 0.0,
    temperature: float = 0.1,
    history=None,
):
    """流式直接回答 Skill: 仅 LLM, 无检索。"""
    _deg_reset()
    gen, r = _run_stream_llm(
        llm,
        messages=_direct_messages(question, history),
        temperature=temperature,
        max_tokens=500,
        start=start,
    )
    async for delta in gen:
        yield {"type": "delta", "content": delta}
    answer, token_usage, llm_ok, llm_ms = r["answer"], r["token_usage"], r["ok"], r["llm_ms"]

    if not llm_ok:
        answer = "（LLM 暂时不可用，无法直接回答。）"
        token_usage = None
        _deg_record_stage_only("llm")

    yield build_done_event(
        answer=answer,
        docs=[],
        token_usage=token_usage,
        degradation_level=3 if not llm_ok else 0,
        router_ms=router_ms,  # _stream_direct 签名含 router_ms, 直接用真实值
        retrieval_ms=0.0,
        rerank_ms=0.0,
        llm_ms=llm_ms,
        total_ms=(time.time() - start) * 1000,
        top1_score=0.0,
        cross_kb_kbs=[],
        retrieval_rounds=0,
    )


# ────────────────────────── 直接回答 Skill ──────────────────────────


async def _skill_direct(
    question: str,
    llm: "LLMClient | SyncLLMClient",
    routing: RoutingResult,
    start: float,
    temperature: float = 0.1,
    history=None,
) -> dict:
    _deg_reset()
    t0 = time.time()
    try:
        resp = await asyncio.wait_for(
            llm.chat(
                messages=_direct_messages(question, history),
                temperature=temperature,
                max_tokens=500,
            ),
            timeout=_remaining(start, settings.llm_generate_timeout_s),
        )
        answer = resp.content or "（无回复）"
        token_usage = {
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "total_tokens": resp.total_tokens,
            "llm_latency_ms": round(resp.latency_ms, 1),
        }
    except Exception as e:  # noqa: BLE001 — 降级边界: 直接回答 LLM 失败
        logger.warning("直接回答 LLM 失败: %s", str(e)[:120])
        answer = "（LLM 暂时不可用，无法直接回答。）"
        token_usage = None
    llm_ms = (time.time() - t0) * 1000
    track_llm_latency.observe(llm_ms / 1000)
    no_llm = token_usage is None
    if no_llm:
        _deg_record_stage_only("llm")

    return build_qa_result(
        answer=answer,
        docs=[],
        routing=routing,
        kb=None,
        retrieval_rounds=0,
        rewritten_queries=[],
        graph_context=None,
        token_usage=token_usage,
        degradation_level=3 if no_llm else 0,
        router_ms=0.0,  # 占位: 由 ask() 覆盖为真实路由耗时
        retrieval_ms=0.0,
        rerank_ms=0.0,
        llm_ms=llm_ms,
        total_ms=(time.time() - start) * 1000,
        top1_score=0.0,
        cross_kb_kbs=[],
    )
