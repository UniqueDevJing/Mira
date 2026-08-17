"""编排层 — Router 分发 + Skill 执行 + 分阶段超时/降级。

用户方案:
  全局 5s 计时 → Router(规则≥0.85直通 / LLM 分类 1.5s / fallback)
  → RAG Skill: Embedding(缓存+重试) + 向量|BM25 并行(0.8s) → RRF 融合
               → top1<阈值 跨库兜底(并行, 预算内) → Rerank(0.5s, 超时跳过=L1)
               → LLM 生成(2s, 失败→检索摘要=L3)
  → 直接回答 Skill: 仅 LLM

降级等级: 0 正常 / 1 rerank 跳过 / 2 向量失败仅 BM25 / 3 LLM 失败返回摘要。
所有阈值/超时读 Settings (env RAG_* 可调)。
"""

import asyncio
import copy
import logging
import time

from api.config import settings
from api.core.llm_client import (
    CircuitBreakerOpenError,
    LLMClient,
    SyncLLMClient,
    get_llm_client,
    get_sync_llm_client,
)

# retrieval_rounds 别名: _skill_rag 内同名局部变量遮蔽指标对象
from api.core.metrics import (
    cross_kb_fallback_total,
    embed_cache_hits_total,
    embed_cache_misses_total,
    qa_cache_hits_total,
    qa_cache_misses_total,
    qa_faithfulness,
    qa_latency_seconds,
    qa_requests_total,
    qa_top1_score,
    retrieved_docs_count,
    track_degradation,
    track_llm_latency,
    track_rerank_latency,
    track_retrieval_latency,
    track_routing,
)
from api.core.metrics import (
    retrieval_rounds as retrieval_rounds_hist,
)
from api.core.qa_cache import get_qa_cache
from api.core.qa_metrics import _calc_qa_metrics, _faithfulness
from api.state import get_bm25_index, get_embedder, get_reranker, get_vector_store
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.evaluator import RetrievalEvaluator
from engines.retrieval.fusion import rrf_fuse
from engines.retrieval.hybrid_retriever import HybridRetriever
from engines.retrieval.query_preprocessor import preprocess_query
from engines.retrieval.query_rewriter import QueryRewriter
from engines.retrieval.self_retrieval import SelfRetrieval
from engines.router.intent_router import IntentRouter, RoutingResult
from engines.router.routing_rules import SKILLS

logger = logging.getLogger(__name__)


def _faithfulness_guard(answer: str, docs: list[dict]) -> float:
    """忠实度护栏求值: 词重合 + 可选 embedding 语义信号。

    embed_fn 在 settings.fidelity_use_embedding 为真时由全局 embedder 单例注入;
    单例加载/推理失败则退回纯词重合 (与旧行为一致), 不阻断护栏。
    """
    contexts = [d.get("content", "") for d in docs]
    embed_fn = None
    if settings.fidelity_use_embedding:
        try:
            embed_fn = get_embedder().embed_query
        except Exception:  # noqa: BLE001 — 嵌入不可用时降级为词重合护栏
            embed_fn = None
    return _faithfulness(answer, contexts, embed_fn=embed_fn)


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


# 超时/阈值动态读 settings (测试可改 settings 实时生效, 不冻结在导入时)

RAG_SYSTEM_PROMPT = """你是严谨的知识库助手，只能依据参考文档回答，严禁编造。
铁律：
1. 回答的每个事实必须能在参考文档中找到依据，逐句核对，禁止脑补
2. 参考文档没有的信息：明确回答"文档中未提及"，不得推测、编造或脑补
3. 引用内容标注来源（[来源N]），无法标注来源的表述一律不写
4. 只回答问题本身，不做额外扩展、总结或建议
5. 回答控制在 300 字以内
6. 若用户问题指代了前文（如"它""这个""那怎么办"），结合「对话历史」理解指代对象，不要当作全新孤立问题"""

RAG_KBS = [s["kb"] for s in SKILLS.values() if s["kb"]]  # ["service", "tech"]


def _history_to_messages(history) -> list[dict]:
    """多轮历史 → LLM messages（仅取 user/assistant 轮，截断到最近 20 轮防上下文膨胀）。

    兼容 pydantic ChatTurn 对象与裸 dict（便于测试）。非法角色/空内容跳过。
    """
    out: list[dict] = []
    for turn in (history or [])[-20:]:
        role = getattr(turn, "role", None) if not isinstance(turn, dict) else turn.get("role")
        content = getattr(turn, "content", None) if not isinstance(turn, dict) else turn.get("content")
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def _chat_messages(context: str, question: str, history=None) -> list[dict]:
    """组装 RAG 生成用 messages: system + 历史 + 当前(参考文档+问题)。"""
    return (
        [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        + _history_to_messages(history)
        + [{"role": "user", "content": f"参考文档：\n{context}\n\n问题：{question}"}]
    )


def _direct_messages(question: str, history=None) -> list[dict]:
    """组装 direct 技能 messages: system + 历史 + 当前(直接回答)。"""
    return (
        [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        + _history_to_messages(history)
        + [{"role": "user", "content": f"直接简洁回答用户问题：{question}"}]
    )


# Embedding 缓存命中统计上报（增量方式写入 Prometheus Counter）
_cache_stats_last = {"hits": 0, "misses": 0}


def _remaining(start: float, cap: float) -> float:
    """剩余预算: 不超过全局总预算。"""
    left = settings.total_timeout_s - (time.time() - start)
    return max(0.05, min(cap, left))


async def _route(
    question: str, skill: str | None, llm: "LLMClient | SyncLLMClient", start: float
) -> tuple[RoutingResult, float]:
    """路由: 手动指定 Skill 直通, 否则 LLM 分类。返回 (routing, router_ms)。"""
    if skill and skill in SKILLS:
        routing = RoutingResult(skill, SKILLS[skill]["kb"], 1.0, "manual")
    else:
        router = IntentRouter(llm_client=llm)
        routing = await router.route(question)
    track_routing.labels(source=routing.source, skill=routing.skill).inc()
    return routing, (time.time() - start) * 1000


async def ask(
    question: str,
    skill: str | None = None,
    top_k: int = 10,
    enable_self_retrieval: bool = False,
    temperature: float = 0.1,
    mode: str = "hybrid",
    history=None,
) -> dict:
    """编排入口: 缓存命中直接返回 → 路由 → skill 执行 → 组装响应。"""
    start = time.time()

    # QA 结果缓存: 相同输入指纹命中则跳过路由+检索+LLM, 直接返回缓存
    cache = get_qa_cache() if settings.qa_cache_enabled else None
    cache_key = (
        cache.make_key(question, skill, top_k, enable_self_retrieval, temperature, mode, history)
        if cache is not None
        else None
    )
    if cache_key is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            qa_cache_hits_total.inc()
            qa_latency_seconds.observe(time.time() - start)
            result = copy.deepcopy(hit)  # 深拷贝: 缓存条目含嵌套 dict, 浅拷贝修改会污染缓存
            result["cache_hit"] = True
            qa_requests_total.labels(mode=mode, status="cache_hit").inc()
            return result
        qa_cache_misses_total.inc()

    llm = get_llm_client()
    routing, router_ms = await _route(question, skill, llm, start)

    if routing.skill == "direct":
        result = await _skill_direct(question, llm, routing, start, temperature, history)
    else:
        result = await _skill_rag(
            question, routing, llm, top_k, start, enable_self_retrieval, temperature, mode, history
        )

    # 先组装完整 result (router_ms/qa_metrics) 再写缓存 — 原 cache.set 在 qa_metrics 计算前执行,
    # 缓存条目永久缺 qa_metrics, 命中路径与 miss 路径契约不一致
    result["latency_breakdown"]["router_ms"] = round(router_ms, 1)
    result["qa_metrics"] = _calc_qa_metrics(
        answer=result.get("answer", ""),
        contexts=[d.get("content", "") for d in result.get("sources", [])],
        top1=result.get("retrieval_meta", {}).get("top1_score", 0.0),
        retrieval_rounds=result.get("retrieval_rounds", 1),
    )
    qa_latency_seconds.observe(time.time() - start)
    qa_requests_total.labels(
        mode=mode, status="fallback" if result.get("degradation_level", 0) >= 3 else "success"
    ).inc()

    if cache_key is not None:
        cached = copy.deepcopy(result)
        cached["cache_hit"] = False
        cache.set(cache_key, cached, settings.qa_cache_ttl_s)

    _record_qa_quality(result)
    _report_embed_cache()
    return result


# ────────────────────────── RAG Skill ──────────────────────────


async def _retrieve_context(
    question: str,
    routing: RoutingResult,
    top_k: int,
    start: float,
    enable_self_retrieval: bool = False,
    mode: str = "hybrid",
) -> dict:
    """检索上下文: 预处理 → Embedding → 向量|BM25 → 图谱增强 → RRF → 跨库兜底 → Rerank。

    mode: hybrid(向量+BM25+图谱) / vector(纯向量) / graph(纯图谱)。
    供 `_skill_rag`（一次性）与 `ask_stream`（流式）共用, 避免检索管线重复。
    enable_self_retrieval=True 时走 Self-Retrieval 多轮循环（graph 模式为单轮, 忽略该参数）。
    返回 dict: docs/context/degradation/retrieval_ms/rerank_ms/top1_score/
               cross_kb_kbs/retrieval_rounds/rewritten_queries/graph_context。
    """
    if mode == "graph":
        return await _retrieve_graph_only(question, routing, top_k, start)
    if enable_self_retrieval:
        return await _retrieve_self(question, routing, top_k, start)

    kb = routing.kb
    embedder = EmbeddingService(model_name=settings.embedding_model, device=settings.embedding_device)
    retr_start = time.time()

    vq, bq = _preprocess_query(question, routing, kb)
    q_emb = await _embed_safe(embedder, vq, kb, start)
    vector_docs, bm25_docs, degradation = await _retrieve_safe(kb, q_emb, bq, top_k, start, mode)

    # 图谱增强 (hybrid 模式): 查询侧仅规则抽取, 不付 LLM 成本; 关联 chunk 按 id 回填参与 RRF/Rerank
    graph_ctx = None
    if mode == "hybrid":
        graph_ctx = await _graph_retrieve_safe(kb, question, top_k, start)
        if graph_ctx:
            fetched = get_vector_store(kb).get_by_ids(list(graph_ctx.get("source_chunks") or [])[:top_k])
            for gd in fetched:
                gd["source"] = "graph"
            vector_docs = vector_docs + fetched

    fused = rrf_fuse(vector_docs, bm25_docs)
    retrieval_ms = (time.time() - retr_start) * 1000
    track_retrieval_latency.observe(retrieval_ms / 1000)

    # top1 < 阈值 → 跨库兜底 (并行, 独立预算; 仅 hybrid 模式, 保持 vector/graph 语义纯净)
    cross_kb_kbs = []
    if mode == "hybrid" and fused and fused[0].get("score", 0.0) < settings.cross_kb_threshold:
        fused, cross_kb_kbs = await _cross_kb_fallback(fused, vq, bq, kb, top_k, start)
        for to_kb in cross_kb_kbs:
            cross_kb_fallback_total.labels(from_kb=kb, to_kb=to_kb).inc()

    # Rerank (超时/失败跳过 = L1)
    docs, rerank_ms, degradation = await _rerank_safe(kb, question, fused, top_k, start, degradation)

    return _build_retrieval_result(docs, top_k, degradation, retrieval_ms, rerank_ms, cross_kb_kbs, graph_ctx)


async def _graph_retrieve_safe(kb: str, question: str, top_k: int, start: float):
    """图谱检索 (带 0.5s 预算, 查询侧规则抽取零 LLM 成本)。失败/超时 → None, 不影响主检索。"""
    try:
        from api.state import get_graph_rag

        graph_rag = get_graph_rag(kb)
        ctx = await asyncio.wait_for(
            asyncio.to_thread(graph_rag.retrieve, question, top_k, True),
            timeout=_remaining(start, 0.5),
        )
        if not ctx.get("entities") and not ctx.get("source_chunks"):
            return None
        return ctx
    except TimeoutError:
        logger.debug("[%s] 图谱检索超时, 跳过", kb)
    except Exception as e:  # noqa: BLE001 — 图谱降级边界: 失败跳过
        logger.debug("[%s] 图谱检索失败, 跳过: %s", kb, str(e)[:100])
    return None


async def _retrieve_graph_only(question: str, routing: RoutingResult, top_k: int, start: float) -> dict:
    """纯图谱模式 (mode=graph): 规则抽取实体 → 多跳遍历 → 关联 chunk 回填 → Rerank。"""
    kb = routing.kb
    retr_start = time.time()
    ctx = await _graph_retrieve_safe(kb, question, top_k, start)
    docs = []
    if ctx:
        docs = get_vector_store(kb).get_by_ids(list(ctx.get("source_chunks") or [])[:top_k])
        for d in docs:
            d["source"] = "graph"
    docs, rerank_ms, degradation = await _rerank_safe(kb, question, docs, top_k, start, 0)
    retrieval_ms = (time.time() - retr_start) * 1000
    track_retrieval_latency.observe(retrieval_ms / 1000)
    return _build_retrieval_result(docs, top_k, degradation, retrieval_ms, rerank_ms, [], ctx)


def _preprocess_query(question: str, routing: RoutingResult, kb: str) -> tuple[str, str]:
    """Query 预处理: llm/fallback 路由时扩展 BM25 查询, 向量保持原义。"""
    vq, bq = question, question
    if routing.source in ("llm", "fallback"):
        vq, bq = preprocess_query(question)
        if vq != question:
            logger.debug("[%s] 查询预处理: '%s' -> '%s'", kb, question[:30], vq[:40])
    return vq, bq


async def _embed_safe(embedder, vq: str, kb: str, start: float):
    """Embedding (TTL 缓存 + 超时), 失败返回 None 走检索降级。"""
    try:
        return await asyncio.wait_for(_embed_query(embedder, vq), timeout=_remaining(start, 2.0))
    except Exception as e:  # noqa: BLE001 — 降级边界: 任何失败走检索降级
        logger.warning("[%s] Embedding 失败: %s", kb, str(e)[:120])
        return None


async def _retrieve_safe(kb: str, q_emb, bq: str, top_k: int, start: float, mode: str = "hybrid"):
    """向量+BM25 并行 (0.8s 预算)。超时或向量空 → 降级仅 BM25 (L2)。

    mode=vector 时跳过 BM25 (纯向量检索), 但向量失败仍降级到 BM25 兜底。
    """
    vector_store = get_vector_store(kb)
    bm25 = get_bm25_index(kb)
    degradation = 0
    try:
        if mode == "vector" and q_emb is not None:
            # 纯向量模式: 仅向量检索, 不查 BM25
            vector_docs = await asyncio.wait_for(
                asyncio.to_thread(vector_store.search, q_emb, top_k * 2),
                timeout=_remaining(start, 0.8),
            )
            bm25_docs = []
        else:
            vector_docs, bm25_docs = await asyncio.wait_for(
                _parallel_retrieve(vector_store, bm25, q_emb, bq, top_k), timeout=_remaining(start, 0.8)
            )
    except TimeoutError:
        logger.warning("[%s] 检索超时, 降级仅 BM25 (L2)", kb)
        degradation = 2
        vector_docs = []
        bm25_docs = await asyncio.to_thread(bm25.search, bq, top_k * 2)
    if not vector_docs and bm25_docs:
        degradation = max(degradation, 2)
    return vector_docs, bm25_docs, degradation


async def _rerank_safe(kb: str, question: str, fused: list, top_k: int, start: float, degradation: int):
    """Rerank 带预算, 超时/失败跳过 (L1)。返回 (docs, rerank_ms, degradation)。"""
    rerank_ms = 0.0
    docs = fused[:top_k]
    if not docs:
        return docs, rerank_ms, degradation
    t0 = time.time()
    try:
        docs = await asyncio.wait_for(
            asyncio.to_thread(get_reranker().rerank, question, fused, top_k),
            timeout=_remaining(start, settings.rerank_timeout_s),
        )
    except TimeoutError:
        logger.warning("[%s] Rerank 超时, 跳过 (L1)", kb)
        degradation = max(degradation, 1)
    except Exception as e:  # noqa: BLE001 — 降级边界: Rerank 失败跳过 (L1)
        logger.warning("[%s] Rerank 失败, 跳过 (L1): %s", kb, str(e)[:120])
        degradation = max(degradation, 1)
    rerank_ms = (time.time() - t0) * 1000
    track_rerank_latency.observe(rerank_ms / 1000)
    return docs, rerank_ms, degradation


def _build_retrieval_result(
    docs: list, top_k: int, degradation: int, retrieval_ms: float, rerank_ms: float, cross_kb_kbs: list, graph_ctx=None
) -> dict:
    """组装检索结果 dict + 忠实度 context 文本。"""
    top1_score = docs[0].get("score", 0.0) if docs else 0.0
    context = (
        "\n\n---\n\n".join(f"[来源{i + 1}] {d.get('content', '')[:800]}" for i, d in enumerate(docs[:5]))
        if docs
        else ""
    )
    return {
        "docs": docs,
        "context": context,
        "degradation": degradation,
        "retrieval_ms": retrieval_ms,
        "rerank_ms": rerank_ms,
        "top1_score": top1_score,
        "cross_kb_kbs": cross_kb_kbs,
        "retrieval_rounds": 1,
        "rewritten_queries": [],
        "graph_context": graph_ctx,
    }


async def _retrieve_self(question: str, routing: RoutingResult, top_k: int, start: float) -> dict:
    """Self-Retrieval 多轮自适应检索。

    同步主循环（评估+改写+重检索）用 to_thread 跑, wait_for 限时;
    超时/失败降级为普通检索一次（L2）。重写器带 LLM（失败降级模板）。
    """
    kb = routing.kb
    vector_store = get_vector_store(kb)
    embedder = EmbeddingService(model_name=settings.embedding_model, device=settings.embedding_device)
    degradation = 0
    retr_start = time.time()

    def _run_once(query: str, k: int) -> dict:
        """单轮检索: 向量 + BM25 → RRF → Rerank。返回 docs + 用时。"""
        bm25 = get_bm25_index(kb)
        t0 = time.time()
        q_emb = embedder.embed_query(query)
        vd = vector_store.search(q_emb, top_k=k * 2)
        bd = bm25.search(query, top_k=k * 2)
        fused = rrf_fuse(vd, bd)
        docs = fused[:k]
        if docs:
            reranker = get_reranker()
            docs = reranker.rerank(query, fused, k)
        return docs, (time.time() - t0) * 1000

    try:
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            reranker=get_reranker(),
        )
        evaluator = RetrievalEvaluator(embedder=embedder)
        rewriter = QueryRewriter(llm_client=get_sync_llm_client(), rewrite_timeout_s=settings.rewrite_timeout_s)
        sr = SelfRetrieval(
            retriever=retriever,
            evaluator=evaluator,
            rewriter=rewriter,
            max_rounds=3,
        )
        result = await asyncio.wait_for(asyncio.to_thread(sr.retrieve, question, top_k), timeout=_remaining(start, 3.0))
        docs = result.get("documents", [])
        rounds = result.get("retrieval_rounds", 1)
        trace = result.get("trace", [])
        rewritten = [t.get("rewritten", []) for t in trace if t.get("rewritten")]
        rewritten = [q for lst in rewritten for q in lst] if rewritten else []
    except TimeoutError:
        logger.warning("[%s] Self-Retrieval 超时, 降级单轮 (L2)", kb)
        degradation = max(degradation, 2)
        docs, _ = await asyncio.wait_for(asyncio.to_thread(_run_once, question, top_k), timeout=_remaining(start, 2.0))
        rounds = 1
        rewritten = []
    except Exception as e:  # noqa: BLE001 — 降级边界: 失败回退单轮检索 (L2)
        logger.warning("[%s] Self-Retrieval 失败, 降级单轮 (L2): %s", kb, str(e)[:120])
        degradation = max(degradation, 2)
        docs, _ = _run_once(question, top_k)
        rounds = 1
        rewritten = []

    retrieval_ms = (time.time() - retr_start) * 1000
    track_retrieval_latency.observe(retrieval_ms / 1000)
    top1_score = docs[0].get("score", 0.0) if docs else 0.0

    context = (
        "\n\n---\n\n".join(f"[来源{i + 1}] {d.get('content', '')[:800]}" for i, d in enumerate(docs[:5]))
        if docs
        else ""
    )

    return {
        "docs": docs,
        "context": context,
        "degradation": degradation,
        "retrieval_ms": retrieval_ms,
        "rerank_ms": 0.0,
        "top1_score": top1_score,
        "cross_kb_kbs": [],
        "retrieval_rounds": rounds,
        "rewritten_queries": rewritten,
    }


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
) -> dict:
    kb = routing.kb
    retr = await _retrieve_context(question, routing, top_k, start, enable_self_retrieval, mode=mode)
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

    answer, token_usage, llm_ms, llm_ok = await _generate(question, context, llm, start, temperature, history)
    track_llm_latency.observe(llm_ms / 1000)
    if not llm_ok:
        degradation = max(degradation, 3)
        if docs:
            answer = "（LLM 暂时不可用）以下为检索到的相关内容：\n" + "\n".join(
                f"· {d.get('content', '')[:200]}" for d in docs[:3]
            )
        else:
            answer = "未在知识库中找到相关信息，请先上传文档。"
    else:
        # 忠实度护栏: 生成内容与检索上下文词重合过低 → 判定无依据, 拒答防编造
        if docs and _faithfulness_guard(answer, docs) < settings.fidelity_threshold:
            logger.warning("[%s] 忠实度护栏触发: 答案与上下文重合 < %.2f, 改为拒答", kb, settings.fidelity_threshold)
            degradation = max(degradation, 3)
            answer = (
                "知识库中未找到足以可靠回答该问题的内容（可能与现有文档不匹配）。\n"
                "以下为检索到的相关片段供参考：\n" + "\n".join(f"· {d.get('content', '')[:150]}" for d in docs[:3])
            )

    track_degradation.labels(level=str(degradation)).inc()
    retrieval_rounds_hist.observe(retrieval_rounds)
    retrieved_docs_count.observe(len(docs))

    return {
        "answer": answer,
        "sources": docs,
        "skill": routing.skill,
        "kb_id": kb,
        "routing_source": routing.source,
        "retrieval_rounds": retrieval_rounds,
        "rewritten_queries": rewritten_queries,
        "graph_context": graph_ctx,
        "token_usage": token_usage,
        "degradation_level": degradation,
        "latency_breakdown": {
            "retrieval_ms": round(retrieval_ms, 1),
            "rerank_ms": round(rerank_ms, 1),
            "llm_ms": round(llm_ms, 1),
            "total_ms": round((time.time() - start) * 1000, 1),
        },
        "retrieval_meta": {
            "top1_score": round(top1_score, 4),
            "result_count": len(docs),
            "cross_kb": bool(cross_kb_kbs),
            "cross_kb_kbs": cross_kb_kbs,
            "degradation_level": degradation,
        },
    }


async def _generate(
    question: str, context: str, llm: "LLMClient | SyncLLMClient", start: float, temperature: float = 0.1, history=None
):
    """调用 LLM 生成答案。返回 (answer, token_usage_dict|None, ms, ok)。"""
    t0 = time.time()
    if not context:
        return "未在知识库中找到相关信息，请先上传文档。", None, 0.0, False
    try:
        resp = await asyncio.wait_for(
            llm.chat(
                messages=_chat_messages(context, question, history),
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
        "sources": hit.get("sources", []),
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

    cache = get_qa_cache() if settings.qa_cache_enabled else None
    cache_key = (
        cache.make_key(question, skill, top_k, enable_self_retrieval, temperature, mode, history)
        if cache is not None
        else None
    )
    if cache_key is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            qa_cache_hits_total.inc()
            qa_latency_seconds.observe(time.time() - start)
            qa_requests_total.labels(mode=mode, status="cache_hit").inc()
            async for ev in _replay_cache_stream(hit):
                yield ev
            return
        qa_cache_misses_total.inc()

    llm = get_llm_client()

    routing, router_ms = await _route(question, skill, llm, start)

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
        _stream_direct(question, llm, routing, start, router_ms, temperature, history)
        if routing.skill == "direct"
        else _stream_rag(question, routing, llm, top_k, start, enable_self_retrieval, temperature, mode, history)
    )
    async for ev in gen:
        if ev["type"] == "sources":
            cached["sources"] = ev.get("sources", [])
            cached["retrieval_meta"] = ev.get("retrieval_meta", {})
        elif ev["type"] == "done":
            cached["answer"] = ev.get("answer", "")
            cached["token_usage"] = ev.get("token_usage")
            cached["degradation_level"] = ev.get("degradation_level", 0)
            cached["latency_breakdown"] = ev.get("latency_breakdown", cached["latency_breakdown"])
            cached["retrieval_meta"] = ev.get("retrieval_meta", cached.get("retrieval_meta", {}))
            cached["qa_metrics"] = ev.get("qa_metrics", {})
            if cache_key is not None:
                cache.set(cache_key, dict(cached), settings.qa_cache_ttl_s)
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
):
    """流式 RAG Skill: 检索 → 发 sources → 流式 LLM 生成。"""
    retr = await _retrieve_context(question, routing, top_k, start, enable_self_retrieval, mode=mode)
    docs = retr["docs"]
    context = retr["context"]
    degradation = retr["degradation"]
    cross_kb_kbs = retr["cross_kb_kbs"]
    top1_score = retr["top1_score"]

    # 检索完成, 先发 sources 让前端展示来源
    yield {
        "type": "sources",
        "sources": docs,
        "retrieval_meta": {
            "top1_score": round(top1_score, 4),
            "result_count": len(docs),
            "cross_kb": bool(cross_kb_kbs),
            "cross_kb_kbs": cross_kb_kbs,
            "degradation_level": degradation,
        },
    }

    # 流式 LLM 生成
    gen, r = _run_stream_llm(
        llm,
        messages=_chat_messages(context, question, history),
        temperature=temperature,
        max_tokens=2000,
        start=start,
    )
    async for delta in gen:
        yield {"type": "delta", "content": delta}
    answer, token_usage, llm_ok, llm_ms = r["answer"], r["token_usage"], r["ok"], r["llm_ms"]

    if not llm_ok:
        degradation = max(degradation, 3)
        if docs:
            fallback = "（LLM 暂时不可用）以下为检索到的相关内容：\n" + "\n".join(
                f"· {d.get('content', '')[:200]}" for d in docs[:3]
            )
        else:
            fallback = "未在知识库中找到相关信息，请先上传文档。"
        answer = fallback
        yield {"type": "delta", "content": fallback}
    elif docs and _faithfulness_guard(answer, docs) < settings.fidelity_threshold:
        # 忠实度护栏 (流式): 答案已流式发送, 无法撤回, 追加拒答提示 + 相关片段
        logger.warning("[%s] 流式忠实度护栏触发: 答案与上下文重合 < %.2f", routing.kb, settings.fidelity_threshold)
        degradation = max(degradation, 3)
        warning = "\n\n⚠️ 以上回答与知识库内容匹配度较低，请谨慎参考。相关片段：\n" + "\n".join(
            f"· {d.get('content', '')[:150]}" for d in docs[:3]
        )
        answer = answer + warning
        yield {"type": "delta", "content": warning}

    track_degradation.labels(level=str(degradation)).inc()
    retrieval_rounds_hist.observe(retr.get("retrieval_rounds", 1))
    retrieved_docs_count.observe(len(docs))

    yield {
        "type": "done",
        "answer": answer,
        "token_usage": token_usage,
        "degradation_level": degradation,
        "latency_breakdown": {
            "router_ms": round((time.time() - start) * 1000, 1),
            "retrieval_ms": round(retr["retrieval_ms"], 1),
            "rerank_ms": round(retr["rerank_ms"], 1),
            "llm_ms": round(llm_ms, 1),
            "total_ms": round((time.time() - start) * 1000, 1),
        },
        "retrieval_meta": {
            "top1_score": round(top1_score, 4),
            "result_count": len(docs),
            "cross_kb": bool(cross_kb_kbs),
            "cross_kb_kbs": cross_kb_kbs,
            "degradation_level": degradation,
        },
        "qa_metrics": _calc_qa_metrics(
            answer=answer,
            contexts=[d.get("content", "") for d in docs],
            top1=top1_score,
            retrieval_rounds=retr.get("retrieval_rounds", 1),
        ),
    }


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

    yield {
        "type": "done",
        "answer": answer,
        "token_usage": token_usage,
        "degradation_level": 3 if not llm_ok else 0,
        "latency_breakdown": {
            "router_ms": router_ms,
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "llm_ms": round(llm_ms, 1),
            "total_ms": round((time.time() - start) * 1000, 1),
        },
        "retrieval_meta": {
            "top1_score": 0.0,
            "result_count": 0,
            "cross_kb": False,
            "cross_kb_kbs": [],
            "degradation_level": 3 if not llm_ok else 0,
        },
        "qa_metrics": _calc_qa_metrics(
            answer=answer,
            contexts=[],
            top1=0.0,
            retrieval_rounds=0,
        ),
    }


# ────────────────────────── 直接回答 Skill ──────────────────────────


async def _skill_direct(
    question: str,
    llm: "LLMClient | SyncLLMClient",
    routing: RoutingResult,
    start: float,
    temperature: float = 0.1,
    history=None,
) -> dict:
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

    return {
        "answer": answer,
        "sources": [],
        "skill": "direct",
        "kb_id": None,
        "routing_source": routing.source,
        "retrieval_rounds": 0,
        "rewritten_queries": [],
        "graph_context": None,
        "token_usage": token_usage,
        "degradation_level": 3 if no_llm else 0,
        "latency_breakdown": {
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "llm_ms": round(llm_ms, 1),
            "total_ms": round((time.time() - start) * 1000, 1),
        },
        "retrieval_meta": {
            "top1_score": 0.0,
            "result_count": 0,
            "cross_kb": False,
            "cross_kb_kbs": [],
            "degradation_level": 3 if no_llm else 0,
        },
    }


# ────────────────────────── 工具函数 ──────────────────────────


async def _embed_query(embedder, query: str, retries: int = 2):
    """Embedding 带重试（模型本地加载，异常罕见）。"""
    last = None
    for attempt in range(retries + 1):
        try:
            return await asyncio.to_thread(embedder.embed_query, query)
        except Exception as e:  # noqa: BLE001 — 重试兜底, 耗尽后 re-raise
            last = e
            if attempt < retries:
                await asyncio.sleep(0.2 * (attempt + 1))
    raise last


async def _parallel_retrieve(vector_store, bm25, q_emb, bm25_query: str, top_k: int):
    """向量 + BM25 并行检索，各自返回列表。"""

    async def _v():
        if q_emb is None:
            return []
        return await asyncio.to_thread(vector_store.search, q_emb, top_k * 2)

    async def _b():
        return await asyncio.to_thread(bm25.search, bm25_query, top_k * 2)

    vd, bd = await asyncio.gather(_v(), _b())
    return vd or [], bd or []


async def _cross_kb_fallback(
    fused: list[dict], vector_query: str, bm25_query: str, current_kb: str, top_k: int, start: float
):
    """跨库兜底: 并行检索其他非空 RAG 库, 独立预算内合并结果。

    返回 (merged_docs, 命中的库列表)。兜底结果随后参与统一 Rerank。
    """
    siblings = [k for k in RAG_KBS if k != current_kb]
    # 跳过空库（BM25 无文档 → 向量大概率也空, 省预算）
    non_empty = [k for k in siblings if len(get_bm25_index(k)) > 0]
    if not non_empty:
        return fused, []

    async def _search(k: str):
        try:
            return await asyncio.to_thread(_retrieve_kb, k, vector_query, bm25_query, top_k)
        except Exception as e:  # noqa: BLE001 — 跨库兜底: 单库失败不影响其他
            logger.warning("跨库 %s 检索失败: %s", k, str(e)[:100])
            return []

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_search(k) for k in non_empty]), timeout=_remaining(start, settings.cross_kb_timeout_s)
        )
    except TimeoutError:
        logger.warning("跨库兜底超时, 放弃")
        return fused, []

    merged = list(fused)
    found = []
    for k, extra in zip(non_empty, results):
        if extra:
            merged = _merge_docs(merged, extra)
            found.append(k)
    return merged, found


def _retrieve_kb(kb: str, vector_query: str, bm25_query: str, top_k: int) -> list[dict]:
    """同步检索单个库（向量 + BM25 → RRF），用于跨库兜底。"""
    vs = get_vector_store(kb)
    bm = get_bm25_index(kb)
    embedder = EmbeddingService(model_name=settings.embedding_model, device=settings.embedding_device)
    try:
        q_emb = embedder.embed_query(vector_query)
        vd = vs.search(q_emb, top_k=top_k * 2)
    except Exception:  # noqa: BLE001 — 嵌入失败时向量检索降级为空
        vd = []
    bd = bm.search(bm25_query, top_k * 2)
    return rrf_fuse(vd, bd)


def _merge_docs(base: list[dict], extra: list[dict]) -> list[dict]:
    """按 chunk_id 合并去重（保留先出现的分数）。"""
    seen = {(d.get("chunk_id") or d.get("id")) for d in base}
    out = list(base)
    for d in extra:
        key = d.get("chunk_id") or d.get("id")
        if key and key not in seen:
            out.append(d)
            seen.add(key)
    return out


def _report_embed_cache():
    """增量上报 Embedding 缓存命中统计到 Prometheus。"""
    try:
        s = EmbeddingService.embed_cache_stats()
        dh = s["hits"] - _cache_stats_last["hits"]
        dm = s["misses"] - _cache_stats_last["misses"]
        _cache_stats_last["hits"], _cache_stats_last["misses"] = s["hits"], s["misses"]
        if dh > 0:
            embed_cache_hits_total.inc(dh)
        if dm > 0:
            embed_cache_misses_total.inc(dm)
    except Exception as e:  # noqa: BLE001 — 指标上报失败不阻断主流程
        logger.debug("缓存统计上报失败: %s", str(e)[:80])
