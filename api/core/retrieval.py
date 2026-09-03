"""检索管线 / 跨库兜底 / 上下文组装 / 检索工具 (从 orchestrator 拆分)。"""

import asyncio
import logging
import time

import numpy as np

from api.config import settings
from api.core.degradation import _deg_bump
from api.core.metrics import (
    cross_kb_fallback_total,
    embed_cache_hits_total,
    embed_cache_misses_total,
    track_rerank_latency,
    track_retrieval_latency,
)
from api.core.routing import _remaining
from api.state import (
    get_bm25_index,
    get_embedder,
    get_reranker,
    get_vector_store,
    rerank_effective_enabled,
)
from engines.doc_types import RAG_KBS
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.dedup import adaptive_alpha
from engines.retrieval.fusion import fuse
from engines.retrieval.query_augmentation import blend_vectors, prf_augment_vector
from engines.retrieval.query_preprocessor import preprocess_query
from engines.retrieval.retrieval_query_rewriter import RetrievalQueryRewriter
from engines.router.intent_router import RoutingResult

logger = logging.getLogger(__name__)


# Embedding 缓存命中统计上报（增量方式写入 Prometheus Counter）
_cache_stats_last = {"hits": 0, "misses": 0}


async def _retrieve_context(
    question: str,
    routing: RoutingResult,
    top_k: int,
    start: float,
    enable_self_retrieval: bool = False,
    mode: str = "hybrid",
    candidate_kbs: list[str] | None = None,
    defer_rerank: bool = False,
) -> dict:
    """检索上下文: 预处理 → Embedding → 向量|BM25 → 图谱增强 → RRF → 跨库兜底 → Rerank。

    mode: hybrid(向量+BM25+图谱) / vector(纯向量) / graph(纯图谱)。
    供 `_skill_rag`（一次性）与 `ask_stream`（流式）共用, 避免检索管线重复。
    enable_self_retrieval=True 时走 Self-Retrieval 多轮循环（graph 模式为单轮, 忽略该参数）。

    defer_rerank=True: **跳过 Rerank**, 直接返回 RRF 融合序的 docs, 并把融合结果挂在
    result["fused"] 上, 供调用方稍后用 `_apply_deferred_rerank()` 补做重排。
    目的 (P0-2): 让流式链路先把来源发给前端 (~50ms 可见), 再在后台做 ~700ms 的 rerank,
    消除用户"检索阶段完全空白"的等待。默认 False = 保持原行为。

    返回 dict: docs/context/degradation/retrieval_ms/rerank_ms/top1_score/
               cross_kb_kbs/retrieval_rounds/rewritten_queries/graph_context[/fused]。
    """
    if mode == "graph":
        return await _retrieve_graph_only(question, routing, top_k, start)
    if enable_self_retrieval:
        return await _retrieve_self(question, routing, top_k, start)

    kb = routing.kb
    embedder = get_embedder()
    retr_start = time.time()

    # P1-1: 单轮 LLM 查询改写(门控) — 改写查询只用于检索, 原问题仍交 LLM 生成答案。
    # 改写结果再过一遍规则预处理(保留 BM25 同义扩展)。超时/失败由 RetrievalQueryRewriter 回退原查询。
    retrieve_query = question
    if settings.query_rewrite_enabled:
        retrieve_query = await RetrievalQueryRewriter().rewrite(question)
    vq, bq = _preprocess_query(retrieve_query, routing, kb)
    q_emb = await _embed_safe(embedder, vq, kb, start)
    # 查询嵌入增广 (P 实验项, 默认关): 仅作用于向量腿, BM25 腿仍用原查询。
    if settings.query_augmentation_enabled:
        q_emb = await _augment_query_embedding(embedder, vq, q_emb, kb, start)
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

    fused = fuse(vector_docs, bm25_docs, method=settings.fusion_method, w_bm25=settings.fusion_bm25_weight)
    retrieval_ms = (time.time() - retr_start) * 1000
    track_retrieval_latency.observe(retrieval_ms / 1000)

    # 主库无结果 或 top1 < 阈值 → 跨库兜底 (并行, 独立预算; 仅 hybrid 模式)
    # 关键修复: 原逻辑要求 fused[0] 存在才兜底, 主库空(fused=[])时直接跳过 →
    # 意图路由一旦命中空知识库(如 tech/policy 均为 0 行)即永久失检索, 表现为"检索不到来源/幻觉"
    # OPT-X1: 显式指定技能 (source=manual) 时禁用跨库兜底 —— 用户点名要 A 库,
    # 就不该静默混入 B 库内容 (P0' 评测实证: 强制 tech 仍检出 service 库"物流赔付"文本)。
    # 行为变更说明: manual 路径主库无结果时如实拒答, 不再越权联想。
    cross_kb_kbs = []
    if (
        mode == "hybrid"
        and routing.source != "manual"
        and (not fused or fused[0].get("score", 0.0) < settings.cross_kb_threshold)
    ):
        fused, cross_kb_kbs = await _cross_kb_fallback(fused, vq, bq, kb, top_k, start, candidate_kbs)
        for to_kb in cross_kb_kbs:
            cross_kb_fallback_total.labels(from_kb=kb, to_kb=to_kb).inc()

    # Rerank (超时/失败跳过 = L1)。defer_rerank 时留给调用方在发出来源后再补做。
    if defer_rerank:
        result = _build_retrieval_result(fused[:top_k], top_k, degradation, retrieval_ms, 0.0,
                                         cross_kb_kbs, graph_ctx)
        result["fused"] = fused
        result["rerank_deferred"] = True
        return result

    docs, rerank_ms, degradation = await _rerank_safe(kb, question, fused, top_k, start, degradation)

    return _build_retrieval_result(docs, top_k, degradation, retrieval_ms, rerank_ms, cross_kb_kbs, graph_ctx)


async def _retrieve_fanout(
    question: str,
    routings: list[RoutingResult],
    top_k: int,
    start: float,
    enable_self_retrieval: bool = False,
    mode: str = "hybrid",
    candidate_kbs: list[str] | None = None,
) -> dict:
    """P1' 选择性扇出检索: 对候选 KB 各跑检索管线, 跨库去重合并后**全局一次性重排**取 top_k。

    用于主候选模糊(conf < early_exit)时覆盖正确库, 解决单库路由错误导致的端到端召回丢失。

    跨库 rerank (#3): 各库先只出 RRF 融合候选(defer_rerank=True, 不各自 rerank), 按 chunk_id
    跨库去重合并成单一候选池, 再对整池跑一次 _rerank_safe —— Cross-Encoder 在**同一语义空间**
    直接比较不同库片段, 跨库边界不再丢失; 旧逻辑"逐库各自 rerank 后按不可比的 CE 分合并"会把
    跨库排序切碎。降级安全: rerank 关闭/超时/失败时 _rerank_safe 退回 RRF 序, 不会比旧逻辑差。

    防御性: 单个 KB 检索失败不影响其他(异常由 _retrieve_context 内部降级, 此处空 docs 跳过)。
    """
    # 各库融合候选按库分组保留(每库内部已按融合分降序), 供跨库轮转交织
    per_kb_fused: list[list[dict]] = []
    cross_kb_kbs: list[str] = []
    retrieval_ms_total = 0.0
    for r in routings:
        if not r.kb:
            continue
        try:
            retr = await _retrieve_context(
                question, r, top_k, start, enable_self_retrieval, mode=mode,
                candidate_kbs=candidate_kbs, defer_rerank=True,  # 各库只出融合候选, 合并后全局重排
            )
            fused = retr.get("fused")
            if fused:
                per_kb_fused.append(fused)
            if retr.get("cross_kb_kbs"):
                cross_kb_kbs.extend(retr["cross_kb_kbs"])
            retrieval_ms_total += float(retr.get("retrieval_ms", 0.0) or 0.0)
        except Exception as e:  # noqa: BLE001 — 扇出单库失败不阻断其他库
            logger.warning("[%s] 扇出检索失败, 跳过: %s", r.kb, str(e)[:120])

    # 跨库去重: 同 chunk_id 保留融合分(RRF/interp 的 _rrf)最高的一份 (各库融合分可比)
    best: dict[str, dict] = {}
    for fused in per_kb_fused:
        for d in fused:
            cid = d.get("chunk_id") or d.get("id")
            if cid is None:
                continue
            sc = d.get("_rrf", d.get("score", 0.0))
            if cid not in best or sc > best[cid].get("_rrf", best[cid].get("score", 0.0)):
                best[cid] = d

    # 关键修复 (跨库 rerank 截断偏差): 直接按路由顺序 concat 去重后, 排在前面的库会占满
    # rerank_candidate_k 预算, 答案所在的后序库候选被截在 Cross-Encoder 视野之外 —— 全局重排
    # 名存实亡, rerank 关闭的降级路径 top_k 也只剩主库内容。改为**跨库轮转交织**(round-robin):
    # 逐轮每库取下一个未入选候选, 直至预算或用尽。各库 top 候选都能进入 CE 语义空间参与比较,
    # 且候选总数不突破 rerank_candidate_k(CE 成本预算不变)。跨库重复(chunk 同时命中多库)已
    # 由 best 去重, 交织时跳过重复 id。

    cap = settings.rerank_candidate_k
    pool: list[dict] = []
    emitted: set[str] = set()
    max_rank = max((len(f) for f in per_kb_fused), default=0)
    for rank in range(max_rank):
        for fused in per_kb_fused:
            if rank >= len(fused):
                continue
            d = fused[rank]
            cid = d.get("chunk_id") or d.get("id")
            if cid is None or cid in emitted:
                continue
            emitted.add(cid)
            pool.append(best.get(cid, d))  # 用去重后的优胜版本(内容相同)
            if cap > 0 and len(pool) >= cap:
                break
        if cap > 0 and len(pool) >= cap:
            break

    # 全局一次性跨库重排序 (true cross-KB rerank); primary_kb 仅用于日志
    primary_kb = routings[0].kb if (routings and routings[0].kb) else (pool[0].get("kb") if pool else "")
    docs, rerank_ms, degradation = await _rerank_safe(primary_kb, question, pool, top_k, start, 0)
    top1_score = docs[0].get("score", 0.0) if docs else 0.0
    context, sources = _build_context(docs, top_k)
    return {
        "docs": sources,
        "context": context,
        "degradation": degradation,
        "retrieval_ms": retrieval_ms_total,
        "rerank_ms": rerank_ms,
        "top1_score": top1_score,
        "cross_kb_kbs": cross_kb_kbs,
        "retrieval_rounds": 1,
        "rewritten_queries": [],
        "graph_context": None,
    }


async def _apply_deferred_rerank(retr: dict, kb: str, question: str, top_k: int, start: float) -> dict:
    """补做被 defer 的 Rerank, 并用重排后的结果重建 context/sources。

    与 defer_rerank=True 配对使用。若 rerank 超时/失败, `_rerank_safe` 内部会降级,
    这里拿到的 docs 退化为 RRF 序 —— 与"不重排"等价, 不会更差。
    """
    fused = retr.pop("fused", None)
    retr.pop("rerank_deferred", None)
    if fused is None:
        return retr
    docs, rerank_ms, degradation = await _rerank_safe(kb, question, fused, top_k, start, retr["degradation"])
    context, sources = _build_context(docs, top_k)
    retr.update({
        "docs": sources,
        "context": context,
        "rerank_ms": rerank_ms,
        "degradation": degradation,
        "top1_score": docs[0].get("score", 0.0) if docs else 0.0,
    })
    return retr


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
    """Query 预处理: 无条件执行查询重写，统一所有路由路径的检索质量。"""
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


_HYDE_SYSTEM = (
    "你是问答知识库的检索增强器。给定用户问题, 写一段可能出现在知识库中的『标准答案文档』"
    "(包含具体术语、步骤、要点), 用于提升向量检索召回。只输出文档正文, 不超过 80 字, 不要解释。"
)


async def _hyde_hypothetical_doc(question: str) -> str | None:
    """HyDE 假设文档生成。无 key/超时/失败 → 返回 None (上层回退原查询向量)。"""
    try:
        from api.core.llm_client import get_llm_client

        client = get_llm_client()
        if not getattr(client, "api_key", None):
            return None
        resp = await asyncio.wait_for(
            client.chat(
                [{"role": "system", "content": _HYDE_SYSTEM}, {"role": "user", "content": question}],
                temperature=0.0,
                max_tokens=96,
            ),
            timeout=settings.query_augmentation_hyde_timeout_s,
        )
        return (getattr(resp, "content", "") or "").strip() or None
    except Exception as e:  # noqa: BLE001 — HyDE 是增强项, 失败回退原查询向量
        logger.warning("[%s] HyDE 假设文档生成失败, 回退原查询向量: %s", question[:30], str(e)[:100])
        return None


async def _augment_query_embedding(embedder, vq: str, q_emb, kb: str, start: float):
    """查询嵌入增广: 返回(可能增广的)查询向量列表; 任何异常/超时回退原 q_emb。

    仅作用于向量检索腿; BM25 腿由调用方用原查询, 不受此影响。
    start: 请求起点, 用于按剩余全局预算收敛增广超时(预算见底时 wait_for 立刻超时回退)。
    """
    try:
        if q_emb is None:
            return None  # 嵌入已失败走 BM25 降级, 增广无意义
        q_arr = np.asarray(q_emb, dtype=float)
        if settings.query_augmentation_strategy == "prf":
            # 伪相关反馈: 首轮向量检索 top-k 取反馈向量, 与查询向量加权融合。
            # 超时加固: PRF 默认开启后, 这是每问新增的同步阻塞调用(to_thread 向量检索),
            # 必须带预算 —— 否则向量库挂起会拖垮整条请求。HyDE 分支自带 wait_for, 此处原缺失。
            vs = get_vector_store(kb)
            fb = await asyncio.wait_for(
                asyncio.to_thread(vs.search, q_emb, settings.query_augmentation_prf_k),
                timeout=_remaining(start, settings.query_augmentation_prf_timeout_s),
            )
            fb_embs = [np.array(d["embedding"]) for d in fb if d.get("embedding")]
            if fb_embs:
                return prf_augment_vector(q_arr, fb_embs, settings.query_augmentation_weight).tolist()
            return q_emb
        # hyde: 生成假设文档并嵌入, 与查询向量融合 (passage 前缀由 embed_batch 统一加)
        hyp = await _hyde_hypothetical_doc(vq)
        if hyp:
            hyp_emb = await asyncio.to_thread(embedder.embed_batch, [hyp[:512]])
            return blend_vectors(q_arr, np.asarray(hyp_emb[0], dtype=float), settings.query_augmentation_weight).tolist()
        return q_emb
    except Exception as e:  # noqa: BLE001 — 增广是增强项, 失败绝不阻断检索
        logger.warning("[%s] 查询增广失败, 回退原查询向量: %s", kb, str(e)[:120])
        return q_emb


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
        degradation = _deg_bump(degradation, 2, "retrieval")
        vector_docs = []
        bm25_docs = await asyncio.to_thread(bm25.search, bq, top_k * 2)
    if not vector_docs and bm25_docs:
        degradation = _deg_bump(degradation, 2, "retrieval")
    return vector_docs, bm25_docs, degradation


async def _rerank_safe(kb: str, question: str, fused: list, top_k: int, start: float, degradation: int):
    """Rerank 带预算, 超时/失败跳过 (L1)。返回 (docs, rerank_ms, degradation)。"""
    rerank_ms = 0.0
    # 只重排前 rerank_candidate_k 条: cross-encoder 延迟随候选数线性增长,
    # 而 RRF 排名靠后的文档进入最终 top_k 的概率极低, 截顶几乎无损。
    cap = settings.rerank_candidate_k
    candidates = fused[:cap] if cap > 0 else fused
    docs = candidates[:top_k]
    if not docs:
        return docs, rerank_ms, degradation
    if not settings.reranker_model:
        return docs, rerank_ms, degradation
    if not rerank_effective_enabled():  # P2#11: 综合显式开关与 GPU 自适应
        return docs, rerank_ms, degradation
    reranker = get_reranker()
    t0 = time.time()
    try:
        if settings.rerank_fusion_enabled:
            # P0-1 正确修复: CE 分与 RRF 检索分融合, 用 golden 天然靠前的高检索分托住它,
            # 避免密集近重复语料下纯 Cross-Encoder 把 golden 误排到干扰项之后 (Recall@3 下滑)。
            # alpha 自适应: 池子越近重复密集 -> 越低 alpha (越依赖检索分托底)。
            alpha = settings.rerank_fusion_alpha
            if settings.rerank_fusion_adaptive:
                try:
                    alpha = adaptive_alpha(
                        candidates,
                        threshold=settings.rerank_density_threshold,
                        mode=settings.rerank_density_mode,
                        alpha_max=settings.rerank_alpha_max,
                        alpha_min=settings.rerank_alpha_min,
                        density_full=settings.rerank_density_full,
                    )
                except Exception as e:  # noqa: BLE001 — 密度计算失败不应阻断检索, 退回固定 alpha
                    logger.warning("[%s] 自适应 alpha 计算失败, 用固定值: %s", kb, str(e)[:120])
            rrf_map = {d.get("chunk_id") or d.get("id"): d.get("_rrf", 0.0) for d in candidates}
            docs = await asyncio.wait_for(
                asyncio.to_thread(reranker.rerank_fused, question, candidates, rrf_map, top_k, alpha),
                timeout=_remaining(start, settings.rerank_timeout_s),
            )
        else:
            docs = await asyncio.wait_for(
                asyncio.to_thread(reranker.rerank, question, candidates, top_k),
                timeout=_remaining(start, settings.rerank_timeout_s),
            )
    except TimeoutError:
        logger.warning("[%s] Rerank 超时, 跳过 (L1)", kb)
        degradation = _deg_bump(degradation, 1, "rerank")
    except Exception as e:  # noqa: BLE001 — 降级边界: Rerank 失败跳过 (L1)
        logger.warning("[%s] Rerank 失败, 跳过 (L1): %s", kb, str(e)[:120])
        degradation = _deg_bump(degradation, 1, "rerank")
    rerank_ms = (time.time() - t0) * 1000
    track_rerank_latency.observe(rerank_ms / 1000)
    return docs, rerank_ms, degradation


def _reorder_for_attention(parts: list[str]) -> list[str]:
    """偶数位升序 + 奇数位降序: 最相关的落到首尾, 最差的被挤到中间。

    LLM 对长上下文**中间位置**的证据存在系统性忽略 (lost-in-the-middle)。
    按相关度顺序直接拼接会把次相关的证据全塞在中间, 恰好是最容易被忽略的位置。
    重排后 [p0,p1,p2,p3,p4] → [p0,p2,p4,p3,p1]: p0 首位、p1 末位、最差的 p4 居中。

    只重排送入 prompt 的片段; sources 仍按相关度顺序返回, 前端展示不受影响。
    """
    if len(parts) <= 2:
        return parts
    return parts[0::2] + parts[1::2][::-1]


def _build_context(docs: list, top_k: int = 5) -> tuple[str, list]:
    """按 doc_id 分组, 合并相邻 chunk, 展示 title_chain 节标题。

    sources 必须保留 source_file / score: 该列表会经 _build_retrieval_result 变成
    retr["docs"], 再由 build_sources_event 原样发给前端; web/index.html 的
    finalizeAnswer 直接渲染 s.source_file 与 s.score。此前这里只输出
    doc_id/title_chain/content, 两个字段被丢弃, 前端来源面板因此显示空文件名与 0.000 分。
    """
    groups = {}
    for d in docs[:top_k * 2]:
        did = d.get("doc_id", "")
        if did not in groups:
            groups[did] = {
                "chunks": [],
                "chunk_ids": [],
                # 记录首个 chunk 的单值 id/chunk_id, 供 API SourceDocument.id/chunk_id 透传。
                # 此前 _build_context 分组后丢弃了这两个字段, 导致 HTTP 响应里恒为空字符串。
                "id": d.get("id") or d.get("chunk_id") or "",
                "chunk_id": d.get("chunk_id") or d.get("id") or "",
                "title_chain": d.get("title_chain", []),
                "doc_title": d.get("doc_title", ""),
                "source_file": d.get("source_file", ""),
                "score": d.get("score", 0.0),
            }
        elif (d.get("score") or 0.0) > groups[did]["score"]:
            # 同一 doc_id 的多个 chunk: 取组内最高分代表该文档
            groups[did]["score"] = d.get("score", 0.0)
        # 父子文档机制: 检索命中的是子块(精确), 但回给 LLM 用父块大上下文(parent_content);
        # 来源面板 sources 仍用子块片段(精确引用), 二者分离。
        groups[did]["chunks"].append(d.get("parent_content") or d.get("content", ""))
        cid = d.get("id") or d.get("chunk_id")
        if cid:
            groups[did]["chunk_ids"].append(cid)

    parts = []
    sources = []
    for i, (did, g) in enumerate(groups.items()):
        chain = " > ".join(g["title_chain"]) if g["title_chain"] else g["doc_title"]
        content = "\n\n".join(g["chunks"])[:800]
        label = f"[来源{i + 1}] {chain}" if chain else f"[来源{i + 1}]"
        parts.append(f"{label}\n【片段开始】\n{content}\n【片段结束】")
        sources.append({
            "id": g.get("id", ""),
            "chunk_id": g.get("chunk_id", ""),
            "doc_id": did,
            "title_chain": g["title_chain"],
            "chunk_ids": g["chunk_ids"],
            "content": content[:200],
            # LLM 实际看到的完整文本(每段上限 800 字)。仅供忠实度护栏与 qa_metrics 使用:
            # 拿 200 字显示片段判忠实度, 会把"LLM 见过但片段没展示的内容"误判为无依据,
            # 触发误拒答。API 路由按 SourceDocument 显式构造响应, 此内部字段不会外泄。
            "context_full": content,
            "source_file": g["source_file"],
            "score": g["score"],
        })

    return "\n\n---\n\n".join(_reorder_for_attention(parts)), sources


def _build_retrieval_result(
    docs: list, top_k: int, degradation: int, retrieval_ms: float, rerank_ms: float, cross_kb_kbs: list, graph_ctx=None
) -> dict:
    """组装检索结果 dict + 忠实度 context 文本。"""
    top1_score = docs[0].get("score", 0.0) if docs else 0.0
    context, sources = _build_context(docs, top_k)
    return {
        "docs": sources,
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
    embedder = get_embedder()
    degradation = 0
    retr_start = time.time()

    def _run_once(query: str, k: int) -> dict:
        """单轮检索: 向量 + BM25 → RRF → Rerank。返回 docs + 用时。"""
        bm25 = get_bm25_index(kb)
        t0 = time.time()
        q_emb = embedder.embed_query(query)
        vd = vector_store.search(q_emb, top_k=k * 2)
        bd = bm25.search(query, top_k=k * 2)
        fused = fuse(vd, bd, method=settings.fusion_method, w_bm25=settings.fusion_bm25_weight)
        docs = fused[:k]
        if docs:
            reranker = get_reranker()  # 未生效时返回 None (P2#11)
            if reranker is not None:
                docs = reranker.rerank(query, fused, k)
        return docs, (time.time() - t0) * 1000

    # 原: HybridRetriever 只传 vector_store, 跳过 BM25/图谱/跨库
    # 改: 复用 _run_once 的完整管线 (向量+BM25 -> RRF -> Rerank)
    try:
        docs, _ = await asyncio.wait_for(
            asyncio.to_thread(_run_once, question, top_k), timeout=_remaining(start, 3.0)
        )
        rounds = 1
        rewritten = []
    except TimeoutError:
        logger.warning("[%s] Self-Retrieval 超时, 降级单轮 (L2)", kb)
        degradation = _deg_bump(degradation, 2, "retrieval")
        docs, _ = await asyncio.wait_for(asyncio.to_thread(_run_once, question, top_k), timeout=_remaining(start, 2.0))
        rounds = 1
        rewritten = []
    except Exception as e:  # noqa: BLE001 — 降级边界: 失败回退单轮检索 (L2)
        logger.warning("[%s] Self-Retrieval 失败, 降级单轮 (L2): %s", kb, str(e)[:120])
        degradation = _deg_bump(degradation, 2, "retrieval")
        # 同步检索同样卸载到线程池并限时, 防止阻塞事件循环
        docs, _ = await asyncio.wait_for(
            asyncio.to_thread(_run_once, question, top_k), timeout=_remaining(start, 2.0)
        )
        rounds = 1
        rewritten = []

    retrieval_ms = (time.time() - retr_start) * 1000
    track_retrieval_latency.observe(retrieval_ms / 1000)
    top1_score = docs[0].get("score", 0.0) if docs else 0.0

    context, sources = _build_context(docs, top_k)

    return {
        "docs": sources,
        "context": context,
        "degradation": degradation,
        "retrieval_ms": retrieval_ms,
        "rerank_ms": 0.0,
        "top1_score": top1_score,
        "cross_kb_kbs": [],
        "retrieval_rounds": rounds,
        "rewritten_queries": rewritten,
    }


# ────────────────────────── 工具函数 ──────────────────────────


def _dedupe_docs(docs: list[dict], max_n: int = 3) -> list[dict]:
    """按 content 去重: 保留首次出现的唯一文本块。防 LLM 不可用 fallback 时展示重复 chunk。"""
    seen: set[str] = set()
    out: list[dict] = []
    for d in docs:
        c = d.get("content") or ""
        # 截短到 200 字符做指纹匹配 — 长片段前 200 字相同视为同一来源
        sig = c[:200]
        if sig and sig not in seen:
            seen.add(sig)
            out.append(d)
            if len(out) >= max_n:
                break
    return out


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
    fused: list[dict],
    vector_query: str,
    bm25_query: str,
    current_kb: str,
    top_k: int,
    start: float,
    candidate_kbs: list[str] | None = None,
):
    """跨库兜底: 并行检索其他非空 RAG 库, 独立预算内合并结果。

    返回 (merged_docs, 命中的库列表)。兜底结果随后参与统一 Rerank。
    candidate_kbs: RBAC 授权范围, 仅在该范围内跨库; None=全部。
    """
    # None=全部(RBAC 未限制); []=明确无权访问任何库 → 空集, 不可回退为全部(否则越权)
    scope = list(candidate_kbs) if candidate_kbs is not None else list(RAG_KBS)
    # 关键修复: 遗留 documents 默认表(存量文档 + 未指定类型的上传均落此处)不在 RAG_KBS 中,
    # 但存有用户真实文档, 必须纳入兜底检索范围, 否则这部分数据永远查不到
    if "documents" not in scope:
        scope = scope + ["documents"]
    siblings = [k for k in scope if k != current_kb]
    # 跳过空库（BM25 无文档 → 向量大概率也空, 省预算）; get_bm25_index 惰性构建可能耗时, 卸载线程池
    async def _is_non_empty(k: str) -> bool:
        try:
            return await asyncio.to_thread(lambda: len(get_bm25_index(k)) > 0)
        except Exception as e:  # noqa: BLE001 — 单库异常按空库处理
            logger.warning("跨库 %s 空库探测失败: %s", k, str(e)[:100])
            return False

    flags = await asyncio.gather(*[_is_non_empty(k) for k in siblings])
    non_empty = [k for k, ok in zip(siblings, flags) if ok]
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
    embedder = get_embedder()
    try:
        q_emb = embedder.embed_query(vector_query)
        vd = vs.search(q_emb, top_k=top_k * 2)
    except Exception:  # noqa: BLE001 — 嵌入失败时向量检索降级为空
        vd = []
    bd = bm.search(bm25_query, top_k * 2)
    return fuse(vd, bd, method=settings.fusion_method, w_bm25=settings.fusion_bm25_weight)


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
