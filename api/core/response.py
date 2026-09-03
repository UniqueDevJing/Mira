"""统一响应组装 — QA result dict 与 SSE 事件结构。

从 orchestrator 下沉。原先 latency_breakdown / retrieval_meta / qa_metrics 三件套
分散在 _skill_rag(2 处) / _stream_rag(3 处) / _stream_direct(1 处) / ask / ask_stream
共 7 处各自拼装, 其中 _stream_rag 的两处把 router_ms 写成
`(time.time() - start) * 1000` —— 那是 total_ms, 与 ask / ask_stream /
_stream_direct 的正确取值不一致(它们用的是 _route 返回的真实路由耗时)。

router_ms 填充约定
------------------
`_skill_rag` / `_stream_rag` 的签名不含 router_ms(改动签名会打断测试里的
monkeypatch fake), 因此传 0.0 占位, 由入口 ask() / ask_stream() 拿到真实值后
统一覆盖。`_stream_direct` 签名已含 router_ms, 直接传真实值。

`build_latency_breakdown` 要求 router_ms 由调用方显式传入(毫秒), 不接受 start
兜底 —— 从结构上杜绝"用总耗时冒充路由耗时"再次发生。
"""

from api.core.qa_metrics import _calc_qa_metrics


def build_retrieval_meta(
    top1_score: float,
    result_count: int,
    cross_kb_kbs: list[str] | None,
    degradation_level: int,
) -> dict:
    """检索元数据 — sources 事件 / result / done 三方共用同一份结构。"""
    return {
        "top1_score": round(top1_score, 4),
        "result_count": result_count,
        "cross_kb": bool(cross_kb_kbs),
        "cross_kb_kbs": cross_kb_kbs or [],
        "degradation_level": degradation_level,
    }


def build_latency_breakdown(
    router_ms: float,
    retrieval_ms: float,
    rerank_ms: float,
    llm_ms: float,
    total_ms: float,
) -> dict:
    """耗时分解 — 五个字段恒定齐全, 避免各处缺字段导致下游 KeyError。"""
    return {
        "router_ms": round(router_ms, 1),
        "retrieval_ms": round(retrieval_ms, 1),
        "rerank_ms": round(rerank_ms, 1),
        "llm_ms": round(llm_ms, 1),
        "total_ms": round(total_ms, 1),
    }


def calc_qa_metrics(answer: str, docs: list[dict], top1_score: float, retrieval_rounds: int = 1) -> dict:
    """从 docs 抽取 context 计算运行时近似质量指标。"""
    return _calc_qa_metrics(
        answer=answer,
        # context_full = LLM 实际看到的完整文本; content 只是 200 字显示片段,
        # 用片段算忠实度会系统性低估 (与忠实度护栏/评测判分器同源问题)
        contexts=[d.get("context_full") or d.get("content", "") for d in docs],
        top1=top1_score,
        retrieval_rounds=retrieval_rounds,
    )


def build_qa_result(
    *,
    answer: str,
    docs: list[dict],
    routing,
    kb: str | None,
    retrieval_rounds: int,
    rewritten_queries: list[str],
    graph_context,
    token_usage,
    degradation_level: int,
    router_ms: float,
    retrieval_ms: float,
    rerank_ms: float,
    llm_ms: float,
    total_ms: float,
    top1_score: float,
    cross_kb_kbs: list[str] | None,
    refusal: dict | None = None,
) -> dict:
    """非流式 result —— ask() 的返回体。

    router_ms 传 0.0 占位时由 ask() 覆盖为真实值; qa_metrics 同样由 ask() 追加,
    因为 answer 在降级分支可能被改写, 入口处计算才是最终值。
    refusal 仅在答案确为拒答时非空(由编排层构造), 透传给 QAResponse.refusal。
    """
    result = {
        "answer": answer,
        "sources": docs,
        "skill": routing.skill,
        "kb_id": kb,
        "routing_source": routing.source,
        "retrieval_rounds": retrieval_rounds,
        "rewritten_queries": rewritten_queries,
        "graph_context": graph_context,
        "token_usage": token_usage,
        "degradation_level": degradation_level,
        "latency_breakdown": build_latency_breakdown(
            router_ms, retrieval_ms, rerank_ms, llm_ms, total_ms
        ),
        "retrieval_meta": build_retrieval_meta(top1_score, len(docs), cross_kb_kbs, degradation_level),
    }
    if refusal is not None:
        result["refusal"] = refusal
    return result


def public_sources(docs: list[dict]) -> list[dict]:
    """剔除仅内部使用的字段后返回前端/评测可见的 sources。

    context_full 是忠实度护栏/qa_metrics 用的完整文本(每段最长 800 字),
    不应随 SSE/HTTP 响应外发 (payload 冗余 + 暴露内部实现细节)。
    浅拷贝逐条剔除, 不改动原 dict (原列表在流式链路后续仍会被护栏使用)。
    """
    return [{k: v for k, v in d.items() if k != "context_full"} for d in docs]


def build_sources_event(
    docs: list[dict],
    top1_score: float,
    cross_kb_kbs: list[str] | None,
    degradation_level: int,
    final: bool = True,
) -> dict:
    """流式首个事件: 先把检索来源发给前端展示。

    final=False 表示这是"重排前"的临时来源 (见 stream_sources_before_rerank):
    rerank 完成后会再发一次 final=True 的同类型事件覆盖它。前端按整体替换渲染, 无副作用。
    """
    return {
        "type": "sources",
        "sources": public_sources(docs),
        "final": final,
        "retrieval_meta": build_retrieval_meta(top1_score, len(docs), cross_kb_kbs, degradation_level),
    }


def build_done_event(
    *,
    answer: str,
    docs: list[dict],
    token_usage,
    degradation_level: int,
    router_ms: float,
    retrieval_ms: float,
    rerank_ms: float,
    llm_ms: float,
    total_ms: float,
    top1_score: float,
    cross_kb_kbs: list[str] | None,
    retrieval_rounds: int = 1,
    refusal: dict | None = None,
) -> dict:
    """流式收尾事件 —— 字段与非流式 result 对齐(多一个 type, 多一个 qa_metrics)。"""
    ev = {
        "type": "done",
        "answer": answer,
        "token_usage": token_usage,
        "degradation_level": degradation_level,
        "latency_breakdown": build_latency_breakdown(
            router_ms, retrieval_ms, rerank_ms, llm_ms, total_ms
        ),
        "retrieval_meta": build_retrieval_meta(top1_score, len(docs), cross_kb_kbs, degradation_level),
        "qa_metrics": calc_qa_metrics(answer, docs, top1_score, retrieval_rounds),
    }
    if refusal is not None:
        ev["refusal"] = refusal
    return ev


def build_refusal_info(reason: str, docs: list[dict], question: str = "", kb: str = "") -> dict:
    """构造拒答分级元数据 —— 把"未找到"从死胡同变成带候选来源的岔路口。

    入参 docs 为 _build_context 产出的分组来源(含 doc_id/title_chain/doc_title/
    source_file/score/chunk_ids), 与 _retrieve_context 透传给护栏的同源。

    防御性: docs 为空/None 或字段缺失均不抛异常, 仅产出候选为空的 refusal
    (真拒答语义), 不阻断主流程。candidates 按 doc_id 去重, 取前 5 个最高分。

    reason 必为 low_confidence / low_fidelity / no_docs 之一; kb 透传给候选,
    供 U2「来源展开全文」端点做 KB 级 RBAC。
    """
    candidates = []
    for d in (docs or [])[:5]:
        tc = d.get("title_chain") or []
        title = (
            " > ".join(tc)
            if tc
            else (d.get("doc_title") or d.get("source_file") or d.get("doc_id") or "")
        )
        if not title:
            continue
        candidates.append({
            "doc_id": d.get("doc_id", ""),
            "kb": kb,
            "title": title,
            "score": float(d.get("score", 0.0) or 0.0),
            "chunk_ids": list(d.get("chunk_ids") or []),
        })
    # 按 doc_id 去重(同一文档多 chunk 命中只列一次), 保持原顺序
    seen, uniq = set(), []
    for c in candidates:
        if c["doc_id"] and c["doc_id"] in seen:
            continue
        seen.add(c["doc_id"])
        uniq.append(c)

    return {
        "is_refusal": True,
        "reason": reason,
        "candidates": uniq,
        "suggested_questions": _suggest_followups(question, uniq),
    }


def _suggest_followups(question: str, candidates: list[dict]) -> list[str]:
    """确定性生成引导追问(不调用 LLM, 零额外延迟)。

    两类提示: ① 基于原问题的改写建议; ② 基于 Top 候选的「展开来源」引导(衔接 U2)。
    最多 3 条, 始终保证至少 1 条可用。
    """
    out: list[str] = []
    q = (question or "").strip().strip("?？。.！!").strip()
    if q:
        out.append(f"能否换一种表述方式，例如更具体地描述您想了解的「{q[:20]}」？")
    if candidates:
        top_title = candidates[0]["title"]
        out.append(f"查看《{top_title}》中的相关内容（展开来源全文）")
    if len(out) < 2:
        out.append("确认问题所属的知识库（产品 / 服务 / 技术等）是否正确。")
    return out[:3]
