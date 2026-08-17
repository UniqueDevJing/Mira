"""知识问答 API — 入口交给 Router + Skill 编排层"""

import asyncio
import json
import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.core.document_store import get_document_store
from api.core.limiter import limiter
from api.core.orchestrator import ask as orchestrate
from api.core.orchestrator import ask_stream as orchestrate_stream
from api.core.session_store import clear_session, load_session
from api.schemas.qa import QARequest, QAResponse, SourceDocument, TokenUsage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/qa", tags=["qa"])


def _rate_limited(fn):
    """限流开启时附加 slowapi 限制, 未开启原样返回 (防本地/测试误触)。"""
    if limiter is None:
        return fn
    from api.config import settings

    return limiter.limit(f"{settings.rate_limit_per_minute}/minute")(fn)


@router.post("/ask", response_model=QAResponse)
@_rate_limited
async def ask_question(req: QARequest):
    start = time.time()

    # 编排层: Router 路由 → Skill 执行（分阶段超时/降级/跨库兜底）
    result = await orchestrate(
        req.question,
        skill=req.skill,
        top_k=req.top_k,
        enable_self_retrieval=req.enable_self_retrieval,
        temperature=req.temperature,
        mode=req.mode,
        history=req.history,
        session_id=req.session_id,
    )
    latency_ms = (time.time() - start) * 1000

    token_usage = result.get("token_usage") or {}
    _log_qa_async(
        question=req.question,
        answer=result.get("answer", ""),
        skill=result.get("skill", ""),
        kb_id=result.get("kb_id"),
        routing_source=result.get("routing_source"),
        degradation_level=result.get("degradation_level", 0),
        latency_ms=latency_ms,
        tokens_total=token_usage.get("total_tokens", 0),
        sources=result.get("sources", []),
    )

    return QAResponse(
        answer=result["answer"],
        sources=[
            SourceDocument(
                id=d.get("id", ""),
                chunk_id=d.get("chunk_id", ""),
                doc_id=d.get("doc_id", ""),
                content=d.get("content", ""),
                score=d.get("score", 0.0),
            )
            for d in result.get("sources", [])
        ],
        skill=result.get("skill", ""),
        kb_id=result.get("kb_id"),
        routing_source=result.get("routing_source"),
        degradation_level=result.get("degradation_level", 0),
        latency_breakdown=result.get("latency_breakdown", {}),
        retrieval_meta=result.get("retrieval_meta", {}),
        latency_ms=round(latency_ms, 2),
        token_usage=TokenUsage(**token_usage) if token_usage else None,
        retrieval_rounds=result.get("retrieval_rounds", 1),
        rewritten_queries=result.get("rewritten_queries", []),
        qa_metrics=result.get("qa_metrics", {}),
        graph_context=result.get("graph_context") or None,  # 契约修复: 此前 schema 声明但 route 从未传入, 恒 null
    )


@router.get("/eval-summary")
async def get_eval_summary():
    """离线评估整体指标 (由 scripts/evaluate.py 生成)。"""
    import json
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "eval-summary.json")
    if not os.path.exists(path):
        return {"summary": None, "cases": []}

    def _load_summary() -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return await asyncio.to_thread(_load_summary)


@router.post("/ask/stream")
@_rate_limited
async def ask_question_stream(req: QARequest):
    """SSE 流式问答 — 逐块返回 answer，首 token 后即可渲染。

    事件流: meta → sources → delta(多) → done。
    保留 JSON `/ask` 端点不变（向后兼容 + 非流式客户端）。
    """

    start = time.time()

    async def _event_stream():
        meta, done, sources = {}, {}, []
        try:
            async for ev in orchestrate_stream(
                req.question,
                skill=req.skill,
                top_k=req.top_k,
                enable_self_retrieval=req.enable_self_retrieval,
                temperature=req.temperature,
                mode=req.mode,
                history=req.history,
                session_id=req.session_id,
            ):
                # 流式协议 meta/done 分两个事件, 补 QA 日志需从 meta 取路由字段、done 取答案/用量
                if ev.get("type") == "meta":
                    meta = ev
                elif ev.get("type") == "sources":
                    sources = ev.get("sources", [])
                elif ev.get("type") == "done":
                    done = ev
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001 — SSE 兜底: 客户端能区分正常 done / 异常中断
            logger.error("流式问答中断: %s", str(e)[:200])
            yield f"data: {json.dumps({'type': 'error', 'detail': '生成中断，请重试'}, ensure_ascii=False)}\n\n"

        if done:
            usage = done.get("token_usage") or {}
            _log_qa_async(
                question=req.question,
                answer=done.get("answer", ""),
                skill=meta.get("skill", ""),
                kb_id=meta.get("kb_id"),
                routing_source=meta.get("routing_source"),
                degradation_level=done.get("degradation_level", 0),
                latency_ms=int((time.time() - start) * 1000),
                tokens_total=usage.get("total_tokens", 0),
                sources=sources,
            )

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """读取会话历史(用于前端刷新后恢复上下文)。"""
    return {"session_id": session_id, "history": [t.model_dump() for t in load_session(session_id)]}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """清除会话历史(对应前端『清空对话』)。"""
    clear_session(session_id)
    return {"session_id": session_id, "cleared": True}


# 后台任务集合: asyncio.create_task 不存引用可能被 GC 中途取消 (事件循环无强引用)
_bg_tasks: set[asyncio.Task] = set()


def _log_qa_async(
    question, answer, skill, kb_id, routing_source, degradation_level, latency_ms, tokens_total, sources=None
):
    """异步写 QA 日志（fire-and-forget，不阻塞响应）。"""

    def _write():
        get_document_store().log_qa(
            question, answer, skill, kb_id, routing_source, degradation_level, int(latency_ms), tokens_total, sources
        )

    try:
        task = asyncio.create_task(asyncio.to_thread(_write))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)
    except Exception as e:  # noqa: BLE001 — 异步日志失败不影响响应
        logger.warning("QA 日志任务创建失败: %s", str(e)[:100])
