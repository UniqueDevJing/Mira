"""知识问答 API — 入口交给 Router + Skill 编排层"""

import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.core.auth import KBForbiddenError, get_principal
from api.core.document_store import get_document_store
from api.core.limiter import limiter
from api.core.orchestrator import ask as orchestrate
from api.core.orchestrator import ask_stream as orchestrate_stream
from api.core.session_store import clear_session, load_session, session_owner
from api.schemas.qa import QARequest, QAResponse, TokenUsage, to_source_document
from api.state import get_vector_store
from engines.router.routing_rules import SKILLS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/qa", tags=["qa"])


def _rate_limited(fn):
    """限流开启时附加 slowapi 限制, 未开启原样返回 (防本地/测试误触)。"""
    if limiter is None:
        return fn
    from api.config import settings

    return limiter.limit(f"{settings.rate_limit_per_minute}/minute")(fn)


def _merge_image_text(image_base64: str, question: str) -> tuple[str, str]:
    """图片输入 → 视觉理解/OCR → 并入问题。返回 (合并后问题, 识别文字)。

    双链路:
      1) VL 优先: 配置 RAG_LLM_VL_MODEL (如 qwen-vl-plus) 时走视觉模型理解图片内容 —
         能回答"图里画的是什么/图表数据/截图里的操作步骤", 不止文字提取;
      2) OCR 兜底: VL 未配置或调用失败时回落 RapidOCR 纯文字提取。
    失败不静默: 多模态入口失败让用户知道原因, 而非当无图处理。
    """
    import base64
    import binascii

    try:
        raw = base64.b64decode(image_base64, validate=False)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"图片 base64 解码失败: {e}") from e
    if not raw:
        raise HTTPException(status_code=400, detail="图片内容为空。")

    # ── 链路 1: 视觉模型理解 (qwen-vl) ──
    from api.config import settings

    if settings.llm_vl_model:
        try:
            from api.core.llm_client import get_sync_llm_client

            mime = "image/png"
            if raw[:3] == b"\xff\xd8\xff":
                mime = "image/jpeg"
            elif raw[:8] == b"\x89PNG\r\n\x1a\n":
                mime = "image/png"
            elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
                mime = "image/webp"
            data_url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
            prompt = (
                "请仔细观察这张图片, 提取其中所有与用户提问相关的信息: "
                "若有文字请完整转录, 若是图表/截图/照片请客观描述其内容。"
                + (f"\n用户想了解: {question}" if question.strip() else "")
            )
            client = get_sync_llm_client(vl_model=True)
            resp = client.chat(
                [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ]}],
                temperature=0.1,
                max_tokens=1000,
                timeout=45,
            )
            content = getattr(resp, "content", None)
            if content and str(content).strip():
                vl_text = str(content).strip()
                merged = f"{question}\n[图片内容] {vl_text[:2000]}" if question.strip() else f"[图片内容] {vl_text[:2000]}"
                return merged.strip(), vl_text
        except Exception as e:  # noqa: BLE001 — VL 失败回落 OCR, 不中断
            import logging

            logging.getLogger(__name__).warning("视觉模型理解失败, 回落 OCR: %s", str(e)[:160])

    return _ocr_extract(raw, question)


def _ocr_extract(raw: bytes, question: str) -> tuple[str, str]:
    """OCR 兜底链路: RapidOCR 纯文字提取。"""
    import numpy as np

    try:
        import cv2

        from engines.parsing.ocr import _ensure_rapidocr, _get_ocr

        _ensure_rapidocr()
        ocr = _get_ocr()
    except ImportError as e:
        raise HTTPException(
            status_code=400,
            detail=f"图片理解依赖 OCR 组件未安装 ({e.__class__.__name__}), 请直接输入文本问题, 或安装 rapidocr_onnxruntime 后重试。",
        ) from e

    import cv2

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="图片格式无法解析, 请提供 PNG/JPG。")

    result, _ = ocr(img)
    # RapidOCR 各版本返回结构不一: [(box, text, conf)...] (conf 可能为 str) / [text...] / str / dict。
    # 统一防御性解析, 只取文字。
    texts: list[str] = []
    for item in (result or []):
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            t = item.get("text") or ""
            if t:
                texts.append(t)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            t = item[1]
            c = item[2] if len(item) > 2 else 1.0
            try:
                ok = float(c) >= 0.3
            except (TypeError, ValueError):
                ok = True  # 无置信度信息时保留
            if t and ok:
                texts.append(str(t))
    ocr_text = " ".join(texts).strip()
    if not ocr_text:
        return question, ""  # 图里没字, 退化为纯文本问题
    merged = f"{question}\n[图片文字] {ocr_text[:2000]}" if question else f"[图片文字] {ocr_text[:2000]}"
    return merged.strip(), ocr_text


async def _remember_async(user_id: str, question: str, answer: str) -> None:
    """长期记忆异步写入: 永不抛出, 失败仅记日志 (记忆层整体降级安全)。"""
    try:
        from api.core.memory_layer import remember

        remember(user_id, question, answer)
    except Exception as e:  # noqa: BLE001 — 记忆写入绝不可影响问答主流程
        logger.warning("长期记忆异步写入异常(忽略): %s", str(e)[:160])


def _vision_answer(question: str, image_description: str, image_b64: str,
                   allowed_kbs: list[str] | None, top_k: int = 4) -> str:
    """图片场景端到端回答: VL 看图 + 知识库片段增强 → 综合作答。

    检索: 用户原问题在全部授权 KB 内检索 (VL 描述已含图片信息, 问题保持用户语义)。
    生成: VL 二次调用, 基于图片内容 + 知识库片段回答; 片段不相关时如实说明, 不编造。
    """

    from api.core.llm_client import get_sync_llm_client
    from api.state import get_embedder, get_vector_store

    # 1) 知识库片段: 授权范围内逐库检索
    from engines.doc_types import RAG_KBS

    kbs = [k for k in (allowed_kbs if allowed_kbs is not None else RAG_KBS)
           if allowed_kbs is None or k in RAG_KBS]
    snippets: list[str] = []
    if question.strip():
        try:
            emb = get_embedder().embed_query(question[:500])
            for kb in kbs[:10]:
                try:
                    for hit in get_vector_store(kb).search(emb, top_k=2):
                        c = (hit.get("content") or "").strip()
                        if c and hit.get("score", 0) >= 0.55:
                            snippets.append(f"[{kb}] {c[:300]}")
                except Exception as e:  # noqa: BLE001 — 单库失败跳过
                    logger.debug("视觉问答片段检索跳过 KB %s: %s", kb, str(e)[:100])
                    continue
        except Exception as e:  # noqa: BLE001 — 检索失败则纯看图作答
            logger.warning("视觉问答检索片段失败(降级纯看图): %s", str(e)[:120])

    kb_block = "\n".join(snippets[:6]) if snippets else "(知识库中未检索到相关片段)"
    data_url = "data:image/png;base64," + image_b64
    prompt = (
        "你是企业智能客服。用户上传了一张图片并提问。\n"
        f"图片内容描述: {image_description[:800]}\n\n"
        f"知识库参考片段:\n{kb_block[:2000]}\n\n"
        f"用户问题: {question.strip() or '请解读这张图片'}\n\n"
        "请综合图片内容与知识库片段回答用户问题。规则:\n"
        "- 优先使用知识库中与问题相关的信息, 引用格式 [来源];\n"
        "- 图片中的信息可直接使用;\n"
        "- 若知识库与图片都无法回答, 如实说明, 不得编造。"
    )
    client = get_sync_llm_client(vl_model=True)
    if client is None:
        raise RuntimeError("视觉模型未配置")
    resp = client.chat(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]}],
        temperature=0.2,
        max_tokens=1200,
        timeout=60,
    )
    answer = (getattr(resp, "content", None) or "").strip()
    if not answer:
        raise RuntimeError("视觉模型返回空")
    return answer


@router.get("/tickets")
async def list_tickets_endpoint(request: Request, limit: int = 50):
    """投诉工单列表 (运维/客服后台用)。"""
    from api.core.agents import list_tickets

    principal = get_principal(request)
    if principal.allowed_kbs is not None and not principal.allowed_kbs:
        raise HTTPException(status_code=403, detail="无权限查看工单")
    return {"tickets": list_tickets(limit=max(1, min(limit, 200)))}


@router.post("/ask", response_model=QAResponse)
@_rate_limited
async def ask_question(req: QARequest, request: Request):
    start = time.time()

    # RBAC: 按 principal 授权范围做知识库级拦截
    principal = get_principal(request)
    if req.skill and principal.allowed_kbs is not None:
        kb = SKILLS.get(req.skill, {}).get("kb")
        if kb is not None and kb not in principal.allowed_kbs:
            raise HTTPException(status_code=403, detail=f"该 API Key 无权访问知识库: {req.skill}")

    # ── 多模态输入: 视觉理解 (qwen-vl) / OCR 兜底 ──
    question = req.question
    ocr_text = ""
    if req.image_base64:
        question, ocr_text = _merge_image_text(req.image_base64, question)
        if not question.strip():
            raise HTTPException(status_code=400, detail="图片未能提取有效内容, 请直接输入文本问题或更换图片。")

        # ── 图片场景走"看图 + 知识库增强"端到端链路: 不进文字检索主链路 ──
        # (VL 已理解图片; 检索 query 用用户原问题拿知识库参考片段, 由 VL 综合图+片段作答。
        #  避开低相关拒答护栏 — 用户传图必有意图, 片段是否相关由 VL 判断而非文字匹配度)
        try:
            vision_answer = _vision_answer(req.question, ocr_text, req.image_base64, principal.allowed_kbs)
        except Exception as e:  # noqa: BLE001 — VL 链路失败回落文字主链路
            logger.warning("视觉问答链路失败, 回落文字检索: %s", str(e)[:160])
        else:
            latency_ms = (time.time() - start) * 1000
            _log_qa_async(
                question=question[:500],
                answer=vision_answer,
                skill="vision",
                kb_id=None,
                routing_source="vision",
                degradation_level=0,
                latency_ms=latency_ms,
                tokens_total=0,
                sources=[],
            )
            return QAResponse(
                answer=vision_answer,
                message_type="consult",
                agent="vision",
                ocr_text=ocr_text,
                latency_ms=round(latency_ms, 2),
                latency_breakdown={"vision_ms": round(latency_ms, 1)},
            )

    # ── 意图分类 (规则, 零延迟) → 四路分发: 咨询走 RAG, 其余由对应 Agent 接管 ──
    from api.core.agents import (
        TYPE_CHAT,
        TYPE_COMPLAINT,
        TYPE_OPERATION,
        classify_message,
        handle_chitchat,
        handle_complaint,
        handle_operation,
    )

    message_type, sentiment = classify_message(question)
    agent_result: dict | None = None

    # 前端切换条: force_agent 优先于自动分类 (consult=强制 RAG, 其余=指定 Agent)
    if req.force_agent == TYPE_CHAT:
        message_type = TYPE_CHAT
    elif req.force_agent == TYPE_COMPLAINT:
        message_type = TYPE_COMPLAINT
        sentiment = "upset"
    elif req.force_agent == TYPE_OPERATION:
        message_type = TYPE_OPERATION
    elif req.force_agent == "consult":
        message_type = "consult"

    if req.confirm_operation or req.pending_operation_id:
        # 二次确认流: 无论分类结果直接进操作 Agent 确认分支
        message_type = TYPE_OPERATION
        agent_result = handle_operation(question, confirm=req.confirm_operation,
                                        pending_id=req.pending_operation_id)
    elif message_type == TYPE_CHAT and not req.skill:
        agent_result = await handle_chitchat(question)
    elif message_type == TYPE_COMPLAINT and not req.skill:
        agent_result = await handle_complaint(question, sentiment, user_id=principal.key_id)
    elif message_type == TYPE_OPERATION and not req.skill:
        agent_result = handle_operation(question)

    if agent_result is not None:
        agent_latency = (time.time() - start) * 1000
        _log_qa_async(
            question=question,
            answer=agent_result.get("answer", ""),
            skill=agent_result.get("agent", ""),
            kb_id=None,
            routing_source="agent",
            degradation_level=0,
            latency_ms=agent_latency,
            tokens_total=0,
            sources=[],
        )
        return QAResponse(
            answer=agent_result["answer"],
            message_type=agent_result.get("message_type", message_type),
            agent=agent_result.get("agent", "agent"),
            ticket=agent_result.get("ticket"),
            pending_operation=agent_result.get("pending_operation"),
            latency_ms=round(agent_latency, 2),
            latency_breakdown=agent_result.get("latency_breakdown", {}),
        )

    # ── 长期记忆层: 提问前召回该用户相关历史, 注入为对话历史 ──
    memory_used: list[dict] = []
    history_turns: list = list(req.history)
    if principal.key_id:
        from api.core.memory_layer import recall

        memory_used = recall(principal.key_id, question, top_k=3)
    if memory_used:
        from api.schemas.qa import ChatTurn

        injected: list = []
        for m in memory_used:
            injected.append(ChatTurn(role="user", content=f"[历史提问] {m['question']}"))
            injected.append(ChatTurn(role="assistant", content=f"[历史回答] {m['answer']}"))
        history_turns = injected + history_turns

    # 编排层: Router 路由 → Skill 执行（分阶段超时/降级/跨库兜底）
    try:
        result = await orchestrate(
            question,
            skill=req.skill,
            top_k=req.top_k,
            enable_self_retrieval=req.enable_self_retrieval,
            temperature=req.temperature,
            mode=req.mode,
            history=history_turns,
            session_id=req.session_id,
            allowed_kbs=principal.allowed_kbs,
            owner=principal.key_id,  # S6: session 归属绑定
        )
    except KBForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e))
    latency_ms = (time.time() - start) * 1000

    token_usage = result.get("token_usage") or {}

    # ── 长期记忆写入 (异步, 失败仅记日志不阻断响应) ──
    if principal.key_id and result.get("answer"):
        asyncio.get_running_loop().create_task(
            _remember_async(principal.key_id, question, result.get("answer", ""))
        )

    _log_qa_async(
        question=question,
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
        sources=[to_source_document(d) for d in result.get("sources", [])],
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
        message_type=message_type,
        agent="rag",
        memory_used=memory_used,
        ocr_text=ocr_text,
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
async def ask_question_stream(req: QARequest, request: Request):
    """SSE 流式问答 — 逐块返回 answer，首 token 后即可渲染。

    事件流: meta → sources → delta(多) → done。
    保留 JSON `/ask` 端点不变（向后兼容 + 非流式客户端）。
    """
    principal = get_principal(request)
    if req.skill and principal.allowed_kbs is not None:
        kb = SKILLS.get(req.skill, {}).get("kb")
        if kb is not None and kb not in principal.allowed_kbs:
            raise HTTPException(status_code=403, detail=f"该 API Key 无权访问知识库: {req.skill}")

    start = time.time()

    # ── 多 Agent 分发 (流式): 闲聊/投诉/操作由对应 Agent 接管, 合成 SSE 事件流 ──
    from api.core.agents import (
        TYPE_CHAT,
        TYPE_COMPLAINT,
        TYPE_OPERATION,
        classify_message,
        handle_chitchat,
        handle_complaint,
        handle_operation,
    )

    _mt, _sentiment = classify_message(req.question)
    _agent_result: dict | None = None
    # 前端切换条: force_agent 优先于自动分类
    if req.force_agent == TYPE_CHAT:
        _mt = TYPE_CHAT
    elif req.force_agent == TYPE_COMPLAINT:
        _mt = TYPE_COMPLAINT
        _sentiment = "upset"
    elif req.force_agent == TYPE_OPERATION:
        _mt = TYPE_OPERATION
    elif req.force_agent == "consult":
        _mt = "consult"
    if req.confirm_operation or req.pending_operation_id:
        _agent_result = handle_operation(req.question, confirm=req.confirm_operation,
                                         pending_id=req.pending_operation_id)
    elif _mt == TYPE_CHAT and not req.skill:
        _agent_result = await handle_chitchat(req.question)
    elif _mt == TYPE_COMPLAINT and not req.skill:
        _agent_result = await handle_complaint(req.question, _sentiment, user_id=principal.key_id)
    elif _mt == TYPE_OPERATION and not req.skill:
        _agent_result = handle_operation(req.question)

    if _agent_result is not None:
        _agent_answer = _agent_result.get("answer", "")

        async def _agent_event_stream():
            try:
                yield f"data: {json.dumps({'type': 'meta', 'message_type': _agent_result.get('message_type', _mt), 'agent': _agent_result.get('agent', 'agent'), 'skill': '', 'kb_id': None, 'routing_source': 'agent'}, ensure_ascii=False)}\n\n"
                # 逐段推送, 前端按 delta 渲染
                step = max(1, len(_agent_answer) // 4)
                for i in range(0, len(_agent_answer), step):
                    yield f"data: {json.dumps({'type': 'delta', 'content': _agent_answer[i:i + step]}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'answer': _agent_answer, 'message_type': _agent_result.get('message_type', _mt), 'agent': _agent_result.get('agent', 'agent'), 'ticket': _agent_result.get('ticket'), 'pending_operation': _agent_result.get('pending_operation'), 'degradation_level': 0}, ensure_ascii=False)}\n\n"
            except Exception:  # noqa: BLE001
                yield f"data: {json.dumps({'type': 'error', 'detail': 'Agent 处理中断'}, ensure_ascii=False)}\n\n"
            _log_qa_async(
                question=req.question,
                answer=_agent_answer,
                skill=_agent_result.get("agent", ""),
                kb_id=None,
                routing_source="agent",
                degradation_level=0,
                latency_ms=int((time.time() - start) * 1000),
                tokens_total=0,
                sources=[],
            )

        return StreamingResponse(_agent_event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

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
                allowed_kbs=principal.allowed_kbs,
                owner=principal.key_id,  # S6: session 归属绑定
            ):
                # 流式协议 meta/done 分两个事件, 补 QA 日志需从 meta 取路由字段、done 取答案/用量
                if ev.get("type") == "meta":
                    meta = ev
                    ev.setdefault("message_type", "consult")  # 契约一致: 咨询路径也带意图字段
                    ev.setdefault("agent", "rag")
                elif ev.get("type") == "sources":
                    sources = ev.get("sources", [])
                elif ev.get("type") == "done":
                    done = ev
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except KBForbiddenError as e:
            logger.warning("流式问答 RBAC 拒绝: %s", str(e)[:120])
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"
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


def _check_session_ownership(request: Request, session_id: str) -> None:
    """S6 IDOR 防护: 会话归属校验 — 非 admin 且非创建者禁止读写他人会话。

    无归属(旧数据/未绑定)视为宽松兼容: 仅 admin 可访问; reader 一律拒绝。
    """
    principal = get_principal(request)
    if principal.is_admin():
        return
    owner = session_owner(session_id)
    if owner is None or principal.key_id != owner:
        raise HTTPException(status_code=403, detail="无权访问该会话 (会话归属其他 Key)")


@router.get("/session/{session_id}")
async def get_session(session_id: str, request: Request):
    """读取会话历史(用于前端刷新后恢复上下文)。S6: 归属校验防 IDOR 越权读。"""
    _check_session_ownership(request, session_id)
    return {"session_id": session_id, "history": [t.model_dump() for t in load_session(session_id)]}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """清除会话历史(对应前端『清空对话』)。S6: 归属校验防 IDOR 越权删。"""
    _check_session_ownership(request, session_id)
    clear_session(session_id)
    return {"session_id": session_id, "cleared": True}


@router.get("/sources/{doc_id}")
async def get_source_detail(doc_id: str, kb: str | None = None, request: Request = None):
    """U2 来源展开全文 —— 按 (doc_id, kb) 拉取该文档全部 chunk 完整内容。

    消费 U1 拒答候选(RefusalCandidate.doc_id + kb)。过鉴权: KB 级 RBAC,
    越权直接 403; doc 不存在/无内容回 404(不泄露跨库存在性)。

    入参:
      doc_id  — 必填(path), 来自 U1 候选
      kb      — 必填(query), 候选所属知识库(用于 RBAC 与定位向量表)
    出参: {doc_id, kb, chunk_count, chunks:[{id,chunk_id,doc_id,content,title_chain,doc_title,file_name,update_time,parent_id}]}
    """
    if not kb:
        raise HTTPException(status_code=400, detail="kb 参数必填 (来源所属知识库)")
    principal = get_principal(request)
    if not principal.can_access_kb(kb):
        raise HTTPException(status_code=403, detail=f"该 API Key 无权访问知识库: {kb}")
    try:
        store = get_vector_store(kb)
        chunks = store.get_by_doc_id(doc_id)
    except Exception as e:  # noqa: BLE001 — 查询异常统一降级为 404, 不暴露内部错误
        logger.warning("来源展开查询失败 doc_id=%s kb=%s: %s", doc_id[:32], kb, str(e)[:120])
        raise HTTPException(status_code=404, detail="未找到该文档或查询失败")
    if not chunks:
        raise HTTPException(status_code=404, detail="未找到该文档")
    return {"doc_id": doc_id, "kb": kb, "chunk_count": len(chunks), "chunks": chunks}


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
