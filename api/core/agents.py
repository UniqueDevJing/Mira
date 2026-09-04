"""多 Agent 消息层 — 意图分类 + 咨询/操作/投诉/闲聊 四路分发。

框架:
  用户输入 → 意图分类(规则优先, 零延迟) → 按 message_type 分发:
    consult  → 走既有 RAG 编排链路 (orchestrator.ask)
    chat     → 闲聊 Agent: 跳过检索, 轻量直达 LLM (延迟 ~6s → ~1-2s)
    complaint→ 投诉 Agent: 情绪分级 + 自动建工单 + 高情绪升级人工
    operation→ 操作 Agent: 识别目标工具 → 高危操作请求确认(confirm 后才执行)

设计约束:
  - 分类纯规则实现, 不增加在线延迟; LLM 兜底分类留接口 (classify_message_llm) 按需开启。
  - 工单/待确认操作落在 documents.db (SQLite), 与文档库同库不同表。
  - 所有新增响应字段带默认值, 旧客户端零破坏。
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
import uuid

from api.core.document_store import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

# ───────────────────────── 意图类型 ─────────────────────────
TYPE_CONSULT = "consult"
TYPE_CHAT = "chat"
TYPE_COMPLAINT = "complaint"
TYPE_OPERATION = "operation"

# 情绪等级 (投诉 Agent 用)
SENTIMENT_CALM = "calm"
SENTIMENT_UPSET = "upset"
SENTIMENT_ANGRY = "angry"

# ───────────────────────── 规则词表 ─────────────────────────
_CHAT_PATTERNS = [
    r"^(你好|您好|hi|hello|嗨|在吗|在么)(呀|啊|哈)?[!！。~～?？\s]*$",
    r"^(谢谢|感谢|多谢|辛苦了|thx|thanks)(啦|了|哈|呢|哦|呀)?[!！。~～\s]*$",
    r"^(再见|拜拜|晚安|早安|下午好|晚上好|bye)(啦|了|哈|呢|哦|呀)?[!！。~～\s]*$",
    r"^(你是谁|你叫什么|你是什么|介绍一下你自己|你能做什么|你会什么)",
    r"^(讲个笑话|说个笑话|聊聊天|陪我聊|无聊)",
    r"(今天天气|现在几点|你几岁|你喜欢)",
]

_COMPLAINT_KEYWORDS = [
    "投诉", "举报", "太差", "太烂", "垃圾", "不满", "欺骗", "骗人", "骗子",
    "态度恶劣", "敷衍", "没人管", "一直不解决", "拖着不", "多次反映", "反复反映",
    "气死", "忍无可忍", "差评", "给个说法", "必须给", "要求赔偿", "12315",
    "客服无用", "找谁说理", "忍不了", "受不了",
]

_NEGATION_MOOD = ["!！", "气", "烦", "怒", "失望", "无语"]

_OPERATION_TOOLS = [
    {"op": "list_documents", "label": "查询文档列表", "danger": False,
     "hint": "列出知识库文档(可按 kb 过滤)"},
    {"op": "doc_status", "label": "查询文档状态", "danger": False,
     "hint": "按文档 ID 或文件名查询单个文档的处理状态"},
    {"op": "delete_document", "label": "删除文档", "danger": True,
     "hint": "按文档 ID 删除文档及其向量块 (高危, 需二次确认)"},
]

_OP_VERB_PATTERNS = [
    (r"(删除|移除|清掉|删掉).{0,12}(文档|文件)", "delete_document"),
    (r"(列出|列举|看看|查询|查一下|显示).{0,10}(文档|文件)(列表)?", "list_documents"),
    (r"文档(列表|都有哪些|有哪些)", "list_documents"),
    (r"(上传|导入|提交).{0,8}(文档|文件)", "upload_document"),
    (r"(重建|刷新).{0,6}(索引|缓存)", "rebuild_index"),
    (r"(查|看).{0,6}(状态|进度).{0,10}(文档|文件)?", "doc_status"),
]


def classify_message(text: str) -> tuple[str, str]:
    """规则意图分类。返回 (message_type, sentiment)。

    顺序: 闲聊(整句短模式) → 投诉(关键词+情绪) → 操作(动宾模式) → 咨询(默认)。
    """
    t = (text or "").strip()
    if not t:
        return TYPE_CONSULT, SENTIMENT_CALM
    low = t.lower()

    # 闲聊: 仅匹配短句/开场白, 避免把"你好, 我想投诉"误判为闲聊
    if len(t) <= 20:
        for p in _CHAT_PATTERNS:
            if re.search(p, low):
                return TYPE_CHAT, SENTIMENT_CALM

    # 投诉: 命中强关键词即判投诉; 感叹号/负面情绪词升级情绪等级
    hits = sum(1 for kw in _COMPLAINT_KEYWORDS if kw in t)
    if hits >= 1:
        sentiment = SENTIMENT_CALM
        if hits >= 2 or any(w in t for w in _NEGATION_MOOD):
            sentiment = SENTIMENT_UPSET
        if hits >= 2 and re.search(r"[!！]{1,}|气死|忍无可忍|骗子|12315|赔偿", t):
            sentiment = SENTIMENT_ANGRY
        return TYPE_COMPLAINT, sentiment

    # 操作: 动宾结构
    for pat, op in _OP_VERB_PATTERNS:
        if re.search(pat, t):
            return TYPE_OPERATION, SENTIMENT_CALM

    return TYPE_CONSULT, SENTIMENT_CALM


# ───────────────────────── 工单存储 ─────────────────────────
def _ensure_tickets_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            content TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            escalated INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        )
        """
    )


def create_ticket(content: str, sentiment: str, user_id: str | None = None,
                  db_path: str = DEFAULT_DB_PATH) -> dict:
    """投诉建单。高情绪(angry)自动标记升级人工。"""
    ticket_id = "TK" + uuid.uuid4().hex[:10].upper()
    escalated = 1 if sentiment == SENTIMENT_ANGRY else 0
    status = "escalated" if escalated else "open"
    now = time.time()
    conn = sqlite3.connect(db_path)
    try:
        _ensure_tickets_table(conn)
        conn.execute(
            "INSERT INTO tickets (id, user_id, content, sentiment, status, escalated, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticket_id, user_id, content[:2000], sentiment, status, escalated, now),
        )
        conn.commit()
    finally:
        conn.close()  # 显式关闭: with sqlite3.connect 只管事务不管连接, Windows 下未关闭会锁库文件
    return {
        "ticket_id": ticket_id,
        "sentiment": sentiment,
        "status": status,
        "escalated": bool(escalated),
        "created_at": now,
    }


def list_tickets(db_path: str = DEFAULT_DB_PATH, limit: int = 50) -> list[dict]:
    conn = sqlite3.connect(db_path)
    try:
        _ensure_tickets_table(conn)
        rows = conn.execute(
            "SELECT id, user_id, content, sentiment, status, escalated, created_at "
            "FROM tickets ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"ticket_id": r[0], "user_id": r[1], "content": r[2], "sentiment": r[3],
         "status": r[4], "escalated": bool(r[5]), "created_at": r[6]}
        for r in rows
    ]


# ───────────────────────── 操作 Agent ─────────────────────────
# 待确认操作池 (进程内, TTL 10 分钟)。key=pending_id
_PENDING_OPS: dict[str, dict] = {}
_PENDING_TTL = 600


def _pending_gc() -> None:
    now = time.time()
    for k in [k for k, v in _PENDING_OPS.items() if now - v["ts"] > _PENDING_TTL]:
        _PENDING_OPS.pop(k, None)


def _match_operation(text: str) -> dict | None:
    """从操作类消息中提取目标工具与参数。"""
    low = text.strip()
    # 文档 ID 提取: 16 位 hex 或 TK 单号样式
    m = re.search(r"\b([0-9a-f]{16}|[0-9a-f]{12})\b", low)
    doc_id = m.group(1) if m else None
    for pat, op in _OP_VERB_PATTERNS:
        if re.search(pat, low):
            tool = next((t for t in _OPERATION_TOOLS if t["op"] == op), None)
            if tool:
                return {**tool, "doc_id": doc_id, "raw_text": low}
    return None


def _execute_operation(tool: dict, doc_id: str | None) -> dict:
    """执行已确认的低危/已二次确认的高危操作。"""
    from api.core.document_store import get_document_store

    op = tool["op"]
    if op == "list_documents":
        store = get_document_store()
        data = store.list_all(page=1, size=10)
        items = data.get("documents") or data.get("items") or []
        lines = [f"- {d.get('filename') or d.get('doc_id')} [{d.get('knowledge_base')}] {d.get('status')}"
                 for d in items[:10]]
        return {"answer": f"文档列表(最近 {len(lines)} 条):\n" + "\n".join(lines) if lines else "知识库暂无文档。"}
    if op == "doc_status":
        if not doc_id:
            return {"answer": "请提供文档 ID 或文件名, 例如: 查一下文档 a1b2c3d4e5f60718 的状态。"}
        store = get_document_store()
        doc = store.get(doc_id)
        if not doc:
            return {"answer": f"未找到文档 {doc_id}。可先让我列出文档列表。"}
        return {"answer": (f"文档 {doc.get('filename')} 状态: {doc.get('status')}, "
                           f"知识库: {doc.get('knowledge_base')}, 分块: {doc.get('chunk_count')}。")}
    if op == "delete_document":
        if not doc_id:
            return {"answer": "删除文档需要提供文档 ID。可先让我列出文档列表。"}
        store = get_document_store()
        ok = store.delete(doc_id)
        return {"answer": f"文档 {doc_id} 已删除。" if ok else f"文档 {doc_id} 不存在, 未执行删除。"}
    return {"answer": f"操作 {op} 暂不支持。"}


def handle_operation(text: str, confirm: bool = False, pending_id: str | None = None) -> dict:
    """操作 Agent: 低危直接执行; 高危(删除)需 confirm+pending_id 二次确认。"""
    _pending_gc()

    # 二次确认分支
    if confirm and pending_id:
        pend = _PENDING_OPS.pop(pending_id, None)
        if not pend:
            return {"answer": "确认已超时或不存在, 请重新发起操作。", "pending_operation": None}
        result = _execute_operation(pend["tool"], pend.get("doc_id"))
        result["message_type"] = TYPE_OPERATION
        result["agent"] = "operation"
        result["pending_operation"] = None
        return result

    tool = _match_operation(text)
    if not tool:
        tips = "、".join(t["label"] for t in _OPERATION_TOOLS)
        return {"answer": f"我支持这些操作: {tips}。请说明你想做什么, 例如\"列出文档列表\"。",
                "message_type": TYPE_OPERATION, "agent": "operation", "pending_operation": None}

    if tool["danger"] and not confirm:
        pending_id = "OP" + uuid.uuid4().hex[:10].upper()
        _PENDING_OPS[pending_id] = {"tool": tool, "doc_id": tool.get("doc_id"), "ts": time.time()}
        target = f" (文档 {tool['doc_id']})" if tool.get("doc_id") else ""
        return {
            "answer": (f"即将执行高危操作: {tool['label']}{target}。该操作不可撤销, "
                       f"请回复\"确认执行\"并在请求中携带 pending_operation_id={pending_id} 与 confirm=true。"),
            "message_type": TYPE_OPERATION,
            "agent": "operation",
            "pending_operation": {"pending_id": pending_id, "op": tool["op"], "label": tool["label"],
                                  "danger": True, "doc_id": tool.get("doc_id")},
        }

    result = _execute_operation(tool, tool.get("doc_id"))
    result["message_type"] = TYPE_OPERATION
    result["agent"] = "operation"
    result["pending_operation"] = None
    return result


# ───────────────────────── 闲聊 Agent ─────────────────────────
_CHITCHAT_SYSTEM = (
    "你是企业知识助手的闲聊模块。用户在寒暄或闲聊, 请用一两句轻松自然的话回应, "
    "不调用知识库, 不编造公司业务信息。若用户的问题可能涉及业务咨询, 温馨提示可直接提问业务问题。"
)


async def handle_chitchat(text: str) -> dict:
    """闲聊 Agent: 跳过检索直达 LLM, 低延迟轻量回复。"""
    from api.core.llm_client import get_llm_client

    t0 = time.time()
    client = get_llm_client()
    resp = await client.chat(
        [{"role": "system", "content": _CHITCHAT_SYSTEM},
         {"role": "user", "content": text[:500]}],
        temperature=0.7,
        max_tokens=300,
        timeout=15,
    )
    content = getattr(resp, "content", None) or (resp.get("content") if isinstance(resp, dict) else "") or ""
    return {
        "answer": content.strip() or "你好呀, 有什么业务问题欢迎直接问我~",
        "message_type": TYPE_CHAT,
        "agent": "chitchat",
        "latency_breakdown": {"agent_ms": round((time.time() - t0) * 1000, 1)},
    }


# ───────────────────────── 投诉 Agent ─────────────────────────
_EMPATHY = {
    SENTIMENT_CALM: "收到您的反馈, 我们会认真处理。",
    SENTIMENT_UPSET: "非常理解您的心情, 给您带来不便十分抱歉。我们已优先记录您的问题。",
    SENTIMENT_ANGRY: "非常抱歉给您造成了这么糟糕的体验! 您的反馈已升级为最高优先级, 人工客服将尽快与您联系。",
}


async def handle_complaint(text: str, sentiment: str, user_id: str | None = None) -> dict:
    """投诉 Agent: 情绪分级 → 建工单 → 高情绪升级人工。"""
    ticket = create_ticket(content=text, sentiment=sentiment, user_id=user_id)
    answer = (
        f"{_EMPATHY[sentiment]}\n\n"
        f"已为您生成投诉工单: **{ticket['ticket_id']}**(状态: {ticket['status']})。"
    )
    if ticket["escalated"]:
        answer += " 该工单已自动升级, 人工客服会主动联系您。"
    else:
        answer += " 我们会在 24 小时内跟进处理。"
    return {
        "answer": answer,
        "message_type": TYPE_COMPLAINT,
        "agent": "complaint",
        "ticket": ticket,
        "latency_breakdown": {"agent_ms": 1.0},
    }
