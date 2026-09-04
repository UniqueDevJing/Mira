"""多 Agent 框架测试: 意图分类 / 闲聊 / 投诉建单 / 操作确认 / 长期记忆。"""

import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.core.agents import (
    SENTIMENT_ANGRY,
    SENTIMENT_CALM,
    SENTIMENT_UPSET,
    TYPE_CHAT,
    TYPE_COMPLAINT,
    TYPE_CONSULT,
    TYPE_OPERATION,
    classify_message,
    create_ticket,
    handle_chitchat,
    handle_complaint,
    handle_operation,
    list_tickets,
)


# ───────────────────────── 意图分类 ─────────────────────────
@pytest.mark.parametrize(
    "text,expect_type",
    [
        ("你好", TYPE_CHAT),
        ("谢谢啦", TYPE_CHAT),
        ("你是谁呀", TYPE_CHAT),
        ("再见了", TYPE_CHAT),
        ("我要投诉! 你们这个退款流程太差了, 拖着不解决", TYPE_COMPLAINT),
        ("太差了, 要求赔偿!", TYPE_COMPLAINT),
        ("帮我列出文档列表", TYPE_OPERATION),
        ("删除文档 a1b2c3d4e5f60718", TYPE_OPERATION),
        ("七天内可以无理由退款吗", TYPE_CONSULT),
        ("P200 网关的工作温度范围是多少", TYPE_CONSULT),
        ("你好, 我想投诉你们的服务", TYPE_COMPLAINT),  # 长句带投诉词优先于闲聊
    ],
)
def test_classify_message(text, expect_type):
    mt, _ = classify_message(text)
    assert mt == expect_type, f"{text!r} → {mt}, expected {expect_type}"


def test_classify_sentiment_levels():
    _, s1 = classify_message("我要投诉一下这个流程")
    _, s2 = classify_message("服务太差了, 很失望")
    _, s3 = classify_message("骗子! 忍无可忍了! 要求赔偿! 12315!")
    assert s1 == SENTIMENT_CALM
    assert s2 == SENTIMENT_UPSET
    assert s3 == SENTIMENT_ANGRY


def test_classify_empty():
    mt, s = classify_message("")
    assert mt == TYPE_CONSULT and s == SENTIMENT_CALM


# ───────────────────────── 投诉 Agent: 建单 ─────────────────────────
def test_create_ticket_and_list():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "t.db")
        t = create_ticket("服务太差", SENTIMENT_UPSET, user_id="u1", db_path=db)
        assert t["ticket_id"].startswith("TK")
        assert t["status"] == "open" and not t["escalated"]
        # 高情绪 → 升级
        t2 = create_ticket("骗子! 要求赔偿!", SENTIMENT_ANGRY, user_id="u1", db_path=db)
        assert t2["escalated"] and t2["status"] == "escalated"
        all_t = list_tickets(db_path=db)
        assert {x["ticket_id"] for x in all_t} >= {t["ticket_id"], t2["ticket_id"]}


@pytest.mark.asyncio
async def test_handle_complaint_returns_ticket():
    r = await handle_complaint("你们退款太慢了, 我要投诉", SENTIMENT_UPSET, user_id="u9")
    assert r["agent"] == "complaint"
    assert r["ticket"] and r["ticket"]["ticket_id"].startswith("TK")
    assert "投诉工单" in r["answer"]


# ───────────────────────── 操作 Agent ─────────────────────────
def test_operation_list_documents():
    r = handle_operation("帮我列出文档列表")
    assert r["agent"] == "operation"
    assert "文档列表" in r["answer"] or "暂无文档" in r["answer"]


def test_operation_delete_requires_confirm():
    r = handle_operation("删除文档 a1b2c3d4e5f60718")
    assert r["pending_operation"] is not None
    assert r["pending_operation"]["danger"] is True
    assert "确认" in r["answer"]
    pid = r["pending_operation"]["pending_id"]
    # confirm 前不执行
    r2 = handle_operation("删除文档 a1b2c3d4e5f60718")  # 产生第二个 pending, 未确认的不执行
    assert r2["pending_operation"]["pending_id"] != pid
    # confirm 后执行 (文档不存在 → 安全提示而非崩溃)
    r3 = handle_operation("删除", confirm=True, pending_id=pid)
    assert r3["agent"] == "operation"
    assert r3["pending_operation"] is None
    assert "不存在" in r3["answer"] or "已删除" in r3["answer"]


def test_operation_unknown_tool():
    r = handle_operation("帮我把天上的月亮摘下来")
    assert r["agent"] == "operation"
    assert "暂不支持" in r["answer"] or "支持这些操作" in r["answer"]


# ───────────────────────── 闲聊 Agent ─────────────────────────
@pytest.mark.asyncio
async def test_handle_chitchat_uses_llm_directly():
    fake_resp = MagicMock()
    fake_resp.content = "你好呀! 有什么业务问题可以直接问我~"
    with patch("api.core.llm_client.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=fake_resp)
        mock_get.return_value = mock_client
        r = await handle_chitchat("你好")
    assert r["agent"] == "chitchat"
    assert "你好" in r["answer"]
    # 闲聊只发一次轻量 LLM 调用
    assert mock_client.chat.await_count == 1


# ───────────────────────── 长期记忆层 ─────────────────────────
def test_memory_roundtrip(monkeypatch, tmp_path):
    """写入→召回闭环 (隔离到临时 lancedb 目录, 不碰生产库)。"""
    import api.core.memory_layer as ml

    monkeypatch.setattr(ml, "_URI", str(tmp_path))
    monkeypatch.setattr(ml, "_conn", None)

    class _FakeEmbedder:
        def embed_query(self, q):
            v = [0.0] * 8
            v[ord(q[0]) % 8] = 1.0  # 首字符哈希 → 简单可复现向量
            return v

    import api.state as st

    monkeypatch.setattr(st, "get_embedder", lambda: _FakeEmbedder())

    assert ml.remember("user_a", "退款多久到账", "一到三个工作日") is True
    assert ml.remember("user_b", "P200 工作温度", "-40 到 75 度") is True
    hits = ml.recall("user_a", "退款多久能到账", top_k=3)
    assert hits and "退款" in hits[0]["question"]
    # user_b 查不到 user_a 的记忆 (user_id 隔离)
    hits_b = ml.recall("user_b", "退款多久能到账", top_k=3)
    assert all("P200" in h["question"] or "退款" not in h["question"] for h in hits_b) or hits_b == []


def test_memory_degrades_gracefully(monkeypatch, tmp_path):
    """lancedb 不可用 → 静默降级, 不抛异常。"""
    import api.core.memory_layer as ml

    def _raise(*a, **k):
        raise RuntimeError("simulated lancedb outage")

    monkeypatch.setattr("lancedb.connect", _raise)
    monkeypatch.setattr(ml, "_conn", None)
    assert ml.remember("u", "q", "a") is False
    assert ml.recall("u", "q") == []


# ───────────────────────── 路由层冒烟 (不经 LLM) ─────────────────────────
def test_ask_route_importable():
    """路由模块可导入 (接线未破坏既有导入链)。"""
    from api.routes import qa as qa_route

    assert hasattr(qa_route, "_merge_image_text")
    assert hasattr(qa_route, "_remember_async")


# ───────────────────────── force_agent (前端切换条) ─────────────────────────
@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


def test_force_agent_chat_overrides_content(api_client):
    """内容是咨询问题, 但 force_agent=chat → 强制走闲聊 Agent (mock LLM)。"""
    from unittest.mock import AsyncMock, MagicMock, patch
    fake_resp = MagicMock(); fake_resp.content = "嗨~"
    with patch("api.core.llm_client.get_llm_client") as mock_get:
        mock_client = MagicMock(); mock_client.chat = AsyncMock(return_value=fake_resp)
        mock_get.return_value = mock_client
        d = api_client.post("/api/v1/qa/ask", json={"question": "七天内可以无理由退款吗", "force_agent": "chat"}).json()
    assert d["message_type"] == "chat" and d["agent"] == "chitchat"


def test_force_agent_consult_overrides_chitchat(api_client):
    """内容是闲聊, 但 force_agent=consult → 强制走 RAG (skill 路由正常)。"""
    d = api_client.post("/api/v1/qa/ask", json={"question": "你好", "force_agent": "consult", "skill": "direct"}).json()
    assert d["message_type"] == "consult"
    assert d["skill"] == "direct"
