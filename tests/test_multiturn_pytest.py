"""多轮对话后端测试: history 字段透传至 LLM messages。

策略:
- 单测 _history_to_messages 纯函数 (截断/角色过滤/模型与 dict 兼容)
- 集成测: monkeypatch 检索与 LLM, 断言历史轮次确实进入 chat/stream_chat 的 messages
"""

import asyncio

import pytest

import api.core.retrieval as retrieval_mod
import api.core.skills as skills_mod
from api.core import orchestrator
from api.schemas.qa import ChatTurn
from engines.router.intent_router import RoutingResult

# ────────────────────────── 纯函数单测 ──────────────────────────


def test_history_to_messages_filters_roles_and_truncates():
    # 非法角色/空内容应被跳过
    mixed = [
        {"role": "system", "content": "忽略"},
        {"role": "user", "content": "退款多久到账"},
        {"role": "assistant", "content": "一到三个工作日"},
        {"role": "user", "content": ""},  # 空内容跳过
    ]
    out = orchestrator._history_to_messages(mixed)
    assert out == [
        {"role": "user", "content": "退款多久到账"},
        {"role": "assistant", "content": "一到三个工作日"},
    ]


def test_history_to_messages_truncates_to_last_20():
    many = [{"role": "user", "content": f"q{i}"} for i in range(25)]
    out = orchestrator._history_to_messages(many)
    assert len(out) == 20
    assert out[0]["content"] == "q5"  # 仅保留最近 20 条


def test_history_to_messages_accepts_pydantic_models():
    turns = [ChatTurn(role="user", content="a"), ChatTurn(role="assistant", content="b")]
    out = orchestrator._history_to_messages(turns)
    assert out == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]


def test_history_to_messages_empty():
    assert orchestrator._history_to_messages(None) == []
    assert orchestrator._history_to_messages([]) == []


# ────────────────────────── 集成测 (mock 检索 + LLM) ──────────────────────────


class _FakeLLM:
    def __init__(self):
        self.last_messages = None

    async def chat(self, messages, temperature=0.1, max_tokens=2000):
        self.last_messages = messages

        class _R:
            content = "根据文档, 退款一般一到三个工作日到账。"
            prompt_tokens = 50
            completion_tokens = 20
            total_tokens = 70
            latency_ms = 100.0

        return _R()

    async def stream_chat(self, messages, temperature=0.3, max_tokens=2000):
        self.last_messages = messages
        yield {"type": "delta", "content": "根据文档, 退款一般一到三个工作日到账。"}
        yield {"type": "usage", "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}}


@pytest.fixture
def patched(monkeypatch):
    fake = _FakeLLM()

    async def _fake_retrieve(
        question, routing, top_k, start, enable_self_retrieval=False, mode="hybrid",
        candidate_kbs=None, defer_rerank=False,
    ):
        docs = [{"id": "1", "chunk_id": "c1", "doc_id": "d1", "content": "退款一般一到三个工作日到账", "score": 0.9}]
        return {
            "docs": docs,
            "context": "退款一般一到三个工作日到账",
            "degradation": 0,
            "retrieval_ms": 1.0,
            "rerank_ms": 1.0,
            "top1_score": 0.9,
            "cross_kb_kbs": [],
            "retrieval_rounds": 1,
            "rewritten_queries": [],
            "graph_context": None,
        }

    async def _fake_route(question, skill, llm, start, candidate_kbs=None):
        r = RoutingResult(skill="service", kb="service", confidence=1.0, source="rule")
        return r, [r], 0.1

    monkeypatch.setattr(skills_mod, "get_qa_cache", lambda: None)
    monkeypatch.setattr(skills_mod, "get_llm_client", lambda: fake)
    monkeypatch.setattr(skills_mod, "_retrieve_context", _fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_retrieve_context", _fake_retrieve)
    monkeypatch.setattr(skills_mod, "_route", _fake_route)
    return fake


def test_ask_threads_history_into_llm(patched):
    history = [
        ChatTurn(role="user", content="退款多久到账"),
        ChatTurn(role="assistant", content="一般一到三个工作日到账"),
    ]
    asyncio.run(orchestrator.ask("那银行卡呢", history=history, mode="vector"))
    msgs = patched.last_messages
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert {"role": "user", "content": "退款多久到账"} in msgs
    assert {"role": "assistant", "content": "一般一到三个工作日到账"} in msgs
    # 最近一轮必须是当前问题 (含参考文档)
    assert msgs[-1]["role"] == "user"
    assert "那银行卡呢" in msgs[-1]["content"]


def test_ask_without_history_has_no_prior_turns(patched):
    asyncio.run(orchestrator.ask("退款多久到账", mode="vector"))
    msgs = patched.last_messages
    # 仅 system + 当前 user, 无历史轮
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_ask_stream_threads_history_into_llm(patched):
    history = [ChatTurn(role="user", content="退款多久到账"), ChatTurn(role="assistant", content="一到三个工作日")]

    async def _collect():
        return [ev async for ev in orchestrator.ask_stream("那银行卡呢", history=history, mode="vector")]

    events = asyncio.run(_collect())
    assert any(ev.get("type") == "done" for ev in events)
    msgs = patched.last_messages
    assert {"role": "user", "content": "退款多久到账"} in msgs
    assert {"role": "assistant", "content": "一到三个工作日"} in msgs
