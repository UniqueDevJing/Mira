"""服务端会话存储 — 单元测试 + 集成测试(模拟 session 透传, 验证多轮上下文由服务端维护)。"""

import asyncio
import sys
from pathlib import Path

import pytest

import api.core.retrieval as retrieval_mod
import api.core.skills as skills_mod
from api.config import settings
from api.core import session_store
from api.core.session_store import clear_session, load_session, save_session
from api.schemas.qa import ChatTurn

# 让测试可从项目根运行
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeBackend:
    """内存后端, 复刻 CacheBackend 语义(get/set/delete + TTL 惰性过期)。"""

    def __init__(self):
        self._d = {}
        self.sets = 0

    def get(self, key):
        return self._d.get(key)

    def set(self, key, value, ttl_s):
        self._d[key] = value
        self.sets += 1

    def delete(self, key):
        self._d.pop(key, None)


def test_save_then_load_roundtrip():
    be = _FakeBackend()
    save_session("s1", [ChatTurn(role="user", content="你好"), ChatTurn(role="assistant", content="你好!")], backend=be)
    hist = load_session("s1", backend=be)
    assert len(hist) == 2
    assert hist[0].role == "user" and hist[0].content == "你好"
    assert hist[1].role == "assistant" and hist[1].content == "你好!"


def test_load_missing_returns_empty():
    assert load_session("nope", backend=_FakeBackend()) == []


def test_save_caps_to_20_turns():
    be = _FakeBackend()
    turns = [ChatTurn(role="user" if i % 2 == 0 else "assistant", content=f"t{i}") for i in range(50)]
    save_session("s2", turns, backend=be)
    hist = load_session("s2", backend=be)
    assert len(hist) == 20
    # 保留最近 20 轮(末尾)
    assert hist[-1].content == "t49"


def test_save_accepts_dict_turns():
    be = _FakeBackend()
    save_session("s3", [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], backend=be)
    hist = load_session("s3", backend=be)
    assert [t.content for t in hist] == ["x", "y"]


def test_clear_session():
    be = _FakeBackend()
    save_session("s4", [ChatTurn(role="user", content="a")], backend=be)
    clear_session("s4", backend=be)
    assert load_session("s4", backend=be) == []


def test_empty_session_id_noop():
    be = _FakeBackend()
    save_session("", [ChatTurn(role="user", content="a")], backend=be)
    assert be._d == {}


# ───────────────────────── 集成: session 透传多轮 ─────────────────────────
# 复用多轮测试的同款 monkeypatch 思路: 用 FakeLLM 捕获最后一次发送给 LLM 的 messages


class _FakeLLM:
    def __init__(self):
        self.last_messages = None

    async def chat(self, messages, temperature=0.1, max_tokens=512, json_mode=False, timeout=None):
        self.last_messages = messages
        # 把历史里的 user 内容拼进答案, 便于断言上下文进入
        hist_users = [m["content"] for m in messages if m["role"] == "user" and "退款" in m["content"]]
        answer = "已结合上下文: " + (";".join(hist_users)) if hist_users else "无历史"
        return {"answer": answer, "token_usage": {"total_tokens": 10}, "degradation_level": 0}


@pytest.fixture
def patched_session(monkeypatch):
    """注入 FakeLLM + FakeBackend, 暴露 orchestrator 的 ask。"""
    be = _FakeBackend()
    llm = _FakeLLM()

    monkeypatch.setattr(skills_mod, "get_llm_client", lambda: llm)
    monkeypatch.setattr(session_store, "get_cache_backend", lambda: be)
    monkeypatch.setattr(skills_mod, "get_qa_cache", lambda: None)  # 关缓存, 强制走生成

    # 检索/路由/重排 stub: 让 _skill_rag 走 direct 之外的分支返回稳定上下文
    async def _fake_route(q, skill, llm_client, start, candidate_kbs=None):
        from engines.router.intent_router import RoutingResult

        r = RoutingResult(skill="tech", kb="default", confidence=0.9, source="rule")
        return r, [r], 1.0

    monkeypatch.setattr(skills_mod, "_route", _fake_route)

    async def _fake_retrieve(
        question, routing, top_k, start, enable_self_retrieval=False, mode="hybrid", candidate_kbs=None
    ):
        return {
            "docs": [{"content": "退款政策: 1-3 工作日", "id": "d1", "chunk_id": "c1", "doc_id": "doc1", "score": 0.9}],
            "context": "退款政策: 1-3 工作日",
            "degradation": 0,
            "retrieval_ms": 1.0,
            "rerank_ms": 0.0,
            "top1_score": 0.9,
            "cross_kb_kbs": [],
            "retrieval_rounds": 1,
            "rewritten_queries": [],
            "graph_context": None,
        }

    monkeypatch.setattr(skills_mod, "_retrieve_context", _fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_retrieve_context", _fake_retrieve)

    async def _fake_rerank(*a, **k):
        return None

    monkeypatch.setattr(retrieval_mod, "_rerank_safe", _fake_rerank)
    return llm


def test_session_drives_multiturn_via_server_state(patched_session):
    """两轮对话只带 session_id(不带 body history), 第二轮应利用服务端存储的第一轮上下文。"""
    import api.core.orchestrator as oc

    sid = "integ-session-1"
    a1 = asyncio.run(oc.ask("退款多久到账?", session_id=sid))
    assert "退款" in a1["answer"]
    # 第二轮: 不带 body history, 仅 session_id
    asyncio.run(oc.ask("那银行卡呢?", session_id=sid))
    # LLM 收到的 messages 应包含第一轮的 user 问题(证明服务端 history 生效)
    users = [m["content"] for m in patched_session.last_messages if m["role"] == "user"]
    assert any("退款多久到账" in u for u in users), f"服务端历史未进入 LLM: {users}"
    # 且包含当前问题
    assert any("银行卡" in u for u in users)


def test_session_isolated_across_ids(patched_session):
    """不同 session_id 互不串历史。"""
    import api.core.orchestrator as oc

    asyncio.run(oc.ask("退款多久到账?", session_id="A"))
    a2_b = asyncio.run(oc.ask("那银行卡呢?", session_id="B"))  # 另一个 session, 无第一轮上下文
    # B 的 LLM messages 不应含 A 的第一轮问题
    users = [m["content"] for m in patched_session.last_messages if m["role"] == "user"]
    assert not any("退款多久到账" in u for u in users)
    assert "无历史" in a2_b["answer"] or any("银行卡" in u for u in users)


def test_cache_hit_still_persists_session(monkeypatch):
    """缓存命中路径(非流式)也应把本轮写入 session, 否则重复问题会丢失多轮链路。"""
    import api.core.orchestrator as oc

    class _FakeCache:
        """第一轮 miss, 之后恒 hit — 精确触发 ask() 的缓存命中分支。"""

        def __init__(self):
            self._calls = 0

        def make_key(self, *a, **k):
            return "fixed-key"

        def make_scope(self, *a, **k):
            return "fixed-scope"

        def get(self, key, **kwargs):
            self._calls += 1
            return None if self._calls == 1 else {"answer": "cached-answer", "sources": [], "latency_breakdown": {}}

        def set(self, key, value, ttl_s, **kwargs):
            pass

    be = _FakeBackend()
    cache = _FakeCache()
    llm = _FakeLLM()
    monkeypatch.setattr(skills_mod, "get_llm_client", lambda: llm)
    monkeypatch.setattr(session_store, "get_cache_backend", lambda: be)
    monkeypatch.setattr(skills_mod, "get_qa_cache", lambda: cache)
    monkeypatch.setattr(settings, "qa_cache_enabled", True)  # 确保走缓存路径(可能被子测试改过)

    async def _fake_route(q, skill, llm_client, start, candidate_kbs=None):
        from engines.router.intent_router import RoutingResult

        r = RoutingResult(skill="tech", kb="default", confidence=0.9, source="rule")
        return r, [r], 1.0

    monkeypatch.setattr(skills_mod, "_route", _fake_route)

    sid = "cache-hit-session"
    # R1: 缓存未命中, 落盘 [u1, a1]
    asyncio.run(oc.ask("重复问题X?", session_id=sid))
    assert len(load_session(sid, backend=be)) == 2

    # R2: 缓存命中, 但仍应把本轮追加进 session -> [u1,a1,u2,a2] = 4
    r2 = asyncio.run(oc.ask("重复问题X?", session_id=sid))
    assert r2.get("cache_hit") is True
    assert len(load_session(sid, backend=be)) == 4
    assert load_session(sid, backend=be)[-1].content == "cached-answer"
