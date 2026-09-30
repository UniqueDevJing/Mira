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


# ───────────────── 回归: 会话丢失后上下文自愈 (线上"追问变成重新回答"的根因) ─────────────────


def test_server_session_lost_recovers_from_request_history(patched_session):
    """服务端会话为空(进程重启 / 超 TTL)时, 请求里的 history 必须被采用, 不能整个丢弃。

    旧实现 `load_session(sid) if sid else history` 只要带了 session_id 就无条件覆盖
    body.history —— 会话一空, 模型就丢了全部上下文, 表现为「追问时重新回答、不接上一轮」。
    """
    import api.core.orchestrator as oc

    sid = "recovery-session"
    asyncio.run(oc.ask("退款多久到账?", session_id=sid))
    assert len(load_session(sid)) > 0

    # 模拟服务重启 / 超过 TTL: 服务端会话被清空, 前端仍持有这段对话
    clear_session(sid)
    frontend_history = [
        ChatTurn(role="user", content="退款多久到账?"),
        ChatTurn(role="assistant", content="1-3 个工作日到账"),
    ]
    r = asyncio.run(oc.ask("那银行卡呢?", session_id=sid, history=frontend_history))

    users = [m["content"] for m in patched_session.last_messages if m["role"] == "user"]
    assert any("退款多久到账" in u for u in users), f"上下文被丢弃, 未回退到请求历史: {users}"
    # 恢复后回填服务端会话, 后续轮次重新以服务端为准 (否则每轮都要靠前端兜底)
    assert len(load_session(sid)) >= 3
    assert r["memory_meta"]["recovered"] is True


def test_missing_history_does_not_fabricate_context(patched_session):
    """反向断言: 会话空且请求也没带历史时, 不得凭空造出上下文。"""
    import api.core.orchestrator as oc

    sid = "empty-context-session"
    marker = "上一轮问的ZZTOP"
    r = asyncio.run(oc.ask("完全无关的新问题", session_id=sid))
    sent = " ".join(m["content"] for m in patched_session.last_messages)
    assert marker not in sent, f"不应存在无来源的历史: {sent[:200]}"
    assert r["memory_meta"]["turns_used"] == 0
    assert r["memory_meta"]["recovered"] is False


def test_truncation_is_reported_not_silent(patched_session):
    """超出上下文窗口时应如实上报 dropped, 而不是静默截断。"""
    import api.core.orchestrator as oc

    sid = "truncate-session"
    long_history = [
        ChatTurn(role="user" if i % 2 == 0 else "assistant", content=f"历史消息{i}")
        for i in range(30)
    ]
    r = asyncio.run(oc.ask("接着上面继续", session_id=sid, history=long_history))
    meta = r["memory_meta"]
    assert meta["turns_used"] == 20, f"注入窗口应为 20 条, 实际 {meta['turns_used']}"
    assert meta["dropped"] == 10, f"应上报被截掉的 10 条, 实际 {meta['dropped']}"
    # 累计口径: 再问一轮, dropped 继续累加(6 条新消息把最早的挤出窗口)
    r2 = asyncio.run(oc.ask("再问一句", session_id=sid))
    assert r2["memory_meta"]["dropped"] >= 10


def test_current_question_in_history_is_not_duplicated(patched_session):
    """客户端把「本轮提问」也放进 history 时, 会话里不得出现两条同样的提问。

    前端曾在 appendUserMessage 之后才构造请求体, 于是 history 的末条就是本轮提问, 而服务端
    还会再显式追加一次 → 同一提问在会话里存了两遍, 重新打开会话时重复渲染给用户看。
    这里把「客户端可能这样发」固定成契约的一部分: 去重由服务端负责。
    """
    import api.core.orchestrator as oc

    sid = "dup-session"
    q = "公司2025年的营业收入是多少？"
    prior = [
        ChatTurn(role="user", content="更早的问题"),
        ChatTurn(role="assistant", content="更早的回答"),
    ]
    # 模拟前端行为: history 里已含本轮提问
    asyncio.run(oc.ask(q, session_id=sid, history=prior + [ChatTurn(role="user", content=q)],
                       client_history=prior + [ChatTurn(role="user", content=q)]))

    contents = [t.content for t in load_session(sid)]
    assert contents.count(q) == 1, f"本轮提问重复入库: {contents}"
    assert contents[0] == "更早的问题", f"转录顺序异常: {contents}"


def test_memory_injection_never_persisted(patched_session):
    """长期记忆注入只进本轮提示词, 绝不能写回会话。

    这不是洁癖: 注入条目形如 "[历史提问] …", 一旦落进会话, 下次打开会话时会被当成**真实
    对话**渲染出来 —— 实测现象是同一条用户提问在界面上重复出现两遍, 且随轮次继续复制扩散。
    """
    import api.core.orchestrator as oc

    sid = "inject-session"
    injected = [
        ChatTurn(role="user", content="[历史提问] 上次问过的老问题"),
        ChatTurn(role="assistant", content="[历史回答] 上次给过的旧答案"),
    ]
    # client_history 显式为空: 客户端没有转录, 不得据此回填会话
    asyncio.run(oc.ask("这轮的新问题", session_id=sid, history=injected, client_history=[]))

    stored = [t.content for t in load_session(sid)]
    assert not any(c.startswith(("[历史提问]", "[历史回答]")) for c in stored), f"注入条目被写进了会话: {stored}"
    assert stored and stored[0] == "这轮的新问题", f"会话首条应为真实提问, 实际 {stored[:2]}"

    # 反向: 注入内容仍必须进入本轮提示词(否则长期记忆功能等于被关掉)
    sent = " ".join(m["content"] for m in patched_session.last_messages)
    assert "[历史提问]" in sent or "上次问过的老问题" in sent, "注入内容未进入提示词"


def test_session_meta_counts_dropped_across_saves():
    """存储层: dropped 跨多次写入累计, 且不与旧格式冲突。

    契约: 调用方总是传「当前会话内容 + 本轮新消息」, 由存储层负责裁剪与计数;
    若调用方先把历史裁好再传, 截断信息就丢了 (线上曾因此无法告知用户"更早的对话已超窗口")。
    """
    be = _FakeBackend()
    turns = [ChatTurn(role="user", content=f"t{i}") for i in range(25)]
    assert save_session("m1", turns, backend=be) == 5
    assert session_store.session_meta("m1", backend=be) == {"turns": 20, "dropped": 5}
    # 模拟真实调用方: 已裁剪的会话内容 + 本轮一问一答 → 只应新增 2 条截断
    grown = load_session("m1", backend=be) + [
        ChatTurn(role="user", content="u"),
        ChatTurn(role="assistant", content="a"),
    ]
    assert save_session("m1", grown, backend=be) == 7
    assert session_store.session_meta("m1", backend=be)["dropped"] == 7
    # 旧格式(裸列表)仍可读, 且被视作未截断
    be.set("rag:session:legacy", '[{"role":"user","content":"老数据"}]', 60)
    assert [t.content for t in load_session("legacy", backend=be)] == ["老数据"]
    assert session_store.session_meta("legacy", backend=be)["dropped"] == 0
