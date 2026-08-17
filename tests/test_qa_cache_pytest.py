"""QA 结果缓存测试 — QACache 单元 + orchestrator.ask 命中/未命中。

单元部分不依赖 LLM/embedding; 集成部分 mock 掉 _route/_skill_rag,
只验证缓存行为, 不跑真实流水线。
"""

import asyncio

import pytest

from api.config import settings
from api.core import orchestrator
from api.core.qa_cache import QACache, get_qa_cache
from api.core.shared_state import InMemoryBackend
from engines.router.intent_router import RoutingResult

KEY_ARGS = {
    "question": "RRF 算法如何融合?",
    "skill": None,
    "top_k": 10,
    "enable_self_retrieval": False,
    "temperature": 0.1,
}


@pytest.fixture(autouse=True)
def _clean_cache():
    """每测前清空全局缓存, 防止跨测试命中串扰。"""
    get_qa_cache().clear()
    yield
    get_qa_cache().clear()


# ────────────────────────── 单元: QACache ──────────────────────────


def test_make_key_stable_and_distinct():
    key1 = QACache.make_key(**KEY_ARGS)
    key2 = QACache.make_key(**KEY_ARGS)
    assert key1 == key2

    diff = dict(KEY_ARGS, question="换个问题")
    assert QACache.make_key(**diff) != key1

    diff_temp = dict(KEY_ARGS, temperature=0.9)
    assert QACache.make_key(**diff_temp) != key1


def test_set_get_roundtrip():
    c = QACache()
    k = QACache.make_key(**KEY_ARGS)
    c.set(k, {"answer": "A"}, ttl_s=60)
    assert c.get(k) == {"answer": "A"}


def test_ttl_expiry():
    c = QACache()
    k = QACache.make_key(**KEY_ARGS)
    c.set(k, {"answer": "A"}, ttl_s=-1)  # 立即过期
    assert c.get(k) is None


def test_capacity_evicts_oldest():
    # 容量驱逐逻辑在后端层: 通过 backend=InMemoryBackend(max_entries=2) 验证
    c = QACache(backend=InMemoryBackend(max_entries=2))
    keys = [f"k{i}" for i in range(3)]
    for i, k in enumerate(keys):
        c.set(k, {"i": i}, ttl_s=600)
    assert c.get(keys[0]) is None  # 最旧被逐出
    assert c.get(keys[1]) == {"i": 1}
    assert c.get(keys[2]) == {"i": 2}


def test_get_missing_returns_none():
    assert QACache().get("nonexistent") is None


# ────────────────────────── 集成: orchestrator.ask ──────────────────────────


def _canned_result() -> dict:
    return {
        "answer": "RRF 加权融合向量与 BM25 分数",
        "sources": [{"id": "c1", "content": "RRF 算法融合"}],
        "skill": "tech",
        "kb_id": "tech",
        "routing_source": "manual",
        "degradation_level": 0,
        "latency_breakdown": {},
        "retrieval_meta": {"top1_score": 0.9},
        "retrieval_rounds": 1,
    }


def test_ask_caches_then_hits(monkeypatch):
    monkeypatch.setattr(settings, "qa_cache_enabled", True)
    calls = {"n": 0}

    async def fake_route(question, skill, llm, start):
        return RoutingResult(skill="tech", kb="tech", confidence=1.0, source="manual"), 1.0

    async def fake_skill_rag(question, routing, llm, top_k, start, enable_self_retrieval, temperature, mode="hybrid"):
        calls["n"] += 1
        return _canned_result()

    monkeypatch.setattr(orchestrator, "_route", fake_route)
    monkeypatch.setattr(orchestrator, "_skill_rag", fake_skill_rag)

    r1 = asyncio.run(orchestrator.ask(**KEY_ARGS))
    r2 = asyncio.run(orchestrator.ask(**KEY_ARGS))

    assert calls["n"] == 1  # 第二次未走流水线
    assert r1.get("cache_hit") is None
    assert r2["cache_hit"] is True
    assert r2["answer"] == r1["answer"]


def test_ask_cache_disabled_runs_each_time(monkeypatch):
    monkeypatch.setattr(settings, "qa_cache_enabled", False)
    calls = {"n": 0}

    async def fake_route(question, skill, llm, start):
        return RoutingResult(skill="tech", kb="tech", confidence=1.0, source="manual"), 1.0

    async def fake_skill_rag(question, routing, llm, top_k, start, enable_self_retrieval, temperature, mode="hybrid"):
        calls["n"] += 1
        return _canned_result()

    monkeypatch.setattr(orchestrator, "_route", fake_route)
    monkeypatch.setattr(orchestrator, "_skill_rag", fake_skill_rag)

    asyncio.run(orchestrator.ask(**KEY_ARGS))
    asyncio.run(orchestrator.ask(**KEY_ARGS))

    assert calls["n"] == 2


# ────────────────────────── 流式: ask_stream 缓存重放 ──────────────────────────


def test_ask_stream_caches_then_replays(monkeypatch):
    monkeypatch.setattr(settings, "qa_cache_enabled", True)
    calls = {"n": 0}

    async def fake_route(question, skill, llm, start):
        return RoutingResult(skill="tech", kb="tech", confidence=1.0, source="manual"), 1.0

    async def fake_stream_rag(question, routing, llm, top_k, start, enable_self_retrieval, temperature, mode="hybrid"):
        calls["n"] += 1
        yield {"type": "sources", "sources": [{"id": "c1"}], "retrieval_meta": {"top1_score": 0.9}}
        yield {"type": "delta", "content": "RRF 加权融合"}
        yield {
            "type": "done",
            "answer": "RRF 加权融合",
            "token_usage": {"total_tokens": 10},
            "degradation_level": 0,
            "latency_breakdown": {"router_ms": 1.0},
            "retrieval_meta": {"top1_score": 0.9},
        }

    monkeypatch.setattr(orchestrator, "_route", fake_route)
    monkeypatch.setattr(orchestrator, "_stream_rag", fake_stream_rag)

    async def collect():
        return [ev async for ev in orchestrator.ask_stream(**KEY_ARGS)]

    events1 = asyncio.run(collect())
    events2 = asyncio.run(collect())

    assert calls["n"] == 1  # 第二次未走流水线
    types1 = [e["type"] for e in events1]
    assert types1 == ["meta", "sources", "delta", "done"]
    assert events1[-1]["answer"] == "RRF 加权融合"
    # 第二次为缓存重放: 事件序列一致, done 带 cache_hit
    types2 = [e["type"] for e in events2]
    assert types2 == types1
    assert events2[-1]["cache_hit"] is True
    assert events2[-1]["answer"] == events1[-1]["answer"]
    assert events2[2]["content"] == "RRF 加权融合"  # delta 重放
