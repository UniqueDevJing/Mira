"""P0-2 流式契约测试: 来源先于 rerank 发出, 且答案仍用重排后的上下文。

被测行为 (api/core/orchestrator.py::_stream_rag + _apply_deferred_rerank):
  1. stream_sources_before_rerank=True 时, 先发一个 final=False 的临时 sources (检索序),
     rerank 完成后再发 final=True 的最终 sources (重排序) 覆盖它。
  2. 两个事件的时间间隔 ≥ rerank 耗时 —— 证明用户不必等 rerank 就能看到来源。
  3. **最关键**: 送入 LLM 的 context 必须来自重排后的 docs, 不能是临时来源,
     否则引用会错位 (这是"提前发来源"最容易引入的正确性风险)。
  4. 开关关闭时行为不变: 只发一个 sources 事件。

策略: monkeypatch 检索与 rerank (rerank 用可控 sleep 模拟 ~300ms 开销),
断言事件序列 / 时序 / LLM 收到的上下文内容。
"""

import asyncio
import time

import pytest

import api.core.retrieval as retrieval_mod
import api.core.skills as skills_mod
from api.config import settings
from api.core import orchestrator
from engines.router.intent_router import RoutingResult

RERANK_DELAY = 0.30  # 模拟 rerank 耗时 (秒)

DOC_PRE = {"id": "1", "chunk_id": "c1", "doc_id": "d1", "content": "重排前排第一的片段", "score": 0.9}
DOC_PRE2 = {"id": "2", "chunk_id": "c2", "doc_id": "d2", "content": "重排前排第二的片段", "score": 0.8}
# 重排后顺序颠倒: 便于断言"最终来源 ≠ 临时来源"
RERANKED = [DOC_PRE2, DOC_PRE]


class _FakeLLM:
    def __init__(self):
        self.last_messages = None

    async def chat(self, messages, temperature=0.1, max_tokens=2000):
        self.last_messages = messages
        return _Resp()

    async def stream_chat(self, messages, temperature=0.3, max_tokens=2000):
        self.last_messages = messages
        yield {"type": "delta", "content": "答案"}
        yield {"type": "usage", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class _Resp:
    content = "答案"
    prompt_tokens = 1
    completion_tokens = 1
    total_tokens = 2
    latency_ms = 10.0


@pytest.fixture
def patched(monkeypatch):
    """打掉检索/路由/LLM, 让 _stream_rag 的流式编排可被确定性验证。"""
    fake = _FakeLLM()

    base = {
        "degradation": 0,
        "retrieval_ms": 1.0,
        "rerank_ms": 1.0,
        "top1_score": 0.9,  # 高于 answer_confidence_floor, 确保不触发拒答分支
        "cross_kb_kbs": [],
        "retrieval_rounds": 1,
        "rewritten_queries": [],
        "graph_context": None,
    }

    async def _fake_retrieve(question, routing, top_k, start, enable_self_retrieval=False,
                             mode="hybrid", candidate_kbs=None, defer_rerank=False):
        docs = [DOC_PRE, DOC_PRE2]
        if defer_rerank:
            # 模拟 _retrieve_context(defer_rerank=True): 跳过 rerank, 把融合结果挂出去
            return {**base, "docs": docs, "context": "临时上下文", "fused": docs, "rerank_deferred": True}
        return {**base, "docs": RERANKED, "context": "重排后上下文"}

    async def _fake_rerank(kb, question, fused, top_k, start, degradation):
        await asyncio.sleep(RERANK_DELAY)  # 模拟 Cross-Encoder 推理耗时
        return RERANKED[:top_k], RERANK_DELAY * 1000, degradation

    async def _fake_route(question, skill, llm, start, candidate_kbs=None):
        r = RoutingResult(skill="service", kb="service", confidence=1.0, source="rule")
        return r, [r], 0.1

    monkeypatch.setattr(skills_mod, "get_qa_cache", lambda: None)
    monkeypatch.setattr(skills_mod, "get_llm_client", lambda: fake)
    monkeypatch.setattr(skills_mod, "_retrieve_context", _fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_retrieve_context", _fake_retrieve)
    monkeypatch.setattr(retrieval_mod, "_rerank_safe", _fake_rerank)
    monkeypatch.setattr(skills_mod, "_route", _fake_route)
    return fake


def _collect(question="测试问题", **kw):
    async def _run():
        out = []
        async for ev in orchestrator.ask_stream(question, mode="vector", **kw):
            out.append((time.time(), ev))
        return out
    return asyncio.run(_run())


def test_stream_emits_provisional_then_final_sources(patched, monkeypatch):
    """开启开关: 应发出 临时(final=False) → 最终(final=True) 两个 sources 事件。"""
    monkeypatch.setattr(settings, "stream_sources_before_rerank", True)
    monkeypatch.setattr(settings, "rerank_enabled", True)  # provisional 机制依赖 rerank, 显式开启
    events = _collect()
    sources = [ev for _, ev in events if ev.get("type") == "sources"]

    assert len(sources) == 2, f"应有 2 个 sources 事件, 实际 {len(sources)}"
    assert sources[0]["final"] is False, "第一个必须是重排前的临时来源"
    assert sources[1]["final"] is True, "第二个必须是重排后的最终来源"
    # 最终来源应反映重排结果 (顺序被颠倒过)
    assert sources[1]["sources"][0]["content"] == RERANKED[0]["content"]


def test_provisional_sources_precede_rerank_by_its_duration(patched, monkeypatch):
    """临时来源必须比重排完成早 ~RERANK_DELAY 发出 —— 这是"消除白屏"的量化证明。"""
    monkeypatch.setattr(settings, "stream_sources_before_rerank", True)
    monkeypatch.setattr(settings, "rerank_enabled", True)  # provisional 机制依赖 rerank, 显式开启
    timed = _collect()
    ts = [t for t, ev in timed if ev.get("type") == "sources"]

    assert len(ts) == 2
    gap = ts[1] - ts[0]
    assert gap >= RERANK_DELAY * 0.8, (
        f"临时来源与最终来源间隔仅 {gap:.3f}s, 未覆盖 rerank 耗时 {RERANK_DELAY}s, "
        "说明来源并没有真正提前发出"
    )


def test_llm_context_uses_reranked_docs_not_provisional(patched, monkeypatch):
    """最关键的正确性保证: 送入 LLM 的上下文来自重排后的 docs, 引用不会错位。"""
    monkeypatch.setattr(settings, "stream_sources_before_rerank", True)
    monkeypatch.setattr(settings, "rerank_enabled", True)  # provisional 机制依赖 rerank, 显式开启
    _collect()

    messages = patched.last_messages or []
    ctx = "\n".join(m.get("content", "") for m in messages)
    assert RERANKED[0]["content"] in ctx, "LLM 上下文应包含重排后排首位的片段"
    assert "临时上下文" not in ctx, "LLM 上下文不应使用重排前的临时 context (会导致引用错位)"


def test_disabled_emits_single_sources_event(patched, monkeypatch):
    """关闭开关: 行为与改造前完全一致 (只发一个 sources)。"""
    monkeypatch.setattr(settings, "stream_sources_before_rerank", False)
    events = _collect()
    sources = [ev for _, ev in events if ev.get("type") == "sources"]

    assert len(sources) == 1, f"关闭开关时应只有 1 个 sources 事件, 实际 {len(sources)}"
    assert sources[0]["final"] is True
