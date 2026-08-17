"""流式问答测试 — /ask/stream SSE 事件序列。

用 FakeLLM 替代真实 LLM（stream_chat async generator），
规则路由触发（"退货流程"→service），避免 LLM 分类依赖。
不依赖真实 LLM/embedding 成功。
"""

import json

from fastapi.testclient import TestClient

from api.core import orchestrator
from api.main import app

client = TestClient(app)


class FakeLLM:
    """模拟流式 LLM: chat 返回单块, stream_chat 逐块产出。"""

    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or ["退货", "流程", "：", "提交", "申请。"]
        self.error = error

    async def chat(self, messages, **kwargs):
        from api.core.llm_client import LLMResponse

        if self.error:
            raise self.error
        return LLMResponse(content="".join(self.chunks))

    async def stream_chat(self, messages, temperature=0.3, max_tokens=2000):
        if self.error:
            raise self.error
        for c in self.chunks:
            yield {"type": "delta", "content": c, "reasoning": ""}
        yield {"type": "usage", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def _patch_llm(monkeypatch, fake):
    monkeypatch.setattr(orchestrator, "get_llm_client", lambda: fake)


def _collect_events(resp):
    """解析 SSE 文本 → 事件 dict 列表。"""
    events = []
    for block in resp.text.split("\n\n"):
        if not block.strip():
            continue
        data_line = [l for l in block.split("\n") if l.startswith("data:")]
        if data_line:
            events.append(json.loads(data_line[0][5:].strip()))
    return events


def test_stream_meta_and_done(monkeypatch):
    _patch_llm(monkeypatch, FakeLLM())
    r = client.post("/api/v1/qa/ask/stream", json={"question": "退货流程是什么", "skill": "service"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _collect_events(r)

    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert types[-1] == "done"
    # meta 携带路由信息
    assert events[0]["skill"] == "service"
    assert events[0]["routing_source"] == "manual"
    # done 携带最终 answer + 降级等级
    assert "answer" in events[-1]
    assert "degradation_level" in events[-1]


def test_stream_delta_accumulates_answer(monkeypatch):
    _patch_llm(monkeypatch, FakeLLM())
    r = client.post("/api/v1/qa/ask/stream", json={"question": "退货流程是什么", "skill": "service"})
    events = _collect_events(r)
    deltas = [e["content"] for e in events if e["type"] == "delta"]
    assert deltas  # 有内容块
    assert "".join(deltas) == "退货流程：提交申请。"


def test_stream_llm_error_falls_back(monkeypatch):
    """LLM 失败 → 降级 L3, done 含降级答案。"""
    _patch_llm(monkeypatch, FakeLLM(error=RuntimeError("api down")))
    r = client.post("/api/v1/qa/ask/stream", json={"question": "退货流程是什么", "skill": "service"})
    assert r.status_code == 200
    events = _collect_events(r)
    done = events[-1]
    assert done["type"] == "done"
    assert done["degradation_level"] == 3
    # LLM 失败 → 降级为检索摘要（"LLM 暂时不可用"前缀）或空库"未找到"
    assert ("LLM 暂时不可用" in done["answer"]) or ("未在知识库中找到" in done["answer"])


def test_stream_rule_routing(monkeypatch):
    """未指定 skill 时走规则路由 → service。"""
    _patch_llm(monkeypatch, FakeLLM())
    r = client.post("/api/v1/qa/ask/stream", json={"question": "退货流程是什么"})
    events = _collect_events(r)
    assert events[0]["type"] == "meta"
    assert events[0]["skill"] == "service"


def test_json_ask_still_works(monkeypatch):
    """非流式 /ask 端点保持可用（向后兼容）。"""
    _patch_llm(monkeypatch, FakeLLM())
    r = client.post("/api/v1/qa/ask", json={"question": "退货流程是什么", "skill": "service"})
    assert r.status_code == 200
    d = r.json()
    assert d["answer"] != ""
    assert "degradation_level" in d


def test_self_retrieval_flag_passthrough(monkeypatch):
    """enable_self_retrieval=true 端到端可用，返回 retrieval_rounds 字段。"""
    _patch_llm(monkeypatch, FakeLLM())
    r = client.post(
        "/api/v1/qa/ask", json={"question": "退货流程是什么", "skill": "service", "enable_self_retrieval": True}
    )
    assert r.status_code == 200
    d = r.json()
    assert "retrieval_rounds" in d
    assert d["retrieval_rounds"] >= 1
    assert isinstance(d["rewritten_queries"], list)
