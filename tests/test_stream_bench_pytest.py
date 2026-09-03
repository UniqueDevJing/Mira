"""P1-4 流式 sources 先行契约测试 — 单请求验证事件顺序, 不依赖外部 LLM/鉴权。"""
import json
import os
import sys
import types

import pytest
from fastapi.testclient import TestClient

from api.core import orchestrator as _orch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class FakeLLM:
    base_url = "fake://local"
    model = "fake"
    api_key = "fake"

    async def chat(self, messages, **kwargs):
        return types.SimpleNamespace(content="x", latency_ms=1.0)

    async def stream_chat(self, messages, **kwargs):
        yield {"type": "delta", "content": "x"}


@pytest.fixture
def client():
    _orch.get_llm_client = lambda *a, **k: FakeLLM()  # type: ignore[assignment]
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_stream_sources_contract(client, monkeypatch):
    """P0-2 契约(rerank 开启时): sources 先行(final=False) → 最终 sources(final=True) → done。"""
    # 关键: 必须 patch skills 模块的 get_llm_client (skills.py 的名字绑定自 api.core.llm_client,
    # patch _orch 的 re-export 无效)。否则路由用 .env 真实 LLM, "测试问题"被分类为
    # direct(闲聊不检索) → 0 个 sources 事件, 测试随真实 LLM 可达性/回答而抖动。
    import api.core.skills as _skills
    from api.config import settings

    monkeypatch.setattr(_skills, "get_llm_client", lambda: FakeLLM())
    monkeypatch.setattr(settings, "rerank_enabled", True)  # provisional 机制依赖 rerank, 显式开启
    with client.stream("POST", "/api/v1/qa/ask/stream", json={"question": "测试问题"}) as r:
        events = []
        for line in r.iter_lines():
            if line and line.startswith("data: "):
                events.append(json.loads(line[6:]))

    types_seen = [e.get("type") for e in events]
    sources_events = [e for e in events if e.get("type") == "sources"]

    assert len(sources_events) >= 2, "应至少发两次 sources 事件(先行 + 最终)"
    assert sources_events[0].get("final") is False, "首个 sources 应为重排前(final=False)"
    assert any(e.get("final") is True for e in sources_events), "应有 final=True 的最终 sources"
    assert "done" in types_seen, "应以 done 事件结束"
    # 顺序不变量: 首个 sources 必须在 done 之前
    assert types_seen.index("sources") < types_seen.index("done")


def test_stream_no_auth_required_on_loopback(client):
    """127.0.0.1 绑定 + 鉴权默认关, 免 key 应 200。"""
    with client.stream("POST", "/api/v1/qa/ask/stream", json={"question": "免鉴权测试"}) as r:
        assert r.status_code == 200
