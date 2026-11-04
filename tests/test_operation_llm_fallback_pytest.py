"""M-A 操作路由 LLM 兜底测试：规则优先、白名单强制、失败静默降级。"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.common.llm_client import LLMResponse

import api.core.agents as agents
from api.config import settings
from api.core.agents import handle_operation


def _llm_response(tool_name, arguments="{}"):
    return LLMResponse(content="", tool_calls=[
        {"id": "call_1", "name": tool_name, "arguments": arguments},
    ])


def _fake_router(response=None, exc=None):
    """返回一个假的 _get_router_client 工厂（绕过真实 LLM）。"""

    class _FakeClient:
        def chat(self, **kwargs):
            if exc:
                raise exc
            return response

    return lambda: _FakeClient()


def test_rule_hit_does_not_call_llm(monkeypatch):
    """规则命中时零延迟直取，绝不多付一次 LLM 调用。"""

    def _boom():
        raise AssertionError("规则命中不应触发 LLM 兜底")

    monkeypatch.setattr(agents, "_get_router_client", _boom)
    r = handle_operation("列出文档列表")
    assert r["routed_by"] == "rule"
    assert r["agent"] == "operation"


def test_llm_fallback_routes_on_rule_miss(monkeypatch):
    """规则未命中 → LLM function calling 路由到白名单工具。"""
    monkeypatch.setattr(settings, "operation_llm_fallback", True)
    monkeypatch.setattr(agents, "_get_router_client",
                        _fake_router(_llm_response("list_documents")))
    r = handle_operation("把知识库里的东西给我看看")
    assert r["routed_by"] == "llm"
    assert r["agent"] == "operation"


def test_llm_fallback_rejects_tool_outside_whitelist(monkeypatch):
    """模型输出白名单外的工具名 → 丢弃，走原提示话术（不执行）。"""
    monkeypatch.setattr(settings, "operation_llm_fallback", True)
    monkeypatch.setattr(agents, "_get_router_client",
                        _fake_router(_llm_response("format_disk")))
    r = handle_operation("帮我把天上的月亮摘下来")
    assert "支持这些操作" in r["answer"]
    assert r.get("routed_by") is None


def test_llm_fabricated_doc_id_is_dropped(monkeypatch):
    """模型编造的 doc_id 不过格式校验 → 不采信，按缺参提示。"""
    monkeypatch.setattr(settings, "operation_llm_fallback", True)
    monkeypatch.setattr(agents, "_get_router_client",
                        _fake_router(_llm_response("doc_status", '{"doc_id": "随便编的"}')))
    r = handle_operation("帮我看看那份报告处理得怎么样了")
    assert r["routed_by"] == "llm"
    assert "请提供文档 ID" in r["answer"]


def test_fallback_disabled_keeps_legacy_behavior(monkeypatch):
    """开关关闭（RAG_OPERATION_LLM_FALLBACK=false）：行为与改造前逐字一致。"""
    monkeypatch.setattr(settings, "operation_llm_fallback", False)

    def _boom():
        raise AssertionError("开关关闭不应触发 LLM 兜底")

    monkeypatch.setattr(agents, "_get_router_client", _boom)
    r = handle_operation("帮我把天上的月亮摘下来")
    assert "支持这些操作" in r["answer"]


def test_llm_failure_degrades_silently(monkeypatch):
    """LLM 熔断/超时 → 兜底路由静默失败，走原提示话术，不报错给用户。"""
    from engines.common.llm_client import CircuitBreakerOpenError

    monkeypatch.setattr(settings, "operation_llm_fallback", True)
    monkeypatch.setattr(agents, "_get_router_client",
                        _fake_router(exc=CircuitBreakerOpenError("熔断中")))
    r = handle_operation("把知识库里的东西给我看看")
    assert "支持这些操作" in r["answer"]


def test_llm_route_danger_still_requires_confirm(monkeypatch):
    """关键安全边界：LLM 路由到高危操作时，二次确认闸门与规则路径完全一致。"""
    monkeypatch.setattr(settings, "operation_llm_fallback", True)
    monkeypatch.setattr(agents, "_get_router_client",
                        _fake_router(_llm_response("delete_document",
                                                   '{"doc_id": "a1b2c3d4e5f60718"}')))
    r = handle_operation("把那份测试数据清掉吧")  # 词面不命中删除模式
    assert r["routed_by"] == "llm"
    assert r["pending_operation"] is not None
    assert r["pending_operation"]["danger"] is True
