"""LLM 客户端单测 — 熔断开关 / 重试退避 / 双重 JSON 编码。

核心可靠性逻辑此前零直接测试。MockTransport 替身, 不碰真实 API。
"""

import httpx
import pytest

from api.core.llm_client import LLMClient
from engines.common.llm_client import CircuitBreakerOpenError


def _client(handler, **kw) -> LLMClient:
    llm = LLMClient(
        base_url="https://fake.example",
        model="test-model",
        api_key="test-key",
        **kw,
    )
    # 替换真实 httpx 连接为 MockTransport
    llm._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5)
    return llm


# ── 响应解析 ──


def test_parse_completion_double_json():
    """TokenHub 双重编码: 外层是 JSON 字符串, 内层才是 chat/completions 结构。"""
    data = '{"choices":[{"message":{"content":"答案"}}],"usage":{"total_tokens":5,"prompt_tokens":2,"completion_tokens":3}}'
    resp = LLMClient._parse_completion(data)
    assert resp.content == "答案"
    assert resp.total_tokens == 5


def test_parse_completion_reasoning_fallback():
    """DeepSeek reasoning 模型 content 空时回退 reasoning_content。"""
    data = {"choices": [{"message": {"content": "", "reasoning_content": "推理过程"}}], "usage": {}}
    resp = LLMClient._parse_completion(data)
    assert resp.content == "推理过程"


def test_parse_completion_missing_usage_defaults_zero():
    data = {"choices": [{"message": {"content": "x"}}]}
    resp = LLMClient._parse_completion(data)
    assert resp.total_tokens == 0


# ── 熔断 ──


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    """连续失败达阈值 → 熔断打开, 后续请求直接拒绝。"""

    def handler(req):
        return httpx.Response(500, json={"error": "upstream"})

    llm = _client(handler, max_retries=0, circuit_breaker_threshold=2)
    with pytest.raises(httpx.HTTPStatusError):
        await llm.chat([{"role": "user", "content": "hi"}])
    with pytest.raises(httpx.HTTPStatusError):
        await llm.chat([{"role": "user", "content": "hi"}])
    assert llm._circuit_breaker.is_open is True
    with pytest.raises(CircuitBreakerOpenError):
        await llm.chat([{"role": "user", "content": "hi"}])


def test_circuit_breaker_half_open_recovers():
    """半开时间后放行并重置 open 状态 (状态机单元级验证, 不依赖 HTTP 全链路)。"""
    llm = _client(
        lambda req: httpx.Response(200, json={}),
        max_retries=0,
        circuit_breaker_threshold=1,
        circuit_breaker_recovery_time=0.01,  # 半开时间极小, 便于测试
    )
    llm._record_failure()  # 达阈值 → 熔断打开
    assert llm._circuit_breaker.is_open is True
    # 半开时间已过 → _is_circuit_open 放行并关闭熔断
    llm._circuit_breaker.last_failure_time -= 1.0
    assert llm._is_circuit_open() is False
    assert llm._circuit_breaker.is_open is False


# ── 重试 ──


@pytest.mark.asyncio
async def test_retry_429_then_success():
    """429 可重试, 指数退避后成功。"""
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": "rate"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 3}})

    llm = _client(handler, max_retries=2)
    resp = await llm.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "ok"
    assert calls["n"] == 3  # 失败2次 + 成功1次


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, json={"error": "busy"})

    llm = _client(handler, max_retries=1)
    with pytest.raises(httpx.HTTPStatusError):
        await llm.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2  # 1次 + 1次重试


@pytest.mark.asyncio
async def test_4xx_no_retry_no_circuit():
    """4xx 永久错误: 不重试也不记熔断 (与 5xx/429 语义区分)。"""
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401, json={"error": "auth"})

    llm = _client(handler, max_retries=3, circuit_breaker_threshold=2)
    with pytest.raises(httpx.HTTPStatusError):
        await llm.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1
    assert llm._circuit_breaker.is_open is False
