"""api/core/llm_client.py 的补充测试。

该模块此前覆盖率仅 53%，缺口集中在两条最需要保护的链路上：

1. **SSE 流式解析**（chat_stream，约 43 行）—— 逐行解析 `data:` 前缀、
   跳过 `[DONE]` 与非 JSON 行、区分 content 与 reasoning_content、
   单独 yield usage。解析逻辑一旦出错，SSE 推给前端的数据就会缺字段或乱序。
2. **重试与退避**（约 46 行）—— 429/5xx 与超时的指数退避、`Retry-After`
   响应头的尊重与非法值兜底、重试耗尽后的错误计数与抛出。
3. **单例获取** —— 异步/同步/VL 三条单例路径，含 VL 未配置时返回 None。

所有用例都用 httpx.MockTransport 或桩对象替代真实网络，不触碰外部服务。
"""

import asyncio
import json
import sys

import httpx
import pytest

sys.path.insert(0, ".")

from engines.common.llm_client import CircuitBreakerState  # noqa: E402

from api.core import llm_client as lc  # noqa: E402


def make_async_client(handler, **overrides):
    """构造一个不发真实请求的异步客户端。"""
    kwargs = {
        "base_url": "https://llm.test/v1",
        "model": "test-model",
        "api_key": "test-key",
        "circuit_breaker_state": CircuitBreakerState(half_open_time=30.0),
    }
    kwargs.update(overrides)
    client = lc.LLMClient(**kwargs)
    # 替换内部 httpx 客户端为MockTransport 版本
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=5.0,
        trust_env=False,
    )
    return client


def make_sync_client(handler, **overrides):
    kwargs = {
        "base_url": "https://llm.test/v1",
        "model": "test-model",
        "api_key": "test-key",
        "circuit_breaker_state": CircuitBreakerState(half_open_time=30.0),
    }
    kwargs.update(overrides)
    client = lc.SyncLLMClient(**kwargs)
    client._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=5.0,
        trust_env=False,
    )
    return client


def sse_response(*lines):
    """构造一个 SSE 风格的流式响应。"""
    body = "".join(f"{ln}\n" for ln in lines)
    return httpx.Response(200, text=body)


def delta_event(content=None, reasoning=None):
    payload = {}
    if content is not None:
        payload["content"] = content
    if reasoning is not None:
        payload["reasoning_content"] = reasoning
    return "data:" + json.dumps(
        {"choices": [{"delta": payload}]}, ensure_ascii=False
    )


class TestChatStreamParsing:
    """SSE 流式解析（stream_chat）：这是前端增量渲染依赖的协议层。"""

    def test_yields_content_deltas(self):
        """多个 content 分片应逐个 yield，且顺序保持。"""
        handler = lambda req: sse_response(  # noqa: E731
            delta_event("你"),
            delta_event("好"),
            delta_event("！"),
            "data: [DONE]",
        )
        client = make_async_client(handler)

        async def collect():
            return [c async for c in client.stream_chat([{"role": "user", "content": "hi"}])]

        out = asyncio.run(collect())
        contents = [e["content"] for e in out if e["type"] == "delta"]
        assert contents == ["你", "好", "！"]

    def test_reasoning_and_content_are_distinguished(self):
        """推理内容与正式回答必须走不同字段，不能混为一谈。"""
        handler = lambda req: sse_response(  # noqa: E731
            delta_event(reasoning="先思考一下…"),
            delta_event(content="最终答案"),
            "data: [DONE]",
        )
        client = make_async_client(handler)

        async def collect():
            return [c async for c in client.stream_chat([{"role": "user", "content": "hi"}])]

        out = asyncio.run(collect())
        reasoning = [e["reasoning"] for e in out if e["type"] == "delta"]
        contents = [e["content"] for e in out if e["type"] == "delta"]
        assert "先思考一下…" in reasoning
        assert "最终答案" in contents

    def test_usage_chunk_yielded_separately(self):
        """usage 是独立事件类型，供前端统计 token，不应混进 delta。"""
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        handler = lambda req: sse_response(  # noqa: E731
            delta_event(content="答"),
            "data:" + json.dumps({"choices": [], "usage": usage}),
            "data: [DONE]",
        )
        client = make_async_client(handler)

        async def collect():
            return [c async for c in client.stream_chat([{"role": "user", "content": "hi"}])]

        out = asyncio.run(collect())
        usage_events = [e for e in out if e["type"] == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0]["usage"]["total_tokens"] == 30

    def test_skips_non_data_lines(self):
        """SSE 协议里的注释行/空行（心跳）必须跳过，不能当JSON 解析。"""
        handler = lambda req: sse_response(  # noqa: E731
            ": keep-alive",
            "",
            delta_event("有效"),
            "data: [DONE]",
        )
        client = make_async_client(handler)

        async def collect():
            return [c async for c in client.stream_chat([{"role": "user", "content": "hi"}])]

        out = asyncio.run(collect())
        assert [e for e in out if e["type"] == "delta"][0]["content"] == "有效"

    def test_skips_malformed_json_without_aborting(self):
        """单行 JSON 损坏不应中断整个流 —— 真实场景下网络截断很常见。"""
        handler = lambda req: sse_response(  # noqa: E731
            delta_event("前段"),
            "data: {这不是合法JSON",
            delta_event("后段"),
            "data: [DONE]",
        )
        client = make_async_client(handler)

        async def collect():
            return [c async for c in client.stream_chat([{"role": "user", "content": "hi"}])]

        out = asyncio.run(collect())
        contents = [e["content"] for e in out if e["type"] == "delta"]
        assert contents == ["前段", "后段"]

    def test_done_sentinel_is_not_emitted(self):
        """[DONE] 是协议结束标记，绝不能作为内容推给前端。"""
        handler = lambda req: sse_response(delta_event("x"), "data: [DONE]")  # noqa: E731
        client = make_async_client(handler)

        async def collect():
            return [c async for c in client.stream_chat([{"role": "user", "content": "hi"}])]

        out = asyncio.run(collect())
        assert not any("[DONE]" in json.dumps(e, ensure_ascii=False) for e in out)

    def test_success_resets_circuit_breaker(self):
        """成功调用后应复位熔断计数，否则一次偶发失败会持续污染熔断器。"""
        handler = lambda req: sse_response(delta_event("ok"), "data: [DONE]")  # noqa: E731
        client = make_async_client(handler)
        cb = client._circuit_breaker
        cb.failure_count = 3  # 先人为置为失败态，验证成功调用后会复位

        async def collect():
            return [c async for c in client.stream_chat([{"role": "user", "content": "hi"}])]

        asyncio.run(collect())
        assert cb.failure_count == 0

    def test_4xx_does_not_trip_circuit_breaker(self):
        """4xx 是客户端/参数问题，重试无意义，不应累计熔断失败次数。"""
        cb = CircuitBreakerState(half_open_time=30.0)

        def handler(req):
            return httpx.Response(400, json={"error": "bad request"})

        client = make_async_client(handler, circuit_breaker_state=cb)

        async def collect():
            return [c async for c in client.stream_chat([{"role": "user", "content": "hi"}])]

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(collect())
        assert cb.failure_count == 0

    def test_5xx_trips_circuit_breaker(self):
        """5xx 是上游不可用，必须累计熔断失败次数以保护主链路。"""
        cb = CircuitBreakerState(half_open_time=30.0)

        def handler(req):
            return httpx.Response(503, json={"error": "unavailable"})

        client = make_async_client(handler, circuit_breaker_state=cb)

        async def collect():
            return [c async for c in client.stream_chat([{"role": "user", "content": "hi"}])]

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(collect())
        assert cb.failure_count >= 1

    def test_stream_request_payload_is_wellformed(self):
        """发给上游的请求体字段必须正确，尤其是 stream 标记。"""
        captured = {}

        def handler(req):
            captured["url"] = str(req.url)
            captured["auth"] = req.headers.get("Authorization")
            captured["body"] = json.loads(req.content)
            return sse_response("data: [DONE]")

        client = make_async_client(handler)

        async def collect():
            return [c async for c in client.stream_chat([{"role": "user", "content": "问"}])]

        asyncio.run(collect())
        assert captured["url"].endswith("/chat/completions")
        assert captured["auth"] == "Bearer test-key"
        assert captured["body"]["stream"] is True
        assert captured["body"]["model"] == "test-model"


class TestRetryAndBackoff:
    """重试与退避：控制上游抖动时的重试节奏。"""

    def test_retries_on_5xx_then_succeeds(self):
        """5xx 应重试；重试后成功则不抛异常。"""
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "终于成功"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                },
            )

        client = make_async_client(handler, max_retries=3)

        async def run():
            return await client.chat([{"role": "user", "content": "hi"}])

        # 把退避 sleep 缩短，避免测试真实等待
        orig_sleep = asyncio.sleep

        async def fast_sleep(_):
            return await orig_sleep(0)

        asyncio.sleep = fast_sleep
        try:
            resp = asyncio.run(run())
        finally:
            asyncio.sleep = orig_sleep

        assert resp.content == "终于成功"
        assert calls["n"] == 3

    def test_gives_up_after_max_retries(self):
        """重试耗尽后必须抛出原始异常，而不是静默返回空结果。"""
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            return httpx.Response(500, json={"error": "always down"})

        client = make_async_client(handler, max_retries=2)

        orig_sleep = asyncio.sleep

        async def fast_sleep(_):
            return await orig_sleep(0)

        async def run():
            return await client.chat([{"role": "user", "content": "hi"}])

        asyncio.sleep = fast_sleep
        try:
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(run())
        finally:
            asyncio.sleep = orig_sleep
        assert calls["n"] == 3  # 首次 + 2 次重试

    def test_timeout_is_retried(self):
        """超时属于可重试错误，不应立即失败。"""
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            if calls["n"] < 2:
                raise httpx.TimeoutException("simulated timeout", request=req)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "超时后成功"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        client = make_async_client(handler, max_retries=2)

        orig_sleep = asyncio.sleep

        async def fast_sleep(_):
            return await orig_sleep(0)

        asyncio.sleep = fast_sleep
        try:
            resp = asyncio.run(client.chat([{"role": "user", "content": "hi"}]))
        finally:
            asyncio.sleep = orig_sleep

        assert resp.content == "超时后成功"

    def test_unexpected_error_is_not_retried(self):
        """非 HTTP/超时的异常（如代码 bug）重试无意义，应立即上抛。"""
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            raise ValueError("程序逻辑错误")

        client = make_async_client(handler, max_retries=3)

        async def run():
            return await client.chat([{"role": "user", "content": "hi"}])

        with pytest.raises(ValueError):
            asyncio.run(run())
        assert calls["n"] == 1  # 只调一次，没浪费重试预算


class TestSingletons:
    """单例获取：async / sync / VL 三条路径。"""

    def setup_method(self):
        # 每个用例前重置单例，避免相互污染
        lc._llm_client = None
        lc._sync_llm_client = None
        lc._sync_vl_client = None

    def teardown_method(self):
        lc._llm_client = None
        lc._sync_llm_client = None
        lc._sync_vl_client = None

    def test_get_async_client_is_singleton(self):
        """重复获取必须返回同一实例，否则连接池与熔断状态会失效。"""
        a = lc.get_llm_client()
        b = lc.get_llm_client()
        assert a is b

    def test_get_sync_client_is_singleton(self):
        a = lc.get_sync_llm_client()
        b = lc.get_sync_llm_client()
        assert a is b

    def test_vl_client_returns_none_when_not_configured(self, monkeypatch):
        """未配置视觉模型时返回 None，让调用方回落到 OCR。"""
        from api.config import settings

        monkeypatch.setattr(settings, "llm_vl_model", "", raising=False)
        assert lc.get_sync_llm_client(vl_model=True) is None

    def test_vl_client_created_when_configured(self, monkeypatch):
        from api.config import settings

        monkeypatch.setattr(settings, "llm_vl_model", "vl-model-x", raising=False)
        client = lc.get_sync_llm_client(vl_model=True)
        assert client is not None
        assert client.model == "vl-model-x"

    def test_vl_and_text_clients_are_distinct(self, monkeypatch):
        """视觉模型与文本模型必须是两个独立单例，不能互相覆盖。"""
        from api.config import settings

        monkeypatch.setattr(settings, "llm_vl_model", "vl-model-x", raising=False)
        vl = lc.get_sync_llm_client(vl_model=True)
        text = lc.get_sync_llm_client()
        assert vl is not text


class TestJsonMode:
    """json_mode：结构化输出的请求侧标记。"""

    def test_json_mode_sets_response_format(self):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        client = make_async_client(handler)

        async def run():
            return await client.chat(
                [{"role": "user", "content": "给JSON"}], json_mode=True
            )

        asyncio.run(run())
        assert captured["body"].get("response_format") == {"type": "json_object"}

    def test_non_json_mode_omits_response_format(self):
        """普通对话不应带 response_format，否则某些上游会强制输出 JSON。"""
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "普通文本"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        client = make_async_client(handler)

        async def run():
            return await client.chat([{"role": "user", "content": "聊天"}])

        asyncio.run(run())
        assert "response_format" not in captured["body"]