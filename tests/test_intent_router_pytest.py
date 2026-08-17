"""IntentRouter 单元测试 — 规则/LLM/fallback 三源路由。"""

import asyncio

import pytest

from engines.router.intent_router import IntentRouter


def run(coro):
    return asyncio.run(coro)


class FakeLLM:
    def __init__(self, content: str = "", error: Exception | None = None):
        self.content = content
        self.error = error

    async def chat(self, messages, **kwargs):
        if self.error:
            raise self.error
        from api.core.llm_client import LLMResponse

        return LLMResponse(content=self.content)


# ── 规则直通 (conf >= 0.85) ──


def test_rule_service():
    res = run(IntentRouter().route("退货流程是什么"))
    assert res.skill == "service" and res.source == "rule"
    assert res.confidence >= 0.85 and res.kb == "service"


def test_rule_tech():
    res = run(IntentRouter().route("系统架构如何部署"))
    assert res.skill == "tech" and res.source == "rule" and res.kb == "tech"


def test_rule_direct_greeting():
    res = run(IntentRouter().route("你好"))
    assert res.skill == "direct" and res.source == "rule"


def test_greeting_with_business_routes_business():
    """寒暄词与业务词同现时 direct 让位，避免误分类。"""
    res = run(IntentRouter().route("你好，退货怎么处理"))
    assert res.skill == "service" and res.source == "rule"


# ── LLM 分类 (conf < 0.85 时) ──


def test_llm_classify_on_ambiguous():
    llm = FakeLLM(content='{"skill":"tech"}')
    res = run(IntentRouter(llm_client=llm).route("帮我查一下这个怎么弄"))
    assert res.skill == "tech" and res.source == "llm"


def test_llm_classify_service():
    llm = FakeLLM(content='{"skill": "service"}')
    res = run(IntentRouter(llm_client=llm).route("完全没有规则词的模糊问题"))
    assert res.skill == "service" and res.source == "llm"


def test_llm_error_fallback():
    llm = FakeLLM(error=RuntimeError("api down"))
    res = run(IntentRouter(llm_client=llm).route("没有规则词的问题"))
    assert res.skill == "tech" and res.source == "fallback"


def test_no_llm_fallback_tech():
    res = run(IntentRouter().route("完全没有规则词的模糊问题"))
    assert res.skill == "tech" and res.source == "fallback"


# ── JSON 解析容错 ──


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"skill":"tech"}', "tech"),
        ('```json\n{"skill": "service"}\n```', "service"),
        ("  direct  ", "direct"),
        ('{"skill":"unknown"}', "unknown"),  # 解析成功，但路由层会因不在 SKILLS 中而过滤
        ("not json at all", None),
        ("", None),
    ],
)
def test_parse_skill(raw, expected):
    assert IntentRouter._parse_skill(raw) == expected
