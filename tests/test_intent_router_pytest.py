"""P1' 路由多候选 (IntentRouter.route_multi) 契约测试。

全部使用 llm_client=None: 模糊时 _llm_route_multi 直接返回空(不调 LLM),
仅验证规则层多候选 + early-exit + direct 让位 + 解析 clamp, 零外部依赖。
项目约定 sync 测试 + asyncio.run (见 test_p0_contract_pytest)。
"""

import asyncio

import pytest

from engines.router.intent_router import IntentRouter


@pytest.fixture
def patched_rules(monkeypatch):
    """注入极简受控规则集, 避免依赖真实词表波动。"""
    skills = {
        "a": {"kb": "ka"},
        "b": {"kb": "kb"},
        "direct": {"kb": None},
        "tech": {"kb": "ktech"},
    }
    rules = {
        "a": [("退款", 0.9)],
        "b": [("物流", 0.7)],
        "direct": [("你好", 0.5)],
        "tech": [("代码", 0.6)],
    }
    monkeypatch.setattr("engines.router.intent_router.SKILLS", skills)
    monkeypatch.setattr("engines.router.intent_router.SKILL_RULES", rules)
    monkeypatch.setattr("engines.router.intent_router.ROUTE_THRESHOLD", 0.85)
    monkeypatch.setattr("engines.router.intent_router.FALLBACK_SKILL", "tech")
    return skills, rules


def test_route_multi_rule_early_exit_high_conf(patched_rules):
    """规则 best conf >= 阈值 → early-exit 单候选, 不调 LLM。"""
    router = IntentRouter(llm_client=None)
    cands = asyncio.run(router.route_multi("怎么退款"))
    assert len(cands) == 1
    assert cands[0].skill == "a"
    assert cands[0].confidence == 0.9
    assert cands[0].source == "rule"


def test_route_multi_rule_multi_hit_ambiguous(patched_rules):
    """best conf < 阈值(模糊) + 多命中 → 返回多候选(降序), LLM=None 不补全。"""
    _skills, rules = patched_rules
    rules["a"] = [("退款", 0.8)]  # 降到模糊区
    router = IntentRouter(llm_client=None)
    cands = asyncio.run(router.route_multi("退款和物流怎么处理"))
    assert len(cands) == 2
    assert [c.skill for c in cands] == ["a", "b"]
    assert cands[0].confidence >= cands[1].confidence


def test_route_multi_no_hit_returns_empty(patched_rules):
    """无规则命中 → route_multi 空; route() 兼容降级 fallback。"""
    router = IntentRouter(llm_client=None)
    assert asyncio.run(router.route_multi("今天天气真不错")) == []
    r = asyncio.run(router.route("今天天气真不错"))
    assert r.skill == "tech"
    assert r.confidence == 0.0
    assert r.source == "fallback"


def test_route_direct_yields_to_business(patched_rules):
    """direct 与业务同现时, direct 不进候选(让位业务)。"""
    router = IntentRouter(llm_client=None)
    cands = asyncio.run(router.route_multi("你好，退款怎么操作"))
    skills = [c.skill for c in cands]
    assert "direct" not in skills
    assert "a" in skills


def test_route_compat_returns_top1(patched_rules):
    """route() 与 route_multi()[0] 一致(兼容旧单候选调用方)。"""
    router = IntentRouter(llm_client=None)
    multi = asyncio.run(router.route_multi("怎么退款"))
    single = asyncio.run(router.route("怎么退款"))
    assert single.skill == multi[0].skill
    assert single.confidence == multi[0].confidence


def test_parse_skills_clamps_conf():
    """_parse_skills_with_conf 越界 conf 被 clamp 到 [0,1]。"""
    parsed = IntentRouter._parse_skills_with_conf(
        '[{"skill":"x","conf":1.5},{"skill":"y","conf":-0.3}]'
    )
    assert parsed[0] == ("x", 1.0)
    assert parsed[1] == ("y", 0.0)


def test_parse_skills_handles_single_object():
    """退化: 单对象 JSON 也能解析为单候选(默认 conf 0.9)。"""
    parsed = IntentRouter._parse_skills_with_conf('{"skill":"x"}')
    assert parsed == [("x", 0.9)]


def test_parse_skills_skips_invalid_items():
    """缺失 skill / 非 dict 项被跳过, 不抛异常。"""
    parsed = IntentRouter._parse_skills_with_conf('[{"conf":0.5}, "junk", {"skill":"z","conf":0.8}]')
    assert parsed == [("z", 0.8)]
