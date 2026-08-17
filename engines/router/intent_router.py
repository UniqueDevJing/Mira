"""意图路由引擎 — 规则优先，LLM 分类兜底，fallback 保护。

路由流程:
  规则计分 conf ≥ 0.85 → 直通 (source=rule)
  0 < conf < 0.85      → LLM 分类 (1.5s 超时, 复用 LLMClient 熔断) → source=llm
  LLM 失败/超时/熔断    → 默认 tech 库 (source=fallback)

引擎层零 FastAPI 依赖；LLM 客户端由调用方注入。
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from engines.router.routing_rules import (  # 规则配置单一事实来源, 与匹配算法分离
    CLASSIFY_PROMPT,
    FALLBACK_SKILL,
    LLM_TIMEOUT_S,
    ROUTE_THRESHOLD,
    SKILL_RULES,
    SKILLS,
)

# 路由规则与提示词见 engines/router/routing_rules.py (单一事实来源); 本文件只保留匹配算法。


@dataclass
class RoutingResult:
    skill: str
    kb: str | None
    confidence: float
    source: str  # rule | llm | fallback


class IntentRouter:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def route(self, question: str) -> RoutingResult:
        rule = self._rule_route(question)
        if rule is not None:
            return rule

        llm_result = await self._llm_route(question)
        if llm_result is not None:
            return llm_result

        return RoutingResult(FALLBACK_SKILL, SKILLS[FALLBACK_SKILL]["kb"], 0.0, "fallback")

    def _rule_route(self, question: str) -> RoutingResult | None:
        q = question.lower()
        best_skill = None
        best_conf = 0.0
        # 非 direct 技能的最高命中 (含来源), direct 让位时置信度与技能同源
        best_business = None
        best_business_conf = 0.0

        for skill, rules in SKILL_RULES.items():
            max_conf = 0.0
            for keyword, weight in rules:
                if self._kw_hit(keyword, q):
                    max_conf = max(max_conf, weight)
            if max_conf <= 0:
                continue
            if skill != "direct" and max_conf > best_business_conf:
                best_business, best_business_conf = skill, max_conf
            if max_conf > best_conf:
                best_skill, best_conf = skill, max_conf

        if not best_skill:
            return None

        # 寒暄词与真实问题同现时，direct 让位于业务/技术命中
        # (取权重最高的业务技能, conf 取同一来源 — 原实现 skill=service 但 conf 可能来自 tech, 错配)
        if best_skill == "direct" and best_business is not None:
            best_skill, best_conf = best_business, best_business_conf

        if best_conf >= ROUTE_THRESHOLD:
            kb = SKILLS[best_skill]["kb"]
            return RoutingResult(best_skill, kb, round(best_conf, 2), "rule")
        return None

    @staticmethod
    def _kw_hit(keyword: str, q: str) -> bool:
        """关键词命中: 英文按 ASCII 词边界匹配 (防 "hi" 命中 "this"、"api" 命中 "fastapi"),
        且紧邻中文不算边界 (兼容 "API接口" 无空格写法); 中文保持子串匹配。"""
        if keyword.isascii() and keyword.isalnum():
            kw = keyword.lower()
            start = 0
            while True:
                idx = q.find(kw, start)
                if idx < 0:
                    return False
                prev_ok = idx == 0 or not (q[idx - 1].isascii() and q[idx - 1].isalnum())
                end = idx + len(kw)
                next_ok = end >= len(q) or not (q[end].isascii() and q[end].isalnum())
                if prev_ok and next_ok:
                    return True
                start = idx + 1
        return keyword.lower() in q

    async def _llm_route(self, question: str) -> RoutingResult | None:
        if not self.llm_client:
            return None
        try:
            response = await asyncio.wait_for(
                self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": "你是一个意图分类器，只输出 JSON。"},
                        {"role": "user", "content": CLASSIFY_PROMPT.format(question=question)},
                    ],
                    temperature=0.1,
                    max_tokens=20,
                ),
                timeout=LLM_TIMEOUT_S,
            )
            skill = self._parse_skill(response.content)
            if skill and skill in SKILLS:
                logger.info("路由 LLM 分类: skill=%s, question=%s", skill, question[:50])
                return RoutingResult(skill, SKILLS[skill]["kb"], 0.9, "llm")
        except TimeoutError:
            logger.warning("路由 LLM 分类超时 (>%.1fs), 降级 fallback", LLM_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — 路由降级边界: LLM 失败走 fallback
            logger.warning("路由 LLM 分类失败: %s, 降级 fallback", str(e)[:120])
        return None

    @staticmethod
    def _parse_skill(content: str) -> str | None:
        """稳健解析 LLM 返回的 {"skill": "..."}。"""
        if not content:
            return None
        m = re.search(r'\{"skill"\s*:\s*"([^"]+)"\}', content)
        if m:
            return m.group(1).strip().lower()
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return str(data.get("skill", "")).strip().lower()
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug("skill JSON 解析失败, 走正则兜底: %s", str(e)[:80])
        m = re.search(r"(service|tech|direct)", content)
        return m.group(1) if m else None
