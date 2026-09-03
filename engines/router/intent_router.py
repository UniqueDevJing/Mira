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

from engines.router.routing_rules import (  # 规则配置单一事实来源, 与匹配算法分离
    CLASSIFY_MULTI_PROMPT,
    CLASSIFY_PROMPT,
    FALLBACK_SKILL,
    FANOUT_MARGIN,
    LLM_TIMEOUT_S,
    ROUTE_THRESHOLD,
    SKILL_RULES,
    SKILLS,
)

logger = logging.getLogger(__name__)

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
        """单候选兼容入口: 返回 route_multi 的 top-1, 无候选降级 fallback。"""
        candidates = await self.route_multi(question)
        if candidates:
            return candidates[0]
        return RoutingResult(FALLBACK_SKILL, SKILLS[FALLBACK_SKILL]["kb"], 0.0, "fallback")

    async def route_multi(self, question: str, top_n: int = 3) -> list[RoutingResult]:
        """P1' 多候选路由: 返回 top_n 候选(降序), 供 selective fanout 使用。

        - 规则层确定 (best conf >= ROUTE_THRESHOLD): early-exit, 不调 LLM(省延迟)。
          但只有次选与 top-1 置信度差 > FANOUT_MARGIN 时才真的退化为单候选; 差距落在
          余量内 = 主题横跨多库(如退换货同时命中 policy 制度与 service 话术), 返回 top-2
          交扇出并行检索, 主路由归因仍是 top-1。
        - 模糊 (规则低于阈值或无命中): 调 LLM 取 top-2 自评候选, 与规则候选合并去重取最高 conf
        - 始终保留 direct 让位逻辑(有业务命中时 direct 不进候选, 与 _rule_route 一致)
        """
        rule_cands = self._rule_route_all(question)
        if rule_cands and rule_cands[0].confidence >= ROUTE_THRESHOLD:
            # 规则足够确定 → early-exit, 不调 LLM
            # 但"确定"≠"唯一归属": 次选与 top-1 置信度接近时说明规则自己也分不清库归属,
            # 单路会漏掉答案所在的另一个库 (P1' 实证: 退货题单路 service, policy 库完全不查
            # → recall_doc=0)。此时保留 top-2 让扇出覆盖, 主路由归因不变。
            if (
                len(rule_cands) > 1
                and rule_cands[1].confidence >= rule_cands[0].confidence - FANOUT_MARGIN
            ):
                return rule_cands[:2]
            return rule_cands[:1]

        # 模糊: 合并规则候选 + LLM top-2
        merged: dict[str, RoutingResult] = {r.skill: r for r in rule_cands}
        llm_cands = await self._llm_route_multi(question, top_n=2)
        for r in llm_cands:
            if r.skill not in merged or r.confidence > merged[r.skill].confidence:
                merged[r.skill] = r

        ranked = sorted(merged.values(), key=lambda r: r.confidence, reverse=True)
        return ranked[:top_n]

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

    def _rule_route_all(self, question: str) -> list[RoutingResult]:
        """P1' 规则层全候选(降序): 所有 conf>0 的技能, 含 direct 让位处理。供 route_multi 使用。"""
        q = question.lower()
        scored: list[RoutingResult] = []
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
            scored.append(RoutingResult(skill, SKILLS[skill]["kb"], round(max_conf, 2), "rule"))
        # direct 让位: 有业务命中时 direct 不进候选(与 _rule_route 行为一致)
        scored = [r for r in scored if not (r.skill == "direct" and best_business is not None)]
        scored.sort(key=lambda r: r.confidence, reverse=True)
        return scored

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
                        # 用 replace 注入问题: CLASSIFY_PROMPT 是 f-string, 内含字面 JSON 示例 {"skill":...},
                        # 若用 .format() 二次解析会把字面花括号当占位符报 KeyError; replace 只替换 {question} 标记
                        {"role": "user", "content": CLASSIFY_PROMPT.replace("{question}", question)},
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

    async def _llm_route_multi(self, question: str, top_n: int = 2) -> list[RoutingResult]:
        """P1' LLM 多候选: 输出 top_n 带自评置信度的技能。失败/超时返回空(由规则候选兜底)。"""
        if not self.llm_client:
            return []
        try:
            response = await asyncio.wait_for(
                self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": "你是一个意图分类器，只输出 JSON 数组。"},
                        {"role": "user", "content": CLASSIFY_MULTI_PROMPT.replace("{question}", question)},
                    ],
                    temperature=0.1,
                    max_tokens=60,
                ),
                timeout=LLM_TIMEOUT_S,
            )
            parsed = self._parse_skills_with_conf(response.content)
            valid = [
                RoutingResult(skill, SKILLS[skill]["kb"], round(conf, 2), "llm")
                for skill, conf in parsed
                if skill in SKILLS
            ]
            if valid:
                logger.info("路由 LLM 多候选: %s", [(r.skill, r.confidence) for r in valid[:top_n]])
                return valid[:top_n]
        except TimeoutError:
            logger.warning("路由 LLM 多候选分类超时 (>%.1fs), 降级规则候选", LLM_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — 路由降级边界: LLM 失败走规则/fallback
            logger.warning("路由 LLM 多候选分类失败: %s, 降级规则候选", str(e)[:120])
        return []

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
        # 动态匹配: 从 SKILLS 所有类型名中查找, 兼容新增文档类型
        for skill in SKILLS:
            if skill in content.lower():
                return skill
        return None

    @staticmethod
    def _parse_skills_with_conf(content: str) -> list[tuple[str, float]]:
        """解析 LLM 返回的 [{"skill":..,"conf":..}, ...] 为 (skill, conf) 列表。防御性: 缺字段/非法值跳过。"""
        if not content:
            return []
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            # 退化: 尝试当单对象解析
            single = IntentRouter._parse_skill(content)
            return [(single, 0.9)] if single else []
        if isinstance(data, list):
            out: list[tuple[str, float]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                skill = str(item.get("skill", "")).strip().lower()
                if not skill:
                    continue
                try:
                    conf = float(item.get("conf", 0.0))
                except (TypeError, ValueError):
                    conf = 0.0
                conf = max(0.0, min(1.0, conf))  # clamp 防越界
                out.append((skill, conf))
            return out
        if isinstance(data, dict):
            # 单对象 JSON: 退化解析(默认 conf 0.9)
            single = str(data.get("skill", "")).strip().lower()
            return [(single, 0.9)] if single else []
        return []
