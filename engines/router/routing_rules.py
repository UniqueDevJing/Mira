"""路由规则与配置 — 与匹配算法 (intent_router) 分离，作为单一事实来源。

调参 / 扩词表默认改 engines/doc_types.py (文档类型注册表); 亦可通过环境变量在部署期覆盖:
- RAG_ROUTING_RULES_FILE : 指向 JSON 文件 ({skill: [[关键词, 权重], ...]}), 覆盖内置词表
- RAG_ROUTE_THRESHOLD / RAG_LLM_TIMEOUT_S / RAG_FALLBACK_SKILL : 覆盖标量阈值

覆盖在进程启动时生效 (import 期解析), 文件缺失/非法时回退内置并告警。
"""

import json
import logging
import os

from engines.doc_types import DOC_TYPES, RAG_KBS

logger = logging.getLogger(__name__)

# skill 定义与知识库映射 — 由文档类型注册表动态生成 (单一事实来源)
SKILLS: dict[str, dict] = {
    tid: {"kb": spec.kb, "label": spec.label} for tid, spec in DOC_TYPES.items()
}

# 内置规则词表: 从注册表各类 routing_keywords 汇总 (词, 权重)
DEFAULT_SKILL_RULES: dict[str, list] = {
    tid: list(spec.routing_keywords) for tid, spec in DOC_TYPES.items()
}

# 各类型标签 (给 LLM 分类用)
_TYPE_LABELS = " / ".join(f"{tid}({spec.label})" for tid, spec in DOC_TYPES.items())

DEFAULT_CLASSIFY_PROMPT = f"""你是意图分类器。根据用户问题判断应路由到哪个技能，只输出一个 JSON，格式: {{"skill":"<类型id>"}}

可选技能 (类型id 对应文档库):
{_TYPE_LABELS}

说明:
- 各业务类型对应其专属知识库; 寒暄问候/自我介绍/与知识库无关的闲聊归为 direct
- 命中多个时选最贴合的一个

用户问题: {{question}}
只返回 JSON，不要其他文字。"""


def _env_float(name: str, default: float) -> float:
    """环境变量取浮点, 缺失或非数值回退 default (不抛错, 部署容错)。"""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("环境变量 %s=%r 非数值, 回退默认 %s", name, raw, default)
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw else default


def load_rules_from_file(path: str | os.PathLike) -> dict:
    """从 JSON 加载规则覆盖。格式: {skill: [[关键词, 权重], ...]}。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("路由规则文件顶层必须为对象 {skill: [[kw, weight], ...]}")
    return data


# 标量阈值: 部署期可经环境变量覆盖, 默认内置值
ROUTE_THRESHOLD = _env_float("RAG_ROUTE_THRESHOLD", 0.85)
LLM_TIMEOUT_S = _env_float("RAG_LLM_TIMEOUT_S", 1.5)
FALLBACK_SKILL = _env_str("RAG_FALLBACK_SKILL", "tech")

# 规则词表: 设置 RAG_ROUTING_RULES_FILE 即用自定义词表, 否则用内置 (从注册表生成)
_RULES_FILE = os.environ.get("RAG_ROUTING_RULES_FILE")
if _RULES_FILE:
    try:
        SKILL_RULES = load_rules_from_file(_RULES_FILE)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("路由规则文件 %s 加载失败, 回退内置词表: %s", _RULES_FILE, e)
        SKILL_RULES = DEFAULT_SKILL_RULES
else:
    SKILL_RULES = DEFAULT_SKILL_RULES

CLASSIFY_PROMPT = DEFAULT_CLASSIFY_PROMPT
