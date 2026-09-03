"""路由规则与配置 — 与匹配算法 (intent_router) 分离，作为单一事实来源。

调参 / 扩词表默认改 engines/doc_types.py (文档类型注册表); 亦可通过环境变量在部署期覆盖:
- RAG_ROUTING_RULES_FILE : 指向 JSON 文件 ({skill: [[关键词, 权重], ...]}), 覆盖内置词表
- RAG_ROUTE_THRESHOLD / RAG_LLM_TIMEOUT_S / RAG_FALLBACK_SKILL : 覆盖标量阈值

覆盖在进程启动时生效 (import 期解析), 文件缺失/非法时回退内置并告警。
"""

import json
import logging
import os

from engines.doc_types import DOC_TYPES

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


# P1' 多候选分类: 输出 top-2 带相对置信度(自评), 供选择性扇出使用。
# 与 DEFAULT_CLASSIFY_PROMPT 同构, 仅输出格式从单对象改为数组; conf 为 LLM 自评相对相关度(0-1)。
DEFAULT_CLASSIFY_MULTI_PROMPT = f"""你是意图分类器。根据用户问题判断最相关的技能, 按相关度从高到低输出前2个, 只输出 JSON 数组:
[{{"skill":"<类型id>", "conf": <0到1之间的相对置信度>}}, ...]

可选技能 (类型id 对应文档库):
{_TYPE_LABELS}

说明:
- 只输出最相关的前2个技能
- conf 是你对该技能与问题相关度的自评(0到1之间); 最相关的应接近1, 次相关的应明显更低
- 寒暄问候/与知识库无关的闲聊归为 direct

用户问题: {{question}}
只返回 JSON 数组, 不要其他文字。"""


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
# 扇出余量: 规则层 top-1 已达 ROUTE_THRESHOLD 时, 若次选与 top-1 的置信度差 <= 该值,
# 仍保留 top-2 供选择性扇出 (而非死板单路)。
#
# 动机: 单路 early-exit 会让"跨库主题"只检索一个库, 端到端召回丢失。典型如退换货 ——
# policy(制度: 期限/条件/比例) 与 service(话术: 流程/时效) 规则命中同为 0.9,
# 单路只会检索其中一个, 答案所在的另一个库完全不查, 表现为 recall_doc=0。
# 置信度接近 = 模型/规则自己也分不清归属, 此时扇出覆盖两库才是正确取舍。
#
# 调大 = 更容易扇出 (召回↑ / 延迟↑); 设 0 = 恢复严格单路 (仅 top-1 达标即返回单候选)。
FANOUT_MARGIN = _env_float("RAG_FANOUT_MARGIN", 0.1)

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
CLASSIFY_MULTI_PROMPT = DEFAULT_CLASSIFY_MULTI_PROMPT
