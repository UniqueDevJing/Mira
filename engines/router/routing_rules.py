"""路由规则与配置 — 与匹配算法 (intent_router) 分离，作为单一事实来源。

调参 / 扩词表默认改本文件；亦可通过环境变量在部署期覆盖，无需改代码：
- RAG_ROUTING_RULES_FILE : 指向 JSON 文件 ({skill: [[关键词, 权重], ...]})，覆盖内置词表
- RAG_ROUTE_THRESHOLD / RAG_LLM_TIMEOUT_S / RAG_FALLBACK_SKILL : 覆盖标量阈值

覆盖在进程启动时生效（import 期解析），文件缺失/非法时回退内置并告警。
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# skill 定义与知识库映射 (与部署无关, 固定)
SKILLS: dict[str, dict] = {
    "service": {"kb": "service", "label": "客服话术知识库"},
    "tech": {"kb": "tech", "label": "技术文档知识库"},
    "direct": {"kb": None, "label": "直接回答"},
}

# 内置规则词表: (关键词, 置信权重)。强信号词 0.9+ 可直通，弱信号 0.4 仅作 LLM 提示。
DEFAULT_SKILL_RULES: dict[str, list] = {
    "direct": [
        ("你好", 0.95),
        ("您好", 0.95),
        ("hello", 0.95),
        ("hi", 0.95),
        ("谢谢", 0.95),
        ("感谢", 0.95),
        ("再见", 0.95),
        ("在吗", 0.95),
        ("你是谁", 0.95),
        ("你能做什么", 0.95),
    ],
    "service": [
        ("退款", 0.9),
        ("退货", 0.9),
        ("物流", 0.9),
        ("订单", 0.9),
        ("售后", 0.9),
        ("客服", 0.9),
        ("发票", 0.9),
        ("发货", 0.9),
        ("运费", 0.9),
        ("签收", 0.9),
        ("投诉", 0.9),
        ("赔偿", 0.9),
        ("优惠", 0.7),
        ("价格", 0.7),
        ("包邮", 0.7),
    ],
    "tech": [
        ("部署", 0.9),
        ("架构", 0.9),
        ("API", 0.9),
        ("接口", 0.9),
        ("数据库", 0.9),
        ("缓存", 0.9),
        ("安装", 0.9),
        ("配置", 0.9),
        ("版本", 0.7),
        ("环境", 0.7),
        ("报错", 0.7),
        ("FastAPI", 0.9),
        ("Docker", 0.9),
        ("系统", 0.6),
        ("技术", 0.6),
        ("文档", 0.5),
    ],
}

DEFAULT_CLASSIFY_PROMPT = """你是意图分类器。根据用户问题判断应路由到哪个技能，只输出一个 JSON，格式: {{"skill":"service|tech|direct"}}

技能说明:
- service: 客服/售后/订单/物流等业务问题，查客服话术知识库
- tech: 技术/架构/部署/开发问题，查技术文档知识库
- direct: 寒暄问候、自我介绍、与知识库无关的闲聊，直接回答

用户问题: {question}
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

# 规则词表: 设置 RAG_ROUTING_RULES_FILE 即用自定义词表, 否则用内置
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
