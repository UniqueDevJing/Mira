"""金融数值抽取与归一化 — 校验层的地基。

为什么需要它：
  金融答案里的数字不能只做「字符串是否出现在原文」的比对。
  同一笔营收可能写作 "1.2亿元" 或 "12000万元"，字符串不同但语义相同；
  反之 "1.2亿" 与 "1.2万" 字符串相似但相差 4 个数量级。
  所以必须先**抽取 → 归一化到统一基准 → 再比对**。

设计原则：
  - 纯标准库、零依赖、确定性（不调用 LLM —— 金融数字不容模型幻觉）
  - 归一化保留「语义单位」：元/美元/百分比/百分点 互不等价，不可混比
  - 支持中文数字（"百分之十五" / "三千万元"），金融口语与书面语并存
  - 保留原文位置，供前端做高亮溯源

归一化基准：
  货币类 → 归一到「基本单位」（元 / 美元），数量级词已折算进数值
  比率类 → 百分比归一为 0~1 的小数（15.3% → 0.153）；百分点保持原值（语义不同）
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ───────────────────── 数量级词表 ─────────────────────
# 必须紧跟数字才生效（通过「数字+可选空格+数量级词」的紧邻结构规避"百姓"等误判）。

_SCALES: list[tuple[str, float]] = [
    ("万亿", 1e12),
    ("千亿", 1e11),
    ("百亿", 1e10),
    ("十亿", 1e9),
    ("亿", 1e8),
    ("千万", 1e7),
    ("百万", 1e6),
    ("十万", 1e5),
    ("万", 1e4),
    ("千", 1e3),
    ("百", 1e2),
]

# ───────────────────── 单位词表 ─────────────────────
# 同一语义单位的不同写法归一到统一标识；货币不做汇率换算（无汇率时不可跨币种比对）。

_UNITS: list[tuple[str, str]] = [
    ("人民币", "CNY"), ("RMB", "CNY"), ("CNY", "CNY"), ("元", "CNY"), ("块", "CNY"),
    ("美元", "USD"), ("美金", "USD"), ("USD", "USD"),
    ("港元", "HKD"), ("港币", "HKD"), ("HKD", "HKD"),
    ("欧元", "EUR"), ("EUR", "EUR"),
    ("日元", "JPY"), ("JPY", "JPY"),
    ("个百分点", "PP"), ("百分点", "PP"), ("pp", "PP"), ("PP", "PP"),
    ("percent", "PERCENT"), ("%", "PERCENT"),
]

# 前缀式百分比（中文书面语）："百分之十五"
_PERCENT_PREFIX = "百分之"

# 无单位数字的默认语义（视作计数/纯数）
_DEFAULT_UNIT = "COUNT"

# 阿拉伯数字：支持千分位与小数
_NUM_AR = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
# 中文数字本体（**不含** 万亿，万亿交给数量级词处理，避免与 scale 冲突）
_NUM_CN = r"[零〇一二三四五六七八九两十百千]+"

_PATTERN = re.compile(
    r"(?:" + _PERCENT_PREFIX + r")?"
    r"(?P<num>" + _NUM_AR + r"|" + _NUM_CN + r")"
    r"\s*"
    r"(?P<scale>" + "|".join(s for s, _ in _SCALES) + r")?"
    r"\s*"
    r"(?P<unit>" + "|".join(re.escape(u) for u, _ in sorted(_UNITS, key=lambda x: -len(x[0]))) + r")?"
)

_SCALE_MAP = dict(_SCALES)
_UNIT_MAP = dict(_UNITS)

# ───────────────────── 中文数字解析 ─────────────────────
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn_to_number(text: str) -> float | None:
    """中文数字 → 数值。支持 "十五" / "二十三" / "一百二十" / "三千五百"。

    仅处理**不含 万/亿**的中文数字段（万亿由数量级词表负责），
    因此 "三千万" 会先解析出 "三千"=3000，再乘数量级 万=1e4 → 3e7。
    无法解析返回 None。
    """
    if not text:
        return None
    total = 0.0     # 已结算部分
    section = 0.0   # 当前节内累计
    number = 0.0    # 待结算的数字
    seen = False
    for ch in text:
        if ch in _CN_DIGIT:
            number = _CN_DIGIT[ch]
            seen = True
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            # "十五"：十前面没有数字，按 1 计
            section += (number if number else 1) * unit
            number = 0.0
            seen = True
        else:
            return None  # 出现非数字字符 → 整体不是中文数字
    if not seen:
        return None
    return total + section + number


@dataclass(frozen=True)
class NumberToken:
    """一个被抽取出的数值断言。"""

    raw: str          # 原文片段，如 "1.2亿元"
    value: float      # 归一化后的数值（数量级已折算；百分比已转小数）
    unit: str         # 归一化单位：CNY / USD / PERCENT / PP / COUNT ...
    start: int        # 在原文中的起始下标（供高亮溯源）
    end: int          # 结束下标（不含）

    @property
    def is_monetary(self) -> bool:
        return self.unit in ("CNY", "USD", "HKD", "EUR", "JPY")


def _to_float(num_text: str) -> float | None:
    """阿拉伯数字优先；失败则尝试中文数字。"""
    try:
        return float(num_text.replace(",", ""))
    except ValueError:
        return cn_to_number(num_text)


def extract_numbers(text: str) -> list[NumberToken]:
    """从左到右抽取所有数值断言。返回顺序与原文一致；无匹配返回空列表。"""
    if not text:
        return []
    out: list[NumberToken] = []
    for m in _PATTERN.finditer(text):
        num_text = m.group("num")
        base = _to_float(num_text)
        if base is None:
            continue

        scale_word = m.group("scale")
        unit_word = m.group("unit")
        has_pct_prefix = m.group(0).strip().startswith(_PERCENT_PREFIX)

        # 数量级优先：'亿' 等量词已把数值放大
        scale = _SCALE_MAP.get(scale_word, 1.0) if scale_word else 1.0
        value = base * scale

        if unit_word:
            unit = _UNIT_MAP.get(unit_word, _DEFAULT_UNIT)
        elif has_pct_prefix:
            unit = "PERCENT"
        else:
            unit = _DEFAULT_UNIT

        # 百分比统一为小数；百分点保持原值（≠百分比，语义不同不可混比）
        if unit == "PERCENT":
            value = value / 100.0

        out.append(
            NumberToken(
                raw=m.group(0).strip(),
                value=value,
                unit=unit,
                start=m.start(),
                end=m.end(),
            )
        )
    return out


def match_numbers(
    a: NumberToken, b: NumberToken, rel_tol: float = 0.005, abs_tol: float = 1e-9
) -> bool:
    """判断两个数值断言是否「语义等价」。

    规则：
      - 单位必须同族（CNY 只与 CNY 比；PERCENT 只与 PERCENT 比），否则直接不等
      - 数值按相对容差匹配（默认 0.5%，容忍四舍五入/单位换算误差）
    """
    if a.unit != b.unit:
        return False
    diff = abs(a.value - b.value)
    if diff <= abs_tol:
        return True
    denom = max(abs(a.value), abs(b.value))
    if denom == 0:
        return False
    return diff / denom <= rel_tol


def find_unsupported(
    claim_text: str, source_texts: list[str], rel_tol: float = 0.005
) -> list[NumberToken]:
    """找出「在来源中找不到支撑」的数值断言。

    这是溯源校验的核心：答案里每个数字都必须在给定来源中找到等价数值，
    找不到即为 unsupported（潜在的模型臆造/计算错误）。

    参数：
      claim_text   —— 待校验的答案文本
      source_texts —— 允许作为证据的来源文本列表（检索到的上下文）
    """
    claims = extract_numbers(claim_text)
    if not claims:
        return []
    pool: list[NumberToken] = []
    for s in source_texts:
        pool.extend(extract_numbers(s))
    if not pool:
        # 没有任何来源数字 → 全部无法溯源
        return list(claims)

    unsupported: list[NumberToken] = []
    for c in claims:
        if not any(match_numbers(c, p, rel_tol=rel_tol) for p in pool):
            unsupported.append(c)
    return unsupported


def format_token(t: NumberToken) -> str:
    """把断言格式化为可读字符串（用于日志/看板展示）。"""
    if t.unit == "PERCENT":
        return f"{t.value * 100:.4g}%"
    if t.unit == "PP":
        return f"{t.value:g}个百分点"
    if t.unit == "COUNT":
        return f"{t.value:g}"
    return f"{t.value:g} {t.unit}"
