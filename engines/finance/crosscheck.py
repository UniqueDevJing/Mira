"""多源交叉校验 + 时效校验 — 金融数据的「互证」层。

为什么需要它：
  单一来源的数字「能在原文中找到」不代表「正确」——来源本身可能过期、
  可能有错。金融数据的可信来自**多源互证**：同一指标在多个独立来源中
  数值一致才算可信；数值矛盾则必须暴露给上层（拒答或降级）。

核心难点 —— 「什么算同一指标」：
  两个来源的数字不同 ≠ 冲突。"营收 1.2 亿元" 与 "利润 3 千万元" 都是 CNY
  但讲的是不同指标，不构成冲突。判断「同一指标」分三层：
    1. 指标词锚定：token 前方最近距离内出现同一指标词（营收/净利润/…）
    2. 年份消歧：锚上下文中的 4 位年份不同 → 视为不同期间的同一指标，非冲突
    3. 上下文回退：都找不到指标词时，用前缀窗口的字符 bigram 重合度近似

时效校验：
  金融数据有保质期。来源带 as_of 时间点，超过 max_age 判 stale；
  缺失 as_of 判 unknown（无法证明新鲜，同样是风险）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from .numeric import NumberToken, extract_numbers, match_numbers

# ───────────────────── 指标词表 ─────────────────────
# 按长度降序尝试，避免 "净利润" 被 "利润" 抢先匹配。

_METRIC_WORDS: list[str] = [
    "归母净利润", "营业收入", "营收增速", "利润增速", "销售费用", "管理费用",
    "研发费用", "财务费用", "营业成本", "净利润", "营业利润", "毛利润",
    "总资产", "净资产", "每股收益", "市盈率", "市净率", "毛利率", "净利率",
    "总营收", "总成本", "总股本", "毛利率", "营收", "收入", "利润", "毛利",
    "成本", "费用", "支出", "市值", "股价", "增速", "增长", "下降", "下滑",
    "亏损", "盈利", "现金流", "分红", "股息", "负债", "存货", "销量", "产量",
    "产能", "成交额", "成交量", "换手率", "税", "ROE", "ROA", "EPS",
]
_METRIC_WORDS = sorted(set(_METRIC_WORDS), key=len, reverse=True)

# 指标词搜索半径：token 前方多少字符内找指标词
_METRIC_WINDOW = 30
# 回退窗口：无指标词时取 token 前方多少字符做上下文指纹
_ANCHOR_WINDOW = 24
# bigram Jaccard 重合度阈值
_ANCHOR_JACCARD = 0.35

# 锚上下文中的年份（用于区分不同报告期）
_YEAR_RE = re.compile(r"(20\d{2})")

_FRESHNESS_FORMATS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d")


# ───────────────────── 数据结构 ─────────────────────


@dataclass(frozen=True)
class Source:
    """一个可引用的证据来源。

    name  —— 来源的唯一标识（同一文件被切成多个 chunk 时必须互不相同，
             否则跨 chunk 矛盾会被"同名跳过"规则整体漏检）。
    group —— 文件级身份（如文档标题/文件名），用于区分"同文件不同 chunk"
             (warn 级) 与"不同文件"(fail 级) 两类矛盾。None 表示文件身份未知。
    as_of 是「数据时效点」（该来源陈述事实成立的时点，通常是报告期/发布日），
    接受 datetime / date / ISO 字符串 / None。None 表示来源未声明时效。
    """

    name: str
    text: str
    as_of: datetime | date | str | None = None
    group: str | None = None

    def as_of_dt(self) -> datetime | None:
        v = self.as_of
        if v is None:
            return None
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if isinstance(v, date):
            return datetime(v.year, v.month, v.day, tzinfo=timezone.utc)
        if isinstance(v, str):
            for fmt in _FRESHNESS_FORMATS:
                try:
                    dt = datetime.strptime(v.strip(), fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        return None  # 无法解析的格式 → 视同未声明


@dataclass(frozen=True)
class Conflict:
    """跨来源（或同来源内部）的数值矛盾。"""

    metric: str | None   # 锚定的指标词；None 表示用上下文回退锚定
    unit: str
    a_source: str
    a_value: float
    a_raw: str
    b_source: str
    b_value: float
    b_raw: str

    def describe(self) -> str:
        name = self.metric or "(上下文锚定)"
        return (
            f"指标[{name}] 单位[{self.unit}]: "
            f"{self.a_source} 称 {self.a_raw}({self.a_value:g}) vs "
            f"{self.b_source} 称 {self.b_raw}({self.b_value:g})"
        )


@dataclass(frozen=True)
class StaleFinding:
    """时效风险。reason: stale(超期) / unknown(未声明时效)。"""

    source: str
    as_of: datetime | None
    age_days: float | None
    max_age_days: float
    reason: str


# ───────────────────── 指标锚定 ─────────────────────


def _prefix_window(text: str, start: int, size: int) -> str:
    return text[max(0, start - size):start]


def _nearest_metric(window: str) -> str | None:
    """窗口内最靠近 token 的指标词；无则 None。

    必须先消除嵌套子串：窗口含 "归母净利润" 时，"净利润"/"利润" 作为其
    子串匹配位置更靠右，若直接取最右会把长词遮蔽掉。做法：收集所有
    词的匹配区间 → 丢弃被其他区间完全包含的 → 取区间末端最靠右者。
    """
    spans: list[tuple[int, int, str]] = []  # (start, end, word)
    for word in _METRIC_WORDS:
        start = 0
        while True:
            idx = window.find(word, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(word), word))
            start = idx + 1
    kept = [
        s for s in spans
        if not any(o is not s and o[0] <= s[0] and s[1] <= o[1] for o in spans)
    ]
    if not kept:
        return None
    best = max(kept, key=lambda s: (s[1], s[1] - s[0]))
    return best[2]


def nearest_metric(text: str, token: NumberToken) -> str | None:
    """token 前方 _METRIC_WINDOW 字符内最近的指标词（完整短语优先）。"""
    window = _prefix_window(text, token.start, _METRIC_WINDOW)
    return _nearest_metric(window)


def _anchor_year(text: str, token: NumberToken) -> int | None:
    """锚窗口内的 4 位年份，用于区分不同报告期。"""
    window = _prefix_window(text, token.start, _METRIC_WINDOW)
    m = _YEAR_RE.search(window)
    return int(m.group(1)) if m else None


def _bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def same_metric(
    ta: NumberToken, text_a: str, tb: NumberToken, text_b: str
) -> tuple[bool, str | None]:
    """判断两个 token 是否在陈述同一指标。返回 (是否同一指标, 锚定的指标词)。

    保守策略：判不准时宁可放过（不报冲突），冲突检出要求高置信锚定。
    """
    # 年份消歧：锚上下文年份不同 → 不同报告期，非冲突
    ya, yb = _anchor_year(text_a, ta), _anchor_year(text_b, tb)
    if ya is not None and yb is not None and ya != yb:
        return False, None

    ma, mb = nearest_metric(text_a, ta), nearest_metric(text_b, tb)
    if ma is not None and mb is not None:
        return (ma == mb), (ma if ma == mb else None)
    # 一方有指标词一方没有 → 无法确认同一指标，放过
    if ma is not None or mb is not None:
        return False, None
    # 双无指标词 → 上下文 bigram 回退
    wa = _prefix_window(text_a, ta.start, _ANCHOR_WINDOW)
    wb = _prefix_window(text_b, tb.start, _ANCHOR_WINDOW)
    same = _jaccard(_bigrams(wa), _bigrams(wb)) >= _ANCHOR_JACCARD
    return same, None


# ───────────────────── 交叉校验 ─────────────────────


def cross_check(
    sources: list[Source],
    rel_tol: float = 0.005,
    include_intra: bool = True,
) -> list[Conflict]:
    """检测同一指标在多个来源间的数值矛盾。

    逻辑：对每一对「同单位 + 同指标锚定」的数值断言，若数值超出容差
    即报 Conflict。同来源内部的自我矛盾（include_intra）同样有价值
    ——自相矛盾的来源本身不可信。

    注意单位不同（如 CNY vs USD）不构成冲突：无汇率不可比，只能放过。
    """
    tagged: list[tuple[Source, NumberToken]] = []
    for s in sources:
        for t in extract_numbers(s.text):
            tagged.append((s, t))

    def _is_year_like(t: NumberToken) -> bool:
        """年份不是财务断言：1900~2100 的整数 COUNT 视为年份，
        不参与矛盾配对（否则"2024"与员工数 1860 的窗口 bigram 重叠
        会制造伪冲突，语料一多必然误报）。"""
        return t.unit == "COUNT" and 1900 <= t.value <= 2100 and float(t.value).is_integer()

    conflicts: list[Conflict] = []
    for i in range(len(tagged)):
        sa, ta = tagged[i]
        if _is_year_like(ta):
            continue
        for j in range(i + 1, len(tagged)):
            sb, tb = tagged[j]
            if _is_year_like(tb):
                continue
            if not include_intra and sa.name == sb.name:
                continue
            if ta.unit != tb.unit:
                continue
            if match_numbers(ta, tb, rel_tol=rel_tol):
                continue  # 容差内一致 → 互证通过
            same, metric = same_metric(ta, sa.text, tb, sb.text)
            if not same:
                continue
            conflicts.append(
                Conflict(
                    metric=metric,
                    unit=ta.unit,
                    a_source=sa.name, a_value=ta.value, a_raw=ta.raw,
                    b_source=sb.name, b_value=tb.value, b_raw=tb.raw,
                )
            )
    return conflicts


# ───────────────────── 时效校验 ─────────────────────


def check_freshness(
    sources: list[Source],
    now: datetime | None = None,
    max_age: timedelta = timedelta(days=365),
) -> list[StaleFinding]:
    """检查来源时效。超 max_age → stale；未声明/无法解析 as_of → unknown。

    now 可注入以便测试；naive 的 now 按 UTC 处理。
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    findings: list[StaleFinding] = []
    for s in sources:
        dt = s.as_of_dt()
        if dt is None:
            findings.append(
                StaleFinding(
                    source=s.name, as_of=None, age_days=None,
                    max_age_days=max_age.total_seconds() / 86400,
                    reason="unknown",
                )
            )
            continue
        age = now - dt
        if age > max_age:
            findings.append(
                StaleFinding(
                    source=s.name, as_of=dt,
                    age_days=age.total_seconds() / 86400,
                    max_age_days=max_age.total_seconds() / 86400,
                    reason="stale",
                )
            )
    return findings
