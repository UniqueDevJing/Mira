"""金融意图检测 — 领域感知分发 (P5)。

真实问题: 金融校验 Verifier 若靠全局开关一刀切, 两头都不对:
  - 全局开  → 所有咨询(含非金融)都被数字校验拖累, 非金融数字内容有误拒风险;
  - 全局关  → 金融问题裸奔, 校验能力形同虚设。

方案: 规则级意图检测(零 LLM 成本/零在线延迟), 金融问题自动启用 Verifier,
非金融问题维持原行为。词表与 crosscheck._METRIC_WORDS 同源维护。
"""

from __future__ import annotations

import re

# 金融指标/报表术语 — 命中即视为金融域问题。
# 短词放前面便于维护; 匹配用逐词 in, 无顺序依赖。
_FINANCE_TERMS = (
    # 报表/报告
    "财报", "年报", "季报", "半年报", "财务报表", "财务报告", "业绩报",
    # 收入/利润
    "营收", "营业收入", "净利润", "净利", "毛利", "毛利率", "净利率", "利润率",
    "营业利润", "利润总额", "归母净利",
    # 成本/费用
    "营业成本", "研发费用", "销售费用", "管理费用", "财务费用",
    # 资产/负债
    "资产负债", "负债率", "总资产", "净资产", "现金流", "经营现金流",
    # 比率/指标
    "市盈率", "市净率", "每股收益", "roe", "roa", "资产周转",
    # 增长/变动
    "同比增长", "同比下降", "环比增长", "环比下降", "营收增长", "增长率",
    "同比增长率", "增速",
    # 分红/股息
    "分红", "股息", "派息", "每股分红",
)

# 规避误命中的排除模式: 命中金融词但整体是明确非金融语境时放行。
# 目前留空位 — 宁可多校验(Verifier 保守放过), 不漏校验。
_NON_FINANCE_PATTERNS: tuple[re.Pattern[str], ...] = ()

# 编译一次的金融词正则 (长词优先, 独立匹配)
_FINANCE_RE = re.compile(
    "|".join(sorted(_FINANCE_TERMS, key=len, reverse=True)), re.IGNORECASE
)


def is_finance_question(text: str) -> bool:
    """规则判断问题是否属于金融/财务域。

    - 纯规则, 无 LLM 调用, 微秒级, 可放在每个请求的分发路径上。
    - 保守策略: 检测不到就不启用金融校验(维持原行为), 绝不因检测而拒答。
    """
    if not text:
        return False
    if any(p.search(text) for p in _NON_FINANCE_PATTERNS):
        return False
    return bool(_FINANCE_RE.search(text))
