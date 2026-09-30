"""金融校验编排（Verifier）— P0~P2 能力的组合裁判。

为什么需要它：
  P0~P2 提供了原子能力（溯源 / 交叉 / 时效），但调用方需要的是**一个裁决**：
  这条答案能不能放行？Verifier 把三类证据组合成三级裁决：

    fail  — 数字无来源支撑（臆造嫌疑）或跨源矛盾 → 拒答。
            金融场景下，给错数字比拒答严重得多。
    warn  — 答案本身可溯源、无矛盾，但证据来源过期/未声明时效 → 放行 + 附提示。
    pass  — 全部通过 → 放行。

可复算豁免（computed exemption）：
  答案数字若能由来源数字经常见运算（和/差/积/商/增长率）复算得出，
  不判 unsupported —— LLM 做正确的聚合/换算是合法的，不应误拒。
  豁免记录为 info 级 finding（"1.2亿 = 3.5亿 - 2.3亿 可复算"），保留可审计性。

纯标准库、确定性、不调 LLM —— 裁决必须可复现。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .crosscheck import Source, check_freshness as _check_freshness, cross_check as _cross_check
from .numeric import (
    NumberToken,
    extract_numbers,
    find_unsupported,
    format_token,
)

_FAIL_KINDS = {"unsupported", "conflict"}
_WARN_KINDS = {"stale", "unknown"}
_INFO_KINDS = {"computed"}


@dataclass(frozen=True)
class FinanceFinding:
    """一条校验发现。

    kind:     unsupported / conflict / intra_conflict / stale / unknown / computed
    severity: fail（阻断）/ warn（提示）/ info（审计记录）
    """

    kind: str
    severity: str
    message: str


@dataclass(frozen=True)
class FinanceVerdict:
    """校验裁决。verdict: pass / warn / fail。"""

    verdict: str
    findings: tuple[FinanceFinding, ...]

    def summary(self) -> str:
        if self.verdict == "pass" and not self.findings:
            return "校验通过：数值均可溯源，无跨源矛盾。"
        parts = [f.message for f in self.findings]
        head = {"pass": "校验通过", "warn": "校验通过（附提示）", "fail": "校验未通过"}[self.verdict]
        return head + "：" + "；".join(parts)

    @property
    def failed(self) -> bool:
        return self.verdict == "fail"


# ───────────────────── 可复算豁免 ─────────────────────


@dataclass(frozen=True)
class _Recompute:
    op: str
    x: float
    y: float


def _recompute_support(
    token: NumberToken, pool: list[NumberToken], rel_tol: float = 0.005
) -> _Recompute | None:
    """尝试用来源数字两两复算出 token.value。命中返回运算描述，否则 None。

    覆盖聚合问答中最常见的运算：和 / 差 / 积 / 商 / 增长率 / 占比。
    - 和差积商仅对**同单位**数字组合尝试（跨单位需换算，无汇率不做）
    - 增长率/占比的输入是绝对值、输出是 PERCENT —— 对 PERCENT token
      允许全 pool（任意单位）做 (x-y)/y 与 x/y 形态的复算
    """
    # 增长率 / 占比: PERCENT token 可由任意单位绝对值复算
    if token.unit == "PERCENT":
        vals = [p.value for p in pool if p.value != 0]
        for i in range(len(vals)):
            for j in range(len(vals)):
                if i == j:
                    continue
                x, y = vals[i], vals[j]
                for op, val in (("增长率", (x - y) / y), ("占比", x / y)):
                    if abs(val - token.value) <= rel_tol * max(abs(val), 1e-12):
                        return _Recompute(op=op, x=x, y=y)

    # 和差积商: 同单位数字两两组合
    cands = [p for p in pool if p.unit == token.unit and p.value != 0]
    for i in range(len(cands)):
        for j in range(len(cands)):
            if i == j:
                continue
            x, y = cands[i].value, cands[j].value
            for op, val in (("和", x + y), ("差", x - y), ("积", x * y), ("商", x / y)):
                if val == token.value or (
                    val != 0 and abs(val - token.value) / abs(val) <= rel_tol
                ):
                    return _Recompute(op=op, x=x, y=y)
    return None


# ───────────────────── 编排 ─────────────────────


def verify_finance_answer(
    answer: str,
    sources: list[Source],
    *,
    rel_tol: float = 0.005,
    max_age: timedelta = timedelta(days=365),
    now: datetime | None = None,
    check_unsupported: bool = True,
    check_conflicts: bool = True,
    check_freshness: bool = True,
) -> FinanceVerdict:
    """对一条金融答案做三重校验，产出 pass / warn / fail 裁决。

    参数：
      answer  —— 待校验的答案文本
      sources —— 证据来源列表（Source 带 name/text/as_of）

    冲突分级（工程决策）：
      跨源矛盾 → fail：两个独立来源互斥，必有其一错误，拒答防错。
      同源"矛盾" → warn：财报文本常见分部/分期数据（"A业务营收2亿，B业务
      营收1.5亿"），纯文本难以区分分部与自相矛盾 —— 宁提示勿误拒。
    """
    findings: list[FinanceFinding] = []

    # 1) 溯源（含可复算豁免）
    if check_unsupported:
        pool = [t for s in sources for t in extract_numbers(s.text)]
        unsupported = find_unsupported(answer, [s.text for s in sources], rel_tol=rel_tol)
        hard: list[NumberToken] = []
        for t in unsupported:
            rec = _recompute_support(t, pool, rel_tol=rel_tol)
            if rec is not None:
                findings.append(
                    FinanceFinding(
                        kind="computed",
                        severity="info",
                        message=f"{format_token(t)} 可由来源数字复算（{rec.x:g} {rec.op} {rec.y:g}）",
                    )
                )
            else:
                hard.append(t)
        if hard:
            listed = "、".join(format_token(t) for t in hard[:5])
            findings.append(
                FinanceFinding(
                    kind="unsupported", severity="fail",
                    message=f"数字无来源支撑: {listed}",
                )
            )

    # 2) 交叉：跨文件矛盾 fail，同文件跨 chunk / 单来源自相矛盾 warn。
    #    来源身份要求 chunk 级唯一（映射层保证）；同文件判定用 group 而非 name——
    #    name 相等即跳过的旧规则会把"同文件多 chunk 互斥"整体漏检（质检 🔴-1）。
    if check_conflicts:
        group_of = {s.name: s.group for s in sources}
        if len(sources) >= 2:
            for c in _cross_check(sources, rel_tol=rel_tol, include_intra=False):
                ga, gb = group_of.get(c.a_source), group_of.get(c.b_source)
                same_file = ga is not None and gb is not None and ga == gb
                if same_file:
                    findings.append(
                        FinanceFinding(
                            kind="intra_conflict", severity="warn",
                            message=f"同文件不同片段矛盾: {c.describe()}（若为分部/分期数据可忽略）",
                        )
                    )
                else:
                    findings.append(
                        FinanceFinding(kind="conflict", severity="fail", message=c.describe())
                    )
        for s in sources:
            for c in _cross_check([s], rel_tol=rel_tol, include_intra=True):
                findings.append(
                    FinanceFinding(
                        kind="intra_conflict", severity="warn",
                        message=f"来源[{s.name}] 内部数据疑似矛盾: {c.describe()}（若为分部/分期数据可忽略）",
                    )
                )

    # 3) 时效（只降级到 warn，不单独 fail）
    if check_freshness and sources:
        for f in _check_freshness(sources, now=now, max_age=max_age):
            if f.reason == "stale":
                findings.append(
                    FinanceFinding(
                        kind="stale", severity="warn",
                        message=f"来源[{f.source}] 数据已过期（{f.age_days:.0f} 天前，上限 {f.max_age_days:.0f} 天）",
                    )
                )
            else:
                findings.append(
                    FinanceFinding(
                        kind="unknown", severity="warn",
                        message=f"来源[{f.source}] 未声明数据时效",
                    )
                )

    if any(f.severity == "fail" for f in findings):
        verdict = "fail"
    elif any(f.severity == "warn" for f in findings):
        verdict = "warn"
    else:
        verdict = "pass"
    return FinanceVerdict(verdict=verdict, findings=tuple(findings))
