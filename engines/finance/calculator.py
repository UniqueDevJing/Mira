"""金融计算校验 — 安全表达式求值 + 财务公式库。

为什么需要它：
  让 LLM 做算术是不可靠的（它是语言模型，不是计算器）。金融场景下
  "毛利率应该是 32.5% 而不是 41%" 这类错误必须由**独立计算**来发现，
  而不是信任模型的自述。

安全红线（重要）：
  **绝对禁止使用 `eval` / `exec`** —— 那是远程代码执行(RCE)漏洞。
  这里用 **AST 白名单**：只放行算术运算节点，凡是函数调用、属性访问、
  下标、推导式、lambda、导入一律拒绝。即使表达式来自被污染的文档内容，
  也无法逃逸。

财务公式一律用**显式 Python 函数**实现（不走表达式求值），可读可测。
"""

from __future__ import annotations

import ast

# ───────────────────── AST 白名单 ─────────────────────

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,   # 顶层
    ast.BinOp,        # 二元运算
    ast.UnaryOp,      # 一元运算
    ast.Constant,     # 字面量（数字）
    ast.Name,         # 变量名（仅限显式传入的变量）
    ast.Load,         # Name 的读取上下文（ast.Name.ctx，非求值语义）
    # 运算符
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
)

_MAX_EXPR_LEN = 500
_MAX_POW_EXPONENT = 64   # 防 9**9**9 这类幂爆炸


class UnsafeExpressionError(ValueError):
    """表达式包含不允许的语法（潜在注入/滥用）。"""


def _validate(node: ast.AST, allowed_names: set[str]) -> None:
    if not isinstance(node, _ALLOWED_NODES):
        raise UnsafeExpressionError(
            f"不允许的语法节点: {type(node).__name__}"
        )
    if isinstance(node, ast.Name) and node.id not in allowed_names:
        raise UnsafeExpressionError(f"未知变量: {node.id}")
    if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
        raise UnsafeExpressionError("只允许数字字面量")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        exp = node.right
        # 幂指数必须是有界常量，防计算爆炸（DoS）
        if not (isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float))
                and abs(exp.value) <= _MAX_POW_EXPONENT):
            raise UnsafeExpressionError(f"幂指数必须是绝对值 ≤ {_MAX_POW_EXPONENT} 的常量")
    for child in ast.iter_child_nodes(node):
        _validate(child, allowed_names)


def _eval_node(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return float(variables[node.id])
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, variables)
        return -val if isinstance(node.op, ast.USub) else +val
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("除数为 0")
            return left / right
        if isinstance(op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("除数为 0")
            return left // right
        if isinstance(op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("模数为 0")
            return left % right
        if isinstance(op, ast.Pow):
            return left ** right
    raise UnsafeExpressionError(f"无法求值: {type(node).__name__}")


def safe_eval(expr: str, variables: dict[str, float] | None = None) -> float:
    """安全求值一个纯算术表达式。

    >>> safe_eval("(120 - 80) / 80")
    0.5
    >>> safe_eval("revenue - cost", {"revenue": 100, "cost": 60})
    40.0

    抛 UnsafeExpressionError（含注入语法）、ZeroDivisionError（除零）、
    ValueError（语法错误/超长）。
    """
    if not expr or not expr.strip():
        raise ValueError("表达式为空")
    if len(expr) > _MAX_EXPR_LEN:
        raise ValueError(f"表达式过长（>{_MAX_EXPR_LEN} 字符）")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e.msg}") from e

    variables = variables or {}
    # 变量名收敛：只允许调用方显式传入的键，且必须是合法标识符
    allowed_names = {k for k in variables if k.isidentifier()}
    _validate(tree, allowed_names)
    return float(_eval_node(tree, variables))


# ───────────────────── 财务公式库 ─────────────────────

def growth_rate(current: float, previous: float) -> float:
    """同比增长率 = (本期 - 上期) / |上期|。返回小数（0.25 = +25%）。"""
    if previous == 0:
        raise ZeroDivisionError("基期为 0，同比增长率无定义")
    return (current - previous) / abs(previous)


yoy_rate = growth_rate   # 同比
qoq_rate = growth_rate   # 环比（同一公式，语义由调用方区分）


def cagr(begin: float, end: float, years: float) -> float:
    """复合年均增长率 CAGR = (end/begin)^(1/years) - 1。"""
    if begin <= 0 or end < 0:
        raise ValueError("CAGR 要求期初>0 且期末>=0")
    if years <= 0:
        raise ValueError("年数必须为正")
    return (end / begin) ** (1.0 / years) - 1.0


def gross_margin(revenue: float, cost: float) -> float:
    """毛利率 = (营收 - 成本) / 营收。"""
    if revenue == 0:
        raise ZeroDivisionError("营收为 0，毛利率无定义")
    return (revenue - cost) / revenue


def net_margin(net_profit: float, revenue: float) -> float:
    """净利率 = 净利润 / 营收。"""
    if revenue == 0:
        raise ZeroDivisionError("营收为 0，净利率无定义")
    return net_profit / revenue


def roe(net_profit: float, equity: float) -> float:
    """净资产收益率 ROE = 净利润 / 净资产。"""
    if equity == 0:
        raise ZeroDivisionError("净资产为 0，ROE 无定义")
    return net_profit / equity


def roa(net_profit: float, total_assets: float) -> float:
    """总资产收益率 ROA = 净利润 / 总资产。"""
    if total_assets == 0:
        raise ZeroDivisionError("总资产为 0，ROA 无定义")
    return net_profit / total_assets


def debt_ratio(total_liability: float, total_assets: float) -> float:
    """资产负债率 = 总负债 / 总资产。"""
    if total_assets == 0:
        raise ZeroDivisionError("总资产为 0，资产负债率无定义")
    return total_liability / total_assets


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    """流动比率 = 流动资产 / 流动负债。"""
    if current_liabilities == 0:
        raise ZeroDivisionError("流动负债为 0，流动比率无定义")
    return current_assets / current_liabilities


def expense_ratio(expense: float, revenue: float) -> float:
    """费用率 = 费用 / 营收。"""
    if revenue == 0:
        raise ZeroDivisionError("营收为 0，费用率无定义")
    return expense / revenue


# 公式注册表：名称 → (函数, 中文说明, 参数名)
FORMULAS: dict[str, tuple] = {
    "growth_rate": (growth_rate, "同比增长率", ("current", "previous")),
    "cagr": (cagr, "复合年均增长率", ("begin", "end", "years")),
    "gross_margin": (gross_margin, "毛利率", ("revenue", "cost")),
    "net_margin": (net_margin, "净利率", ("net_profit", "revenue")),
    "roe": (roe, "净资产收益率", ("net_profit", "equity")),
    "roa": (roa, "总资产收益率", ("net_profit", "total_assets")),
    "debt_ratio": (debt_ratio, "资产负债率", ("total_liability", "total_assets")),
    "current_ratio": (current_ratio, "流动比率", ("current_assets", "current_liabilities")),
    "expense_ratio": (expense_ratio, "费用率", ("expense", "revenue")),
}


def compute(formula: str, **kwargs) -> float:
    """按名称调用财务公式（供校验器使用，避免动态取值）。"""
    if formula not in FORMULAS:
        raise KeyError(f"未知公式: {formula}")
    fn = FORMULAS[formula][0]
    return float(fn(**kwargs))
