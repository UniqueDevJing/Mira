"""金融领域能力层：数值抽取/归一化、校验器、安全计算。

与 `engines/` 其他子包一致，本包为**纯能力层**（不依赖 api/），
可被 api/core 的护栏与 Agent 复用，也可独立测试。
"""

from engines.finance.crosscheck import (
    Conflict,
    Source,
    StaleFinding,
    check_freshness,
    cross_check,
    same_metric,
)
from engines.finance.intent import is_finance_question
from engines.finance.numeric import (
    NumberToken,
    extract_numbers,
    find_unsupported,
    format_token,
    match_numbers,
)
from engines.finance.verifier import (
    FinanceFinding,
    FinanceVerdict,
    verify_finance_answer,
)

__all__ = [
    "Conflict",
    "FinanceFinding",
    "FinanceVerdict",
    "NumberToken",
    "Source",
    "StaleFinding",
    "check_freshness",
    "cross_check",
    "extract_numbers",
    "find_unsupported",
    "format_token",
    "is_finance_question",
    "match_numbers",
    "same_metric",
    "verify_finance_answer",
]
