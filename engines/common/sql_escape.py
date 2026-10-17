"""LanceDB 过滤表达式（DataFusion SQL）的字符串字面量转义。

LanceDB 的 `.where()` 接收的是 SQL 过滤表达式，不支持参数化绑定，
因此所有外部可控的值都必须先经过本模块转义再拼进表达式。

**为什么不能用 f-string 直接拼**（本项目真实修过的漏洞）：
    .where(f"user_id = '{user_id}'")
传入 `x' OR '1'='1` 后表达式变成 `user_id = 'x' OR '1'='1'`，
过滤条件被闭合 → 返回**所有用户**的记忆。在记忆层这等于跨用户数据泄露。

**转义规则**：SQL 字符串字面量内，单引号写成两个单引号（`''`）。
这是 SQL 标准做法，对普通标识符安全；对用户输入而言，
配合上游的长度截断与字符白名单校验已足够（记忆层的 user_id 来自
已鉴权的 Principal.key_id，不接受任意客户端字符串）。
"""

# 控制字符在 DataFusion SQL 字符串字面量中无合法转义形式，
# 保留会让表达式解析行为不确定 —— 直接剔除而不是尝试编码。
_CONTROL_CHARS = str.maketrans("", "", "\x00\n\r\t\x0b\x0c")


def escape_sql_literal(value: object) -> str:
    """把任意值转成可安全嵌入 SQL 字符串字面量的内容（不含外层引号）。

    >>> escape_sql_literal("O'Brien")
    "O''Brien"
    >>> escape_sql_literal("a\\x00b")
    'ab'
    """
    return str(value).translate(_CONTROL_CHARS).replace("'", "''")


def sql_string_literal(value: object) -> str:
    """转义并补上外层单引号，直接用于拼接 where 表达式。

    >>> sql_string_literal("admin")
    "'admin'"
    """
    return "'" + escape_sql_literal(value) + "'"


def sql_string_list(values) -> str:
    """把一组值转成逗号分隔的字面量列表，用于 `IN (...)`。

    空列表返回 `''`（非法但语法完整的单字面量）而不是空串，
    避免生成 `IN ()` 这种语法错误表达式。
    """
    return ", ".join(sql_string_literal(v) for v in values) if values else "''"
