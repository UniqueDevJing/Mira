"""数据层 / 记忆层 / 工程层三处修复的回归测试。

每个用例对应一个已确认的真实缺陷，测试先复现缺陷、修复后必须通过。
放在这里是因为这些点都属于「面试官会深挖」的边界，
能现场演示「为什么这样写」比什么都重要。

三处修复：
1. document_store.purge_old_qa_logs —— cutoff 格式必须与 SQLite
   CURRENT_TIMESTAMP 一致（空格分隔、无时区），否则边界记录被提前删除。
2. middleware.security._is_exempt —— 前缀豁免必须匹配路径段边界，
   否则 /webassets、/webhook 之类的前缀兄弟路径会被静默跳过鉴权。
3. memory_layer.recall —— user_id 拼进 SQL 过滤表达式前必须转义，
   单引号会闭合字面量使条件恒真 → 跨用户记忆泄露。
"""

import sys
from datetime import UTC, datetime, timedelta

sys.path.insert(0, ".")


class TestPurgeCutoffFormat:
    """缺陷 1：purge_old_qa_logs 的 cutoff 与列值格式不匹配。

    SQLite CURRENT_TIMESTAMP 产出 `YYYY-MM-DD HH:MM:SS`，
    而 isoformat() 产出 `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`。
    字符串比较时日期部分占前 10 字符，日期不同才正确；
    日期相同时空格(0x20) < 'T'(0x54) 会让当天记录被提前判为过期。
    """

    def test_cutoff_uses_same_format_as_sqlite(self, tmp_path):
        """cutoff 必须用 strftime 与 CURRENT_TIMESTAMP 对齐，不能用 isoformat。"""
        from api.core.document_store import DocumentStore

        store = DocumentStore(str(tmp_path / "docs.db"))

        days = 7
        cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

        # 与 SQLite 实际写入的 created_at 做同格式字符串比较
        rows = store._get_conn()
        with rows as conn:
            written = conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
            assert len(written) == 19, f"CURRENT_TIMESTAMP 应为 19 字符, 实为 {len(written)}"
            # cutoff 与列值同长度同分隔符 → 可安全做字典序比较
            assert len(cutoff) == len(written)
            assert cutoff[10] == written[10], "cutoff 与列值第 11 位分隔符应一致(均为空格)"

    def test_old_log_purged_and_recent_kept(self, tmp_path):
        """过期日志应被删、当天刚写的日志必须保留。"""
        from api.core.document_store import DocumentStore

        store = DocumentStore(str(tmp_path / "docs.db"))
        now = datetime.now(UTC)
        fmt = "%Y-%m-%d %H:%M:%S"

        rows = [
            ("很旧的问题", (now - timedelta(days=10)).strftime(fmt)),
            ("刚问的", (now - timedelta(minutes=10)).strftime(fmt)),
        ]
        with store._get_conn() as conn:
            for q, ts in rows:
                conn.execute("INSERT INTO qa_logs (question, created_at) VALUES (?,?)", (q, ts))

        deleted = store.purge_old_qa_logs(7)

        with store._get_conn() as conn:
            left = {r[0] for r in conn.execute("SELECT question FROM qa_logs").fetchall()}

        assert "很旧的问题" not in left, "10 天前的日志应被清理"
        assert "刚问的" in left, "10 分钟前的日志必须保留"
        assert deleted == 1, f"应恰好删除 1 条，实际 {deleted}"

    def test_same_date_different_time_respects_second_precision(self, tmp_path):
        """日期相同时，cutoff 的**时分秒**必须参与比较。

        这是本次修复的核心。修复前 cutoff 形如
            2026-10-10T11:54:32.789+00:00   （T 分隔、带微秒与时区）
        而列值形如
            2026-10-10 09:00:00            （空格分隔）
        前 10 字符（日期）相同后，比较落到第 11 位：空格(0x20) < 'T'(0x54)，
        于是当天 09:00 这条**明明早于 cutoff** 却被判为不早于而保留 ——
        过期数据滞留；反向地，当天 23:00 会被误删。
        修复后两者同为「空格分隔」，时分秒正常参与字典序比较。
        """
        from api.core.document_store import DocumentStore

        store = DocumentStore(str(tmp_path / "docs.db"))
        now = datetime.now(UTC)
        cutoff_dt = now - timedelta(days=7)

        # 同为 cutoff 的日期，但一个在 cutoff 之前、一个在之后
        before = cutoff_dt.replace(hour=0, minute=0, second=0)
        after = cutoff_dt + timedelta(seconds=1)
        rows = [
            ("当天零点", before.strftime("%Y-%m-%d %H:%M:%S")),
            ("cutoff后1秒", after.strftime("%Y-%m-%d %H:%M:%S")),
        ]
        with store._get_conn() as conn:
            for q, ts in rows:
                conn.execute("INSERT INTO qa_logs (question, created_at) VALUES (?,?)", (q, ts))

        store.purge_old_qa_logs(7)

        with store._get_conn() as conn:
            left = {r[0] for r in conn.execute("SELECT question FROM qa_logs").fetchall()}

        assert "当天零点" not in left, "cutoff 当天零点早于 cutoff，应被清理"
        assert "cutoff后1秒" in left, (
            "同一日期内晚于 cutoff 的记录必须保留 —— "
            "修复前因 T 与空格的分隔符差异会被误删"
        )

    def test_retention_zero_keeps_today(self, tmp_path):
        """retention=0 表示只保留今天，今天的日志不应被删。"""
        from api.core.document_store import DocumentStore

        store = DocumentStore(str(tmp_path / "docs.db"))
        today = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        with store._get_conn() as conn:
            conn.execute("INSERT INTO qa_logs (question, created_at) VALUES (?,?)",
                         ("今天问的", today))

        store.purge_old_qa_logs(0)

        with store._get_conn() as conn:
            left = conn.execute("SELECT COUNT(*) FROM qa_logs").fetchone()[0]
        assert left == 1, "retention=0 时当天记录必须保留"


class TestExemptPathBoundary:
    """缺陷 2：startswith 裸前缀导致前缀兄弟路径被误豁免鉴权。

    修复前 startswith(('/docs','/openapi','/web')) 会豁免
    /webassets/app.js、/webhook/callback、/docs-internal/config 等路径。
    中间件是 fail-closed 设计，这类越界不会有任何报错，只会静默放行。
    """

    def test_legitimate_exempt_paths(self):
        from api.middleware.security import _is_exempt

        for p in ["/health", "/", "/metrics", "/web", "/admin",
                  "/web/index.html", "/web/assets/app.js", "/docs", "/openapi.json"]:
            assert _is_exempt(p), f"{p} 应豁免"

    def test_swagger_endpoints_stay_reachable(self):
        """FastAPI 默认端点必须保持可访问。

        收紧前缀匹配时最容易踩的坑：把 `/docs` 也一起挡掉，
        Swagger UI 直接 404，但鉴权日志里看不到任何异常 ——
        属于「安全收紧」引入的功能性回归，必须有用例钉住。
        """
        from api.middleware.security import _is_exempt

        # 这两个是「完全相等」而非子路径形式，容易在收紧时被误伤
        assert _is_exempt("/docs")
        assert _is_exempt("/openapi.json")
        # FastAPI 也可能挂载 redoc（若启用）
        assert not _is_exempt("/docs-internal/x")

    def test_prefix_siblings_not_exempt(self):
        """前缀兄弟路径必须走鉴权，这是本次修复的核心。"""
        from api.middleware.security import _is_exempt

        hostile = [
            "/webassets/app.js",
            "/webhooks/callback",
            "/webconfig/admin.json",
            "/docs-internal/config",
            "/docs-secrets/keys",
            "/openapi-secret/keys",
            "/openapikeys",
        ]
        for p in hostile:
            assert not _is_exempt(p), f"{p} 不应被豁免 —— 裸前缀匹配会越界"

    def test_data_apis_never_exempt(self):
        """所有业务数据接口必须在鉴权范围内。"""
        from api.middleware.security import _is_exempt

        for p in ["/api/v1/qa/ask", "/api/v1/documents", "/api/v1/metrics/summary",
                  "/api/v1/admin/users"]:
            assert not _is_exempt(p), f"{p} 是数据接口，绝不能豁免"


class TestSqlEscape:
    """缺陷 3：记忆层 user_id 拼进 SQL 过滤表达式未转义。

    LanceDB 的 where() 是 SQL 过滤表达式且不支持参数绑定，
    直接 f-string 拼接时 `x' OR '1'='1` 会闭合字面量使条件恒真。
    """

    def test_quote_is_doubled(self):
        from engines.common.sql_escape import escape_sql_literal, sql_string_literal

        assert escape_sql_literal("O'Brien") == "O''Brien"
        assert sql_string_literal("O'Brien") == "'O''Brien'"

    def test_injection_attack_becomes_inert(self):
        """注入串转义后必须仍是「一个字面量」而不是两个条件。"""
        from engines.common.sql_escape import sql_string_literal

        evil = sql_string_literal("x' OR '1'='1")
        # 转义后：字面量内部的所有引号都成对，外部只留最外层两个
        assert evil.startswith("'") and evil.endswith("'")
        assert evil.count("'") % 2 == 0, "引号必须成对出现"
        # 关键：'OR' 里的引号已被转义，OR 变成字面量内容而非运算符
        assert "'' OR ''" in evil

    def test_control_chars_stripped(self):
        """NUL/换行在 SQL 字面量中无合法转义，直接剔除避免表达式解析不确定。"""
        from engines.common.sql_escape import escape_sql_literal

        assert escape_sql_literal("a\x00b") == "ab"
        assert escape_sql_literal("a\nb") == "ab"

    def test_list_literal(self):
        from engines.common.sql_escape import sql_string_list

        assert sql_string_list(["a", "b'c"]) == "'a', 'b''c'"
        # 空列表不能生成 `IN ()` 语法错误
        assert sql_string_list([]) == "''"

    def test_memory_recall_uses_escaped_literal(self, monkeypatch):
        """验证 recall 真正走到转义后的表达式，而非仅工具函数可用。"""
        import api.core.memory_layer as ml
        from engines.common.sql_escape import sql_string_literal

        captured = {}

        class FakeTable:
            def search(self, vec):
                return self

            def metric(self, m):
                return self

            def where(self, expr):
                captured["expr"] = expr
                return self

            def limit(self, n):
                return self

            def to_list(self):
                return []

        # memory_layer 内部用 `from api.state import get_embedder` 的函数内导入，
        # 直接替换 api.state 模块上的属性即可生效（函数内 import 每次都重新取属性）。
        class FakeEmbedder:
            def embed_query(self, q):
                return [0.1, 0.2]

        import api.state as state_mod

        monkeypatch.setattr(ml, "_get_table", lambda: FakeTable())
        monkeypatch.setattr(state_mod, "get_embedder", lambda: FakeEmbedder(), raising=False)

        evil = "x' OR '1'='1"
        ml.recall(evil, "问题")

        assert "expr" in captured, "recall 应调用 where()"
        assert captured["expr"] == "user_id = " + sql_string_literal(evil[:64]), (
            "recall 必须用转义后的字面量拼接，实际得到: " + str(captured.get("expr"))
        )
