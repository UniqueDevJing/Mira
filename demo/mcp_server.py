# -*- coding: utf-8 -*-
"""金融校验引擎 MCP Server —— 把校验能力按 MCP 标准对外输出。

为什么是 MCP：互操作。内部工具调用走函数（零开销），对外输出走协议——
任何支持 MCP 的客户端（Claude Desktop / Claude Code / 其他 Agent）都可以把
这道数值防线即插即用地挂进自己的链路，而不需要改一行接入代码。

传输（双通道，同一套工具）：
  stdio:            python demo/mcp_server.py            # 本地客户端（Claude Desktop 配置）
  streamable-http:  挂载在演示服务 /mcp 路径（远程可调）

工具（3 个）：
  verify_numbers     回答 + 来源 -> 三级裁决 + findings + 数字归一化明细
  extract_numbers    文本 -> 数值断言明细（归一化单位/数量级）
  run_finance_gate   一键重跑 51 条对抗样本门禁 -> 拦截率/误拒率/时效命中

资源（1 个）：
  corpus://documents  Agent 可引用的语料清单（自定义提问范围说明）

运行：
  stdio 模式:  python demo/mcp_server.py
  HTTP 模式:   由 demo/finance_demo_app.py 在启动时挂载（app.mount("/mcp", ...)）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP

from engines.finance import Source, verify_finance_answer
from engines.finance.numeric import extract_numbers as _extract_numbers, format_token as _format_token

mcp = FastMCP(
    "finance-verifier",
    streamable_http_path="/",   # 挂载到主应用 /mcp 后, 对外端点即 /mcp（避免 /mcp/mcp 双层）
    instructions=(
        "金融数值校验工具：校验 LLM 回答中的数字是否有来源支撑。"
        "适用场景：你的 Agent 生成含数字断言的回答后，把回答与检索到的原文"
        "一起传给 verify_numbers，fail 级裁决意味着存在无来源支撑/矛盾的数字，"
        "应拒答或提示用户。"
    ),
)


@mcp.tool()
def verify_numbers(answer: str, sources: list[dict]) -> dict:
    """校验一条回答中的数字断言是否可信。

    Args:
        answer: 待校验的 LLM 回答文本（应含数字断言）
        sources: 证据来源列表，每项 {"name": 文档名, "text": 原文, "as_of": "报告期 ISO 日期或 null"}

    Returns:
        verdict (pass/warn/fail)、summary、findings 明细、
        tokens（回答中每个数字的归一化明细）。
        fail = 存在无来源支撑/矛盾的数字，建议拒答；warn = 附数据提示后放行。
    """
    srcs = [Source(x.get("name", "来源"), x.get("text", ""), x.get("as_of"))
            for x in (sources or [])]
    v = verify_finance_answer(answer, srcs)
    return {
        "verdict": v.verdict,
        "summary": v.summary(),
        "findings": [{"kind": f.kind, "severity": f.severity, "message": f.message}
                     for f in v.findings],
        "tokens": [{"raw": t.raw, "normalized": _format_token(t),
                    "value": t.value, "unit": t.unit}
                   for t in _extract_numbers(answer)],
        "advice": ("fail: 拒答或要求重新生成" if v.verdict == "fail"
                   else "warn: 放行但向用户展示数据提示" if v.verdict == "warn"
                   else "pass: 直接放行"),
    }


@mcp.tool()
def extract_numbers(text: str) -> list[dict]:
    """从中文文本中抽取数值断言并归一化（万/亿数量级折算、币种/百分比语义单位）。

    Args:
        text: 含数字的中文文本，如 "2024年净利润1.2亿元"

    Returns:
        每个数字的 raw（原文片段）/ normalized（归一化展示）/ value / unit
        （CNY/USD/PERCENT/PP/COUNT…，单位不同不可混比）。
    """
    return [{"raw": t.raw, "normalized": _format_token(t),
             "value": t.value, "unit": t.unit}
            for t in _extract_numbers(text)]


@mcp.tool()
def run_finance_gate() -> dict:
    """一键重跑 51 条确定性对抗样本门禁（与 CI 同口径）。

    Returns:
        attack_intercept（攻击拦截率）、false_reject（误拒率）、warn_hit（时效命中率）、
        overall_hit_rate 及 12 类攻击的逐类命中明细。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mira_gen_adversarial", ROOT / "scripts" / "gen_finance_adversarial.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    samples = mod.build_samples()

    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 9, 10, tzinfo=timezone.utc)

    def _hit(exp, got):
        # 与 scripts/eval_finance_adversarial.py / /api/run-all 完全一致的命中判定
        if exp == "fail":
            return got == "fail"
        if exp == "fail_or_warn":
            return got in ("fail", "warn")
        if exp == "warn":
            return got == "warn"
        if exp == "pass_nowarn":
            return got == "pass"
        return got in ("pass", "warn")  # pass

    hard = hard_n = strict = strict_n = stale = stale_n = hit = 0
    for s in samples:
        srcs = [Source(x["name"], x["text"], x["as_of"]) for x in s["sources"]]
        got = verify_finance_answer(s["answer"], srcs, now=now,
                                    max_age=timedelta(days=365)).verdict
        exp, ok = s["expected"], _hit(s["expected"], got)
        hit += 1 if ok else 0
        if exp == "fail":
            hard_n += 1
            hard += 1 if got == "fail" else 0
        elif exp == "pass_nowarn":
            strict_n += 1
            strict += 1 if got == "pass" else 0
        elif exp == "warn":
            stale_n += 1
            stale += 1 if got == "warn" else 0
    n = len(samples)
    return {
        "total": n,
        "attack_intercept": round(hard / hard_n, 4) if hard_n else 0,
        "false_reject": round(1 - strict / strict_n, 4) if strict_n else 0,
        "warn_hit": round(stale / stale_n, 4) if stale_n else 0,
        "overall_hit_rate": round(hit / n, 4),
    }


@mcp.resource("corpus://documents")
def corpus_documents() -> str:
    """Agent 可引用的语料清单（演示环境自定义提问的范围）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mira_pipeline", ROOT / "demo" / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lines = [f"- {c['name']}（报告期 {c['as_of'] or '未声明'}）" for c in mod.CORPUS]
    return "金融语料清单（自定义提问范围）:\n" + "\n".join(lines)


if __name__ == "__main__":
    mcp.run()  # stdio 传输（本地客户端接入）
