"""
金融可信问答·在线验证环境 —— 零嵌入模型、零语料、零 LLM 依赖，可独立部署。
定位：engines/finance 生产级校验引擎的独立运行实例（引擎本体已集成在 Mira 主链路，
本服务把同一套引擎以可交互方式对外提供——不是玩具副本，是同一代码的另一个入口）。

为什么它能独立成站：engines/finance 全部是纯标准库规则引擎，
不依赖 RAG/向量库/嵌入服务，因此可以在任何一台机器上独立运行。

端点（REST）：
  GET  /               演示页
  GET  /api/meta       元信息（评测基准时间 / 样本数 / 类别）
  GET  /api/samples    51 条确定性对抗样本（含答案与来源）
  POST /api/verify     实时校验：answer + sources -> verdict + findings + 数字 token 明细
  POST /api/run-all    一键重跑 51 条门禁 -> 汇总指标 + 逐条明细

确定性说明：评测基准时间固定为 EVAL_NOW（2026-09-10 UTC），
与 scripts/eval_finance_adversarial.py 的门禁口径一致，保证任何人重跑结果逐字节一致。

运行（Mira 仓库根下）：
  uvicorn demo.finance_demo_app:app --host 127.0.0.1 --port 8001
"""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from engines.finance import Source, verify_finance_answer
from engines.finance.numeric import extract_numbers, format_token


def _load_script(name: str, rel: str):
    """scripts/ 目录不是包（无 __init__.py），按文件路径加载脚本模块。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gen = _load_script("mira_gen_adversarial", "scripts/gen_finance_adversarial.py")
SAMPLES = _gen.build_samples()  # 51 条确定性样本（参数表展开，无随机）

_pipe = _load_script("mira_pipeline", "demo/pipeline.py")

# 与门禁评测完全一致的固定基准时间：保证"现场演示 = CI 门禁"同口径
EVAL_NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)
MAX_AGE = timedelta(days=365)

# MCP Server 挂载（streamable-http 传输）：校验引擎按 MCP 标准对外输出，
# 任何 MCP 客户端可连接 http://<host>/mcp 即插即用。stdio 入口见 demo/mcp_server.py。
# 注意：Starlette 不执行被挂载子应用的 lifespan，streamable-http 的会话管理器
# 必须挂进主应用 lifespan（否则请求时报 "Task group is not initialized"）。
import contextlib
from contextlib import asynccontextmanager
from demo.mcp_server import mcp as _finance_mcp


@asynccontextmanager
async def _mcp_lifespan(_app):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(_finance_mcp.session_manager.run())
        yield


app = FastAPI(title="Mira Financial Trust Service - 金融可信问答·在线验证环境",
              version="1.0.0", lifespan=_mcp_lifespan)

app.mount("/mcp", _finance_mcp.streamable_http_app())
STATIC = ROOT / "demo" / "static"


class VerifyReq(BaseModel):
    answer: str
    sources: list[dict] = []       # [{name, text, as_of?}]
    now: str | None = None         # ISO 时间；缺省用评测基准时间（确定性）


@app.get("/")
def index():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/meta")
def meta():
    return {
        "eval_now": EVAL_NOW.isoformat(),
        "max_age_days": MAX_AGE.days,
        "samples": len(SAMPLES),
        "categories": sorted({s["category"] for s in SAMPLES}),
        "attack_count": sum(1 for s in SAMPLES if s["attack"]),
        "benign_count": sum(1 for s in SAMPLES if not s["attack"]),
    }


@app.get("/api/samples")
def samples():
    return {"samples": SAMPLES}


def _hit(expected: str, got: str) -> bool:
    """与 scripts/eval_finance_adversarial.py 完全一致的命中判定。"""
    if expected == "fail":
        return got == "fail"
    if expected == "fail_or_warn":
        return got in ("fail", "warn")
    if expected == "warn":
        return got == "warn"
    if expected == "pass_nowarn":
        return got == "pass"
    return got in ("pass", "warn")  # pass


@app.post("/api/verify")
def verify(req: VerifyReq):
    now = EVAL_NOW
    if req.now:
        try:
            now = datetime.fromisoformat(req.now)
        except ValueError:
            raise HTTPException(status_code=422, detail="now 不是合法的 ISO 时间")
    srcs = [Source(x.get("name", "来源"), x.get("text", ""), x.get("as_of"))
            for x in req.sources]
    v = verify_finance_answer(req.answer, srcs, now=now, max_age=MAX_AGE)
    return {
        "verdict": v.verdict,
        "summary": v.summary(),
        "findings": [{"kind": f.kind, "severity": f.severity, "message": f.message}
                     for f in v.findings],
        # 数字 token 明细：让访客看见"系统是怎么读数字的"（归一化单位/数量级）
        "tokens": [{"raw": t.raw, "normalized": format_token(t),
                    "value": t.value, "unit": t.unit}
                   for t in extract_numbers(req.answer)],
        "eval_now": now.isoformat(),
    }


@app.post("/api/run-all")
def run_all():
    """一键重跑 51 条门禁——与 CI 的 make eval-finance-adversarial 同口径。"""
    details, by_cat = [], {}
    for s in SAMPLES:
        srcs = [Source(x["name"], x["text"], x["as_of"]) for x in s["sources"]]
        v = verify_finance_answer(s["answer"], srcs, now=EVAL_NOW, max_age=MAX_AGE)
        got = v.verdict
        hit = _hit(s["expected"], got)
        c = by_cat.setdefault(s["category"], {"n": 0, "hit": 0})
        c["n"] += 1
        c["hit"] += 1 if hit else 0
        details.append({
            "id": s["id"], "category": s["category"],
            "attack": s["attack"], "expected": s["expected"], "got": got, "hit": hit,
            "answer": s["answer"],
            "findings": [f.message for f in v.findings],
        })

    n = len(details)
    # 口径与 scripts/eval_finance_adversarial.py 严格一致：
    # attack_intercept 的分母是 expected=fail 的硬攻击样本（attack=True 但 expected=fail_or_warn
    # 的跨期混淆判 warn 也算命中，不计入此指标分母，否则与 CI 门禁数字对不上）
    hard_attacks = [d for d in details if d["expected"] == "fail"]
    strict_benign = [d for d in details if d["expected"] == "pass_nowarn"]
    stale = [d for d in details if d["expected"] == "warn"]
    return {
        "eval_now": EVAL_NOW.isoformat(),
        "total": n,
        "overall_hit_rate": round(sum(1 for d in details if d["hit"]) / n, 4) if n else 0,
        "attack_intercept": round(sum(1 for d in hard_attacks if d["got"] == "fail") / len(hard_attacks), 4) if hard_attacks else 0,
        "attack_n": len(hard_attacks),
        "false_reject": round(sum(1 for d in strict_benign if d["got"] in ("fail", "warn")) / len(strict_benign), 4) if strict_benign else 0,
        "strict_n": len(strict_benign),
        "warn_hit": round(sum(1 for d in stale if d["got"] == "warn") / len(stale), 4) if stale else 0,
        "stale_n": len(stale),
        "by_category": by_cat,
        "details": details,
    }


class PipelineReq(BaseModel):
    question_id: str | None = None
    question: str | None = None    # 自定义自由提问（抽取式生成 + 相关度门槛）
    fault_mode: str = "none"       # none / half / all（确定性幻觉注入档位，仅预设问题生效）


class BatchReq(BaseModel):
    fault_mode: str = "none"


@app.get("/api/pipeline/questions")
def pipeline_questions():
    """完整闭环预设问题（含每题的幻觉类型标注，供前端展示）。"""
    return {"questions": [{"id": q["id"], "q": q["q"],
                           "has_fault": q["fault"] is not None,
                           "fault_type": q["fault"][0] if q["fault"] else None}
                          for q in _pipe.QUESTIONS],
            "demo_now": _pipe.DEMO_NOW.isoformat(),
            "corpus": [{"id": c["id"], "name": c["name"], "as_of": c["as_of"]}
                       for c in _pipe.CORPUS]}


@app.post("/api/pipeline")
def pipeline_run(req: PipelineReq):
    """完整闭环单题演示：检索 → 生成 → 校验 → 响应，逐阶段计时与输入输出。"""
    if req.fault_mode not in ("none", "half", "all"):
        raise HTTPException(status_code=422, detail="fault_mode 必须是 none/half/all")
    try:
        return _pipe.run_pipeline(req.question_id, req.fault_mode)
    except KeyError:
        raise HTTPException(status_code=404, detail="未知 question_id")


@app.post("/api/agent/run")
def agent_run(req: PipelineReq):
    """ReAct 决策循环单题演示：Thought → 工具 → 观察 → 自校正，逐步落轨迹。"""
    if req.fault_mode not in ("none", "half", "all"):
        raise HTTPException(status_code=422, detail="fault_mode 必须是 none/half/all")
    qtext = (req.question or "").strip()
    if qtext:
        if len(qtext) > 200:
            raise HTTPException(status_code=422, detail="问题过长（≤200 字符）")
        return _pipe.run_agent(question_text=qtext, fault_mode=req.fault_mode)
    if not req.question_id:
        raise HTTPException(status_code=422, detail="question_id 与 question 至少提供一个")
    try:
        return _pipe.run_agent(req.question_id, req.fault_mode)
    except KeyError:
        raise HTTPException(status_code=404, detail="未知 question_id")


@app.post("/api/agent/batch")
def agent_batch(req: BatchReq):
    """全部问题过 ReAct 循环，输出自校正拦截统计。"""
    if req.fault_mode not in ("none", "half", "all"):
        raise HTTPException(status_code=422, detail="fault_mode 必须是 none/half/all")
    return _pipe.run_agent_batch(req.fault_mode)


@app.post("/api/pipeline/batch")
def pipeline_batch(req: BatchReq):
    """全部预设问题过完整闭环，输出闭环统计（放行 / 提示 / 拒答分布）。"""
    if req.fault_mode not in ("none", "half", "all"):
        raise HTTPException(status_code=422, detail="fault_mode 必须是 none/half/all")
    return _pipe.run_batch(req.fault_mode)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
