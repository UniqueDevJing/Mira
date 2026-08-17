"""可观测性闭环测试 — 质量指标上报 + 结构化日志中间件。

验证方向②填补的真实空白:
  - /metrics 现在能看到"回答靠不靠谱" (faithfulness / top1_score 分布)
  - 请求日志为单行 JSON 且自动携带 trace_id, 可串联全链路
"""

import json
import logging
from io import StringIO

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, generate_latest

from api.core import metrics
from api.core.orchestrator import _record_qa_quality
from api.middleware.logging_middleware import JsonFormatter, RequestLoggingMiddleware, trace_id_ctx


def test_quality_metrics_exposed():
    """质量指标已注册到 REGISTRY 且 observe 后出现在 /metrics 文本中。"""
    metrics.qa_faithfulness.observe(0.9)
    metrics.qa_top1_score.observe(0.75)
    text = generate_latest(REGISTRY).decode()
    assert "rag_qa_faithfulness" in text
    assert "rag_qa_top1_score" in text


def test_record_qa_quality_helper():
    """_record_qa_quality 把 qa_metrics / retrieval_meta 里的质量信号上报且不抛异常。"""
    result = {
        "kb_id": "documents",
        "skill": "rag",
        "degradation_level": 0,
        "qa_metrics": {"faithfulness": 0.82},
        "retrieval_meta": {"top1_score": 0.66},
    }
    _record_qa_quality(result)  # 不应抛异常
    text = generate_latest(REGISTRY).decode()
    assert "rag_qa_faithfulness" in text
    assert "rag_qa_top1_score" in text


def test_json_formatter_carries_trace_id():
    """JsonFormatter 输出单行 JSON 且从 contextvar 取 trace_id。"""
    token = trace_id_ctx.set("abc123")
    try:
        rec = logging.LogRecord("api.x", logging.INFO, __file__, 1, "hi", None, None)
        out = JsonFormatter().format(rec)
        parsed = json.loads(out)
        assert parsed["msg"] == "hi"
        assert parsed["trace_id"] == "abc123"
        assert parsed["level"] == "INFO"
    finally:
        trace_id_ctx.reset(token)


def test_middleware_emits_json_request_log():
    """请求经中间件后, api.access 输出为 JSON 且含 trace_id/method/path/status。"""
    app = FastAPI()

    @app.get("/x")
    async def x():
        return {"ok": True}

    app.add_middleware(RequestLoggingMiddleware)

    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    access = logging.getLogger("api.access")
    access.addHandler(handler)
    access.setLevel(logging.INFO)
    try:
        client = TestClient(app)
        resp = client.get("/x")
        assert resp.status_code == 200
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        assert lines, "应有一条请求日志"
        rec = json.loads(lines[-1])
        assert rec["trace_id"] != "-"
        assert rec["method"] == "GET"
        assert rec["path"] == "/x"
        assert rec["status"] == 200
        assert "ms" in rec
    finally:
        access.removeHandler(handler)
