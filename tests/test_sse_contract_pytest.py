"""SSE 线上契约快照（P1-1）。

锁死的是**前端依赖的流式事件结构**，不是实现细节：
- `sources`：检索来源（首个事件，前端先渲染来源）
- `delta`：逐字增量（必须带 content，reasoning 可选）
- `done`：收尾（字段与非流式 result 对齐，多 type 与 qa_metrics）

契约变更时的正确做法：先改这里的期望值（= 明确承认协议变了），再改实现；
不要删断言。历史教训：T1 重构 DTO 时没有这类快照，红队测试红着没人发现。
"""
from __future__ import annotations

import pathlib
import re

import pytest

from api.core.response import build_done_event, build_sources_event

API_DIR = pathlib.Path(__file__).resolve().parents[1] / "api"

SOURCES_KEYS = {"type", "sources", "final", "retrieval_meta"}
DONE_KEYS = {
    "type",
    "answer",
    "token_usage",
    "degradation_level",
    "latency_breakdown",
    "retrieval_meta",
    "qa_metrics",
}


def test_sources_event_contract() -> None:
    ev = build_sources_event(docs=[], top1_score=0.0, cross_kb_kbs=None, degradation_level=0)

    assert ev["type"] == "sources"
    assert set(ev) == SOURCES_KEYS, f"sources 事件字段变了: {sorted(set(ev))}"


def test_done_event_contract() -> None:
    ev = build_done_event(
        answer="a",
        docs=[],
        token_usage={},
        degradation_level=0,
        router_ms=1.0,
        retrieval_ms=1.0,
        rerank_ms=1.0,
        llm_ms=1.0,
        total_ms=4.0,
        top1_score=0.0,
        cross_kb_kbs=None,
    )

    assert ev["type"] == "done"
    assert set(ev) == DONE_KEYS, f"done 事件字段变了: {sorted(set(ev))}"

    with_refusal = build_done_event(
        answer="a",
        docs=[],
        token_usage={},
        degradation_level=0,
        router_ms=1.0,
        retrieval_ms=1.0,
        rerank_ms=1.0,
        llm_ms=1.0,
        total_ms=4.0,
        top1_score=0.0,
        cross_kb_kbs=None,
        refusal={"reason": "low-score"},
    )
    assert set(with_refusal) == DONE_KEYS | {"refusal"}


def test_delta_events_always_carry_content() -> None:
    """delta 是逐字流式事件；缺 content 前端会渲染空白。

    只扫描真正的构造语句（`yield {` / `return {` 且同一行出现 type=delta），
    避免把文档里的 `{"type": "delta", content}` 简写误判成代码。
    """
    offenders = []
    pattern = re.compile(r'(yield|return)\s*\{[^\n]*["\']type["\']\s*:\s*["\']delta["\']')
    for path in API_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in pattern.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            line = text[line_start : line_end if line_end != -1 else len(text)]
            if '"content"' not in line and "'content'" not in line:
                offenders.append(f"{path.name}: {line.strip()[:80]}")

    assert offenders == [], f"delta 事件缺 content: {offenders}"


@pytest.mark.parametrize("event_type", ["sources", "done"])
def test_event_type_strings_are_stable(event_type: str) -> None:
    """事件类型字符串是前后端的隐式协议，禁止随手改名。"""
    if event_type == "sources":
        assert build_sources_event([], 0.0, None, 0)["type"] == "sources"
    else:
        ev = build_done_event(
            answer="a", docs=[], token_usage={}, degradation_level=0,
            router_ms=0.0, retrieval_ms=0.0, rerank_ms=0.0, llm_ms=0.0,
            total_ms=0.0, top1_score=0.0, cross_kb_kbs=None,
        )
        assert ev["type"] == "done"