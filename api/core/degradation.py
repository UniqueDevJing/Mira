"""降级阶段维度辅助 (从 orchestrator 拆分)。

一次查询可能同时经历多阶段降级 (如 rerank 跳过 + LLM 失败), 等级维度 (0/1/2/3 取最高)
会丢失中间阶段。用 contextvar 在查询内累加命中阶段, 于查询末 flush 到 stage 维度指标。
"""

import contextvars

from api.core.metrics import track_degradation, track_degradation_stage

_deg_stages: contextvars.ContextVar = contextvars.ContextVar("deg_stages")


def _deg_reset() -> None:
    _deg_stages.set(set())


def _deg_bump(degradation: int, level: int, stage: str) -> int:
    """升级降级等级并记录发生阶段 (供 stage 维度指标)。"""
    degradation = max(degradation, level)
    s = _deg_stages.get()
    if s is None:
        s = set()
        _deg_stages.set(s)
    s.add(stage)
    return degradation


def _deg_mark(stage: str) -> None:
    s = _deg_stages.get()
    if s is None:
        s = set()
        _deg_stages.set(s)
    s.add(stage)


def _deg_flush() -> None:
    s = _deg_stages.get()
    if s:
        for stage in s:
            track_degradation_stage.labels(stage=stage).inc()
    _deg_stages.set(set())


def _deg_record_level(degradation: int) -> None:
    """查询末: 记录等级 + flush 阶段维度 (替换各 track_degradation.labels(level=).inc 调用处)。"""
    track_degradation.labels(level=str(degradation)).inc()
    _deg_flush()


def _deg_record_stage_only(stage: str) -> None:
    """仅记录某阶段降级 (如直接回答 Skill 的 LLM 失败), 不触碰等级计数。"""
    _deg_mark(stage)
    _deg_flush()
