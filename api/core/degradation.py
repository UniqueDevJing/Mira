"""降级阶段维度辅助 (从 orchestrator 拆分)。

一次查询可能同时经历多阶段降级 (如 rerank 跳过 + LLM 失败), 等级维度 (0/1/2/3 取最高)
会丢失中间阶段。用 contextvar 在查询内累加命中阶段, 于查询末 flush 到 stage 维度指标。
"""

import contextvars

from api.core.metrics import track_degradation, track_degradation_stage

_deg_stages: contextvars.ContextVar = contextvars.ContextVar("deg_stages")


def _deg_reset() -> None:
    """开始一次降级追踪。

    异常路径的兜底不在调用方, 而由 RequestLoggingMiddleware 在请求结束时
    统一调 `deg_flush_if_pending()` —— 它已是必经出口, 对 Skill 零侵入,
    且未来新增任何调用降级 API 的路径都自动受保护。
    """
    _deg_stages.set(set())


def _deg_bump(degradation: int, level: int, stage: str) -> int:
    """升级降级等级并记录发生阶段 (供 stage 维度指标)。"""
    degradation = max(degradation, level)
    s = _deg_stages.get(None)
    if s is None:
        s = set()
        _deg_stages.set(s)
    s.add(stage)
    return degradation


def _deg_mark(stage: str) -> None:
    s = _deg_stages.get(None)
    if s is None:
        s = set()
        _deg_stages.set(s)
    s.add(stage)


def _deg_flush() -> None:
    s = _deg_stages.get(None)
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


def deg_flush_if_pending() -> None:
    """请求级兜底：把尚未落盘的降级阶段标记写入指标。

    由中间件在请求结束时无条件调用。相比在每个 Skill 里加 try/finally，
    请求级兜底有三个优势：
      1. **零侵入** —— 不需要给四个 Skill（_skill_rag / _stream_rag /
         _stream_direct / _skill_direct，其中两个是 async 生成器）
         的上百行函数体加缩进；
      2. **覆盖面更全** —— 未来新增任何调用降级 API 的路径都自动受保护，
         不依赖开发者记得写 finally；
      3. **幂等** —— 正常路径已由 `_deg_record_level` flush 过，
         这里读到空集合，无重复计数。

    典型漏记场景（此前真实存在）：
        检索阶段已 `_deg_bump(1, "rerank")`，
        随后 LLM 调用抛 5xx → 函数栈退出，从未走到 `_deg_record_level`
        → Prometheus 上 degradation_stage 上一片干净，
        而线上实际正在降级。可观测性恰好在故障场景失效。
    """
    _deg_flush()
