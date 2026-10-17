"""降级阶段追踪（api/core/degradation.py）的行为测试。

该模块此前**零测试覆盖**，而它承载的是「线上是否正在降级」这一最关键的
可观测性信号 —— 覆盖缺失正是下面这类缺陷能长期潜伏的原因。

核心语义：
- 一次查询可经历多阶段降级（rerank 跳过 + LLM 降级），阶段维度用 set 累积，
  而等级维度（level=0/1/2/3）取最高值，两者用途不同。
- 阶段标记用 contextvar 承载：asyncio 每个 task 是独立 context，天然隔离。
- 异常路径必须兜底落盘：Skill 正常路径调 `_deg_record_level` 会 flush，
  但若在 reset 与 record 之间抛异常，此前标记会永久丢失。
  `deg_flush_if_pending()` 由 RequestLoggingMiddleware 在请求结束
  （含异常路径）无条件调用来完成兜底，且必须幂等。
"""

import sys

import pytest

sys.path.insert(0, ".")

from api.core import degradation as deg  # noqa: E402


class TestLevelAggregation:
    """等级维度取最大值 —— 中间阶段不能覆盖更高等级。"""

    def test_level_takes_maximum(self):
        assert deg._deg_bump(0, 1, "rerank") == 1
        assert deg._deg_bump(1, 3, "llm") == 3      # 更高等级胜出
        assert deg._deg_bump(3, 1, "guardrail") == 3  # 更低等级不降级

    def test_level_zero_is_valid(self):
        """level=0 表示未降级，是最常见的值，不能被当成「无值」。"""
        assert deg._deg_bump(0, 0, "none") == 0

    def test_accumulates_multiple_stages_in_one_query(self):
        """一次查询多阶段降级：阶段维度应全部累积（这正是它存在的理由）。"""
        deg._deg_reset()
        deg._deg_bump(0, 1, "rerank")
        deg._deg_bump(0, 2, "llm")
        deg._deg_bump(0, 1, "rerank")  # 同阶段重复不应重复计数
        stages = deg._deg_stages.get()
        assert stages == {"rerank", "llm"}

    def test_mark_adds_stage_without_touching_level(self):
        deg._deg_reset()
        deg._deg_mark("llm")
        assert deg._deg_stages.get() == {"llm"}


class TestFlushIdempotency:
    """flush 必须幂等 —— 这是「请求级兜底」能成立的前提。"""

    def test_second_flush_is_noop(self):
        deg._deg_reset()
        deg._deg_mark("rerank")
        deg._deg_flush()
        after_first = deg._deg_stages.get()
        deg._deg_flush()
        assert after_first == set()
        assert deg._deg_stages.get() == set()

    def test_flush_if_pending_after_record_level_is_noop(self):
        """正常路径：record_level 已 flush，兜底再调不应产生重复计数。"""
        deg._deg_reset()
        deg._deg_bump(0, 1, "rerank")
        deg._deg_record_level(1)      # 正常路径：记等级 + flush
        deg.deg_flush_if_pending()    # 中间件兜底：应无副作用
        assert deg._deg_stages.get() == set()

    def test_no_degradation_produces_nothing(self):
        deg._deg_reset()
        deg.deg_flush_if_pending()
        assert deg._deg_stages.get() == set()


class TestExceptionSafety:
    """异常路径：已发生的降级必须被兜底记录，不能静默吞掉。"""

    def test_exception_path_still_flushes(self):
        """模拟 Skill 在 reset 与 record 之间抛异常。"""
        deg._deg_reset()
        deg._deg_bump(0, 1, "rerank")
        deg._deg_mark("llm")
        # 异常跳过 _deg_record_level，由中间件 finally 兜底
        with pytest.raises(RuntimeError):
            try:
                raise RuntimeError("上游 5xx")
            finally:
                deg.deg_flush_if_pending()
        assert deg._deg_stages.get() == set(), "兜底后容器应被清空"

    def test_guard_flush_helper_available(self):
        """deg_flush_if_pending 必须可从中间件导入（惰性导入路径）。"""
        from api.core.degradation import deg_flush_if_pending

        assert callable(deg_flush_if_pending)


class TestResetSemantics:
    """reset 的语义：显式开始新查询，不继承上一轮的脏数据。"""

    def test_reset_clears_previous_accumulation(self):
        deg._deg_reset()
        deg._deg_mark("stale")
        deg._deg_reset()
        assert deg._deg_stages.get() == set(), "reset 应丢弃上一轮残留"

    def test_reset_returns_none(self):
        """reset 无返回值 —— 兜底职责在中间件，不在调用方。"""
        assert deg._deg_reset() is None

    def test_bump_works_without_explicit_reset(self):
        """未 reset 就 bump 时应自动建容器，不应抛 AttributeError。"""
        deg._deg_stages.set(None)
        assert deg._deg_bump(0, 1, "auto") == 1
        assert deg._deg_stages.get() == {"auto"}

    def test_mark_works_without_explicit_reset(self):
        deg._deg_stages.set(None)
        deg._deg_mark("auto")
        assert deg._deg_stages.get() == {"auto"}
