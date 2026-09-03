"""P2#11 Rerank GPU 自适应决策矩阵 + get_reranker None 契约测试。"""

import types

import pytest

from api.config import settings
from api.state import rerank_effective_enabled


@pytest.fixture()
def _restore():
    """保存/恢复 settings 相关字段, 避免测试间污染。"""
    saved = {k: getattr(settings, k) for k in ("rerank_enabled", "reranker_auto_gpu")}
    saved_fields = set(settings.model_fields_set)
    yield
    for k, v in saved.items():
        # 绕过 setattr, 避免 model_fields_set 被污染
        object.__setattr__(settings, k, v)
    # 恢复 fields_set: 移除测试期间新增的键
    settings.model_fields_set.intersection_update(saved_fields)  # type: ignore[attr-defined]


def _set_explicit(field: str, value) -> None:
    """模拟"用户显式设定": setattr 并让 fields_set 包含该字段。"""
    setattr(settings, field, value)


def _clear_explicit(field: str) -> None:
    """模拟"未显式设定": 恢复默认值并从 fields_set 移除。"""
    object.__setattr__(settings, field, type(settings).model_fields[field].default)
    settings.model_fields_set.discard(field)  # type: ignore[attr-defined]


def test_explicit_true_always_enabled(_restore):
    _set_explicit("rerank_enabled", True)
    _set_explicit("reranker_auto_gpu", False)
    assert rerank_effective_enabled() is True


def test_explicit_false_respected_even_with_gpu(_restore, monkeypatch):
    """显式 False 即使有 GPU 也尊重用户 → 关。"""
    import sys

    _set_explicit("rerank_enabled", False)
    _set_explicit("reranker_auto_gpu", True)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert rerank_effective_enabled() is False


def test_auto_gpu_disabled(_restore):
    _clear_explicit("rerank_enabled")
    _set_explicit("reranker_auto_gpu", False)
    assert rerank_effective_enabled() is False


def test_auto_gpu_no_torch(_restore, monkeypatch):
    """torch 不可用 → 当作无 GPU → 关 (本机 CPU 环境)。"""
    import builtins
    import sys

    _clear_explicit("rerank_enabled")
    _set_explicit("reranker_auto_gpu", True)
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "torch", None)  # 强制重新 import 走 fake_import
    assert rerank_effective_enabled() is False


def test_auto_gpu_with_cuda(_restore, monkeypatch):
    """未显式设定 + 检测到 CUDA → 自动开启。"""
    import sys

    _clear_explicit("rerank_enabled")
    _set_explicit("reranker_auto_gpu", True)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert rerank_effective_enabled() is True


def test_auto_gpu_no_cuda(_restore, monkeypatch):
    """torch 可用但无 CUDA → 关。"""
    import sys

    _clear_explicit("rerank_enabled")
    _set_explicit("reranker_auto_gpu", True)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert rerank_effective_enabled() is False


def test_get_reranker_returns_none_when_disabled(_restore):
    """未生效时 get_reranker 返回 None (调用方契约)。"""
    from api.state import get_reranker

    _set_explicit("rerank_enabled", False)
    _set_explicit("reranker_auto_gpu", False)
    assert get_reranker() is None
