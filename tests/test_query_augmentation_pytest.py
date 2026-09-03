"""查询嵌入增广单测 — prf_augment_vector / blend_vectors 纯 numpy 语义验证。"""

import numpy as np

from engines.retrieval.query_augmentation import blend_vectors, prf_augment_vector


def _norm(v):
    return np.linalg.norm(np.asarray(v))


def test_weight_zero_returns_q_unchanged():
    q = np.array([1.0, 0.0], dtype=float)
    fb = [np.array([0.0, 1.0])]
    out = prf_augment_vector(q, fb, weight=0.0)
    assert np.allclose(out, q)


def test_empty_feedback_returns_q_unchanged():
    q = np.array([1.0, 0.0], dtype=float)
    out = prf_augment_vector(q, [], weight=0.5)
    assert np.allclose(out, q)
    out2 = prf_augment_vector(q, np.zeros((0, 2)), weight=0.5)
    assert np.allclose(out2, q)


def test_output_is_normalized():
    q = np.array([1.0, 0.0], dtype=float)
    fb = [np.array([0.6, 0.8]), np.array([0.8, 0.6])]
    out = prf_augment_vector(q, fb, weight=0.5)
    assert abs(_norm(out) - 1.0) < 1e-6


def test_prf_pulls_toward_feedback():
    # q 沿 x 轴, 反馈朝第一象限 → weight>0.5 时结果更贴近反馈 (与反馈余弦 > 与原始余弦)
    q = np.array([1.0, 0.0], dtype=float)
    fb = [np.array([0.8, 0.6])]  # 与 q 不平行
    out = prf_augment_vector(q, fb, weight=0.8)
    sim_fb = float(np.dot(out, fb[0]))
    sim_q = float(np.dot(out, q))
    assert sim_fb > sim_q  # 被反馈方向吸引


def test_blend_equals_single_feedback_prf():
    q = np.array([1.0, 0.0], dtype=float)
    other = np.array([0.0, 1.0], dtype=float)
    a = blend_vectors(q, other, weight=0.3)
    b = prf_augment_vector(q, [other], weight=0.3)
    assert np.allclose(a, b)


def test_weight_one_is_pure_feedback():
    q = np.array([1.0, 0.0], dtype=float)
    fb = [np.array([0.0, 1.0])]
    out = prf_augment_vector(q, fb, weight=1.0)  # 纯反馈方向(归一化后)
    assert np.allclose(out, np.array([0.0, 1.0], dtype=float))


def test_prf_timeout_falls_back_to_original_embedding(monkeypatch):
    """PRF 首轮检索挂起时必须回退原向量 —— 增广是增强项, 绝不能拖垮请求。

    PRF 默认开启后这是每问新增的同步阻塞调用, 无超时会让向量库故障直接传导为请求挂起。
    """
    import asyncio
    import time

    from api.config import settings
    from api.core import retrieval as R

    class _HangingStore:
        def search(self, *a, **k):
            time.sleep(0.5)  # 模拟向量库挂起
            return []

    monkeypatch.setattr(R, "get_vector_store", lambda kb: _HangingStore())
    monkeypatch.setattr(settings, "query_augmentation_strategy", "prf")
    monkeypatch.setattr(settings, "query_augmentation_prf_timeout_s", 0.1)

    q_emb = [1.0, 0.0]
    out = asyncio.run(R._augment_query_embedding(None, "q", q_emb, "kb", time.time()))
    assert out == q_emb  # 超时 → 原样返回, 不抛异常


def test_prf_augments_when_feedback_available(monkeypatch):
    """正常路径: 拿到反馈文档后查询向量应被拉动(与 PRF 语义一致)。"""
    import asyncio
    import time

    from api.config import settings
    from api.core import retrieval as R

    class _FakeStore:
        def search(self, q, k):
            return [{"embedding": [0.0, 1.0]} for _ in range(k)]

    monkeypatch.setattr(R, "get_vector_store", lambda kb: _FakeStore())
    monkeypatch.setattr(settings, "query_augmentation_strategy", "prf")
    monkeypatch.setattr(settings, "query_augmentation_weight", 0.5)

    q_emb = [1.0, 0.0]
    out = asyncio.run(R._augment_query_embedding(None, "q", q_emb, "kb", time.time()))
    assert out != q_emb
    assert float(np.dot(out, [0.0, 1.0])) > float(np.dot(q_emb, [0.0, 1.0]))  # 朝反馈方向偏移
