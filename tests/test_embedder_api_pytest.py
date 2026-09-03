"""Embedding 双模单测 — local(SentenceTransformer) 与 api(OpenAI 兼容) 后端分发。

API 路径用 mock httpx.Client, 无需真实网络/Key; local 路径 mock _get_model, 不触发模型下载。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.embedding.embedder import EmbeddingService

_DIM = 8


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": self._data}


class _FakeClient:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, headers=None):
        self.calls.append((url, json, headers))
        n = len(json["input"])
        data = [{"index": i, "embedding": [1.0 / (i + 1)] * _DIM} for i in range(n)]
        return _FakeResp(data)

    def close(self):
        return None


def test_api_embed_batch_shape_and_normalized(monkeypatch):
    fc = _FakeClient()
    monkeypatch.setattr("httpx.Client", lambda *a, **k: fc)
    svc = EmbeddingService(backend="api", api_base="https://x/v1", api_key="k", api_model="m", api_dims=_DIM)
    out = svc.embed_batch(["a", "b"])
    assert len(out) == 2
    assert all(len(v) == _DIM for v in out)
    # L2 归一化
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-6
    # URL / 鉴权头 / payload
    url, payload, headers = fc.calls[0]
    assert url == "https://x/v1/embeddings"
    assert headers["Authorization"] == "Bearer k"
    assert payload["model"] == "m"
    assert payload["dimensions"] == _DIM
    assert payload["input"] == ["a", "b"]


def test_api_embed_batch_no_dims_when_zero(monkeypatch):
    fc = _FakeClient()
    monkeypatch.setattr("httpx.Client", lambda *a, **k: fc)
    svc = EmbeddingService(backend="api", api_base="https://x/v1", api_model="m")  # api_dims=0
    svc.embed_batch(["a"])
    assert "dimensions" not in fc.calls[0][1]


def test_api_embed_query_caches(monkeypatch):
    fc = _FakeClient()
    monkeypatch.setattr("httpx.Client", lambda *a, **k: fc)
    svc = EmbeddingService(backend="api", api_base="https://x/v1", api_key="k", api_model="m", api_dims=_DIM)
    v1 = svc.embed_query("同一个问题")
    v2 = svc.embed_query("同一个问题")
    assert v1 == v2
    assert len(fc.calls) == 1  # 第二次命中缓存, 不发请求


def test_api_empty_base_raises():
    svc = EmbeddingService(backend="api", api_base="", api_model="m")
    try:
        svc.embed_batch(["x"])
        raise AssertionError("应当因缺 api_base 抛 ValueError")
    except ValueError:
        pass


def test_local_dispatches_to_model(monkeypatch):
    captured = {}

    def fake_get_model(name, device):
        captured["name"] = name
        captured["device"] = device

        class _M:
            tokenizer = None

            def encode(self, texts, **kw):
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        return _M()

    monkeypatch.setattr("engines.embedding.embedder._get_model", fake_get_model)
    svc = EmbeddingService(model_name="local-m", device="cpu", backend="local")
    out = svc.embed_batch(["x"], max_length=0)  # max_length=0 跳过 tokenizer 截断
    assert captured["name"] == "local-m"
    assert len(out) == 1 and len(out[0]) == 4


def test_default_backend_is_local():
    svc = EmbeddingService()
    assert svc.backend == "local"
    assert svc.model_name == "BAAI/bge-small-zh-v1.5"
