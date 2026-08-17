"""Embedding 查询缓存 key 含模型名 (C3) 单测 — 不加载模型"""

from engines.embedding.embedder import _query_cache_key


def test_model_name_in_key_isolates_vectors():
    """不同模型名 → 不同 key; 同模型同 query → 同 key (防模型升级后旧向量命中)"""
    k1 = _query_cache_key("如何部署", "BAAI/bge-small-zh-v1.5")
    k2 = _query_cache_key("如何部署", "BAAI/bge-large-zh-v1.5")
    k3 = _query_cache_key("如何部署", "BAAI/bge-small-zh-v1.5")
    assert k1 != k2
    assert k1 == k3


def test_key_case_sensitive_preserved():
    """key 与 "query:" 前缀嵌入输入一致, 保持大小写敏感 (不误共享)"""
    assert _query_cache_key("API是什么", "m") != _query_cache_key("api是什么", "m")
    # 仅空白差异仍归并为同 key (与嵌入输入一致)
    assert _query_cache_key("如何部署", "m") == _query_cache_key(" 如何部署 ", "m")
