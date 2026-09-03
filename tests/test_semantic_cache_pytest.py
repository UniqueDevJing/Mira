"""语义近重复缓存单测 — 在精确哈希之上, 同作用域内 near-duplicate 命中, 跨作用域/低相似度不命中。

用注入的 embed_fn (确定性向量表) 避免加载真实 embedding 模型。
"""


from api.core.qa_cache import QACache

# 确定性向量表: 近义问题映射到高余弦, 不同问题低余弦
_VECS = {
    "q1": [1.0, 0.0],
    "q2": [0.99, 0.01],   # 与 q1 余弦 0.99
    "q3": [0.98, 0.02],   # 与 q1 余弦 0.98
    "other": [0.0, 1.0],  # 与 q1 余弦 0
}


def _embed_fn(q):
    return _VECS[q]


_BASE_ARGS = {"question": "q1", "skill": None, "top_k": 10, "enable_self_retrieval": False, "temperature": 0.1}


def _key_and_scope(question):
    a = dict(_BASE_ARGS, question=question)
    return QACache.make_key(**a), QACache.make_scope(**a)


def test_semantic_near_dup_hits_same_scope():
    c = QACache(embed_fn=_embed_fn, semantic_enabled=True)
    key, scope = _key_and_scope("q1")
    c.set(key, {"answer": "A"}, 600, question="q1", scope=scope)
    # 同作用域内近义问题 q2 应语义命中
    k2, s2 = _key_and_scope("q2")
    assert c.get(k2, question="q2", scope=s2) == {"answer": "A"}
    # 不同问题(other)低余弦 → 不命中
    k3, s3 = _key_and_scope("other")
    assert c.get(k3, question="other", scope=s3) is None


def test_semantic_disabled_falls_back_to_exact_only():
    c = QACache(embed_fn=_embed_fn, semantic_enabled=False)
    key, scope = _key_and_scope("q1")
    c.set(key, {"answer": "A"}, 600, question="q1", scope=scope)
    k2, s2 = _key_and_scope("q2")
    assert c.get(k2, question="q2", scope=s2) is None  # 语义关 → 仅精确, 不命中


def test_scope_isolation_blocks_cross_param_hit():
    # 同问题但不同 temperature → 不同 scope → 语义不命中
    c = QACache(embed_fn=_embed_fn, semantic_enabled=True)
    key, scope = _key_and_scope("q1")
    c.set(key, {"answer": "A"}, 600, question="q1", scope=scope)
    a2 = dict(_BASE_ARGS, question="q2", temperature=0.9)
    k2 = QACache.make_key(**a2)
    s2 = QACache.make_scope(**a2)
    assert c.get(k2, question="q2", scope=s2) is None


def test_semantic_expiry_not_returned():
    c = QACache(embed_fn=_embed_fn, semantic_enabled=True)
    key, scope = _key_and_scope("q1")
    c.set(key, {"answer": "A"}, ttl_s=-1, question="q1", scope=scope)  # 立即过期
    k2, s2 = _key_and_scope("q2")
    assert c.get(k2, question="q2", scope=s2) is None


def test_exact_hit_still_works_with_question_passed():
    c = QACache(embed_fn=_embed_fn, semantic_enabled=True)
    key, scope = _key_and_scope("q1")
    c.set(key, {"answer": "A"}, 600, question="q1", scope=scope)
    # 相同 question 精确命中(无需语义)
    assert c.get(key, question="q1", scope=scope) == {"answer": "A"}


def test_make_scope_ignores_question_text():
    s_a = QACache.make_scope(**dict(_BASE_ARGS, question="问题A"))
    s_b = QACache.make_scope(**dict(_BASE_ARGS, question="问题B"))
    assert s_a == s_b  # scope 仅由除 question 外的参数决定
    assert s_a != QACache.make_key(**dict(_BASE_ARGS, question="问题A"))  # 与完整 key 不同


def test_stats_sink_reports_exact_kind():
    c = QACache(embed_fn=_embed_fn, semantic_enabled=True)
    key, scope = _key_and_scope("q1")
    c.set(key, {"answer": "A"}, 600, question="q1", scope=scope)
    stats: dict = {}
    assert c.get(key, question="q1", scope=scope, stats=stats) == {"answer": "A"}
    assert stats["kind"] == "exact"


def test_stats_sink_reports_semantic_kind_with_similarity():
    c = QACache(embed_fn=_embed_fn, semantic_enabled=True)
    key, scope = _key_and_scope("q1")
    c.set(key, {"answer": "A"}, 600, question="q1", scope=scope)
    k2, s2 = _key_and_scope("q2")
    stats: dict = {}
    assert c.get(k2, question="q2", scope=s2, stats=stats) == {"answer": "A"}
    assert stats["kind"] == "semantic"
    assert stats["sim"] >= c._sem_threshold  # 命中必达阈值, 且暴露相似度供阈值调优


def test_stats_sink_stays_empty_on_miss():
    """未命中不回填 kind —— skills.py 据此不计数, 避免把 miss 误计成语义命中。"""
    c = QACache(embed_fn=_embed_fn, semantic_enabled=True)
    key, scope = _key_and_scope("q1")
    c.set(key, {"answer": "A"}, 600, question="q1", scope=scope)
    k3, s3 = _key_and_scope("other")
    stats: dict = {}
    assert c.get(k3, question="other", scope=s3, stats=stats) is None
    assert stats == {}
