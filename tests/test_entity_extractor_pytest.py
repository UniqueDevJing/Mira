"""EntityExtractor 降级健壮性 — 验证 LLM 失败计数与纯规则兜底。

动机: EntityExtractor._llm_extract 此前在 except 中不累加 _fail_count,
导致 extract() 的降级分支形同虚设 (每次仍发起注定失败的 LLM 调用)。
本文件锁定降级计数契约。
"""

from engines.graph_rag.entity_extractor import EntityExtractor


class _RaisingClient:
    def __init__(self, counter):
        self._counter = counter

    def chat(self, **_kw):
        self._counter["n"] += 1
        raise RuntimeError("LLM down")

    def close(self):
        pass


def _make_extractor(counter):
    ex = EntityExtractor(llm_url="http://x", llm_model="m", llm_key="k")
    ex._get_llm_client = lambda: _RaisingClient(counter)
    return ex


def test_rule_extract_finds_tech_entities():
    ex = EntityExtractor()
    ents = ex.extract_rules("系统使用 FastAPI 和 Python 构建", "c1")
    names = {e.name for e in ents}
    assert "FastAPI" in names and "Python" in names


def test_llm_failure_triggers_degrade_count():
    counter = {"n": 0}
    ex = _make_extractor(counter)
    # 连续调用直至降级: 前 _max_fails 次尝试 LLM(失败计数), 之后走纯规则
    for _ in range(ex._max_fails + 1):
        ex.extract("使用 FastAPI 和 Python 构建", "c1")
    # 规则兜底始终生效 (不依赖 LLM)
    assert any(e.name == "FastAPI" for e in ex.extract_rules("使用 FastAPI 和 Python 构建", "c1"))
    # 失败累计到阈值, 触发降级
    assert ex._fail_count == ex._max_fails
    # 降级后不再发起 LLM 调用
    assert counter["n"] == ex._max_fails


def test_extract_batch_llm_failure_falls_back_per_batch():
    counter = {"n": 0}
    ex = _make_extractor(counter)
    for _ in range(ex._max_fails + 1):
        res = ex.extract_batch([("使用 FastAPI 构建", "c1")])
    assert "c1" in res
    assert any(e.name == "FastAPI" for e in res["c1"])
    assert ex._fail_count == ex._max_fails
    assert counter["n"] == ex._max_fails
