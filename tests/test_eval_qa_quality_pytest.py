"""P0-3 端到端 QA 评测脚本的单元测试 — 验证评测代码质量与可行性。

评测脚本 (scripts/eval_qa_quality.py) 依赖 LLM 做 faithfulness/relevance 评判。
若环境没有 RAG_LLM_API_KEY, 真实打分跑不了, 但**代码正确性必须先被证明** ——
否则等拿到 key 再跑, 失败时无法区分是"数据问题"还是"脚本 bug"。

本测试用可控的 fake LLM 覆盖:
  - 分数解析 (JSON / markdown 包裹 / 畸形 / 越界)
  - 离线指标 (groundedness / numeric_ok)
  - 生成+双评判链路 (含 async 契约: LLMClient.chat 是协程, 必须 await)
  - 上下文重排 A/B 开关是否真的生效
  - LLM 不可用时是否优雅降级 (返回 None 而非抛异常)
"""

import asyncio
import os
import sys

import pytest

from api.core import retrieval

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import eval_qa_quality as eq


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """async chat —— 与生产 LLMClient 的契约一致 (漏 await 会拿到协程, 该测试能抓到)。"""

    def __init__(self, reply=None):
        self.reply = reply or '{"score": 0.8, "reason": "ok"}'
        self.calls = []

    async def chat(self, messages, temperature=0.1, max_tokens=2000):
        self.calls.append(messages[0]["content"])
        return _Resp(self.reply)


# ────────────────────────── 分数解析 ──────────────────────────


def test_parse_score_plain_json():
    assert eq._parse_score('{"score": 0.75, "reason": "x"}') == 0.75


def test_parse_score_markdown_fenced():
    """LLM 常把 JSON 包在 ```json 里, 必须能兜底解析, 否则整批评判作废。"""
    raw = '```json\n{"score": 0.4, "reason": "部分编造"}\n```'
    assert eq._parse_score(raw) == 0.4


def test_parse_score_clamps_out_of_range():
    assert eq._parse_score('{"score": 5}') == 1.0
    assert eq._parse_score('{"score": -2}') == 0.0


def test_parse_score_returns_none_on_garbage():
    """解析不了就返回 None (记为缺失), 不能抛异常污染整轮评测。"""
    assert eq._parse_score("我无法判断") is None
    assert eq._parse_score("") is None


# ────────────────────────── 离线指标 ──────────────────────────


def test_groundedness_full_and_zero():
    assert eq._groundedness("退款三个工作日到账", "退款三个工作日到账") == 1.0
    assert eq._groundedness("苹果香蕉橘子", "退款三个工作日到账") < 0.2


def test_groundedness_empty_answer_is_zero():
    assert eq._groundedness("", "任意上下文") == 0.0


def test_numeric_ok_supported_and_hallucinated():
    """答案数字是上下文数字的子集 -> 通过; 出现上下文没有的数字 -> 判 0。"""
    ctx = "退款一般一到三个工作日到账, 手续费百分之二"
    assert eq._numeric_ok("退款一到三个工作日到账", ctx) == 1.0
    assert eq._numeric_ok("退款三十个工作日到账", ctx) == 0.0


def test_numeric_ok_no_numbers_passes():
    assert eq._numeric_ok("退款会到账", "任意上下文") == 1.0


# ────────────────────────── 生成 + 双评判链路 ──────────────────────────


def test_gen_and_judge_awaits_async_chat():
    """核心契约: chat 是协程, 必须 await —— 否则拿到协程对象, 答案为空 (曾真实踩坑)。"""
    llm = _FakeLLM(reply='{"score": 0.9, "reason": "ok"}')
    judge = eq.Judge(enabled=False)
    judge.enabled = True
    judge.llm = llm

    res = asyncio.run(eq._gen_and_judge(llm, judge, "问题?", "上下文"))

    assert res is not None
    assert res["answer"], "答案不应为空 (若为空说明 chat 没 await)"
    assert res["faithfulness"] == 0.9
    assert res["relevance"] == 0.9


def test_judge_returns_none_when_disabled():
    judge = eq.Judge(enabled=False)
    assert asyncio.run(judge.faithfulness("q", "ctx", "ans")) is None
    assert asyncio.run(judge.relevance("q", "ans")) is None


def test_gen_and_judge_returns_none_on_empty_answer():
    class _EmptyLLM:
        async def chat(self, messages, temperature=0.1, max_tokens=2000):
            return _Resp("")

    judge = eq.Judge(enabled=False)
    assert asyncio.run(eq._gen_and_judge(_EmptyLLM(), judge, "q", "ctx")) is None


# ────────────────────────── 上下文重排 A/B ──────────────────────────


def _docs(n):
    return [
        {"doc_id": f"d{i}", "chunk_id": f"c{i}", "title_chain": [], "doc_title": f"文档{i}",
         "content": f"片段{i}的独特内容标记{i}"}
        for i in range(n)
    ]


def test_reorder_ab_produces_different_context():
    """A/B 开关必须真的改变上下文顺序, 否则对比结果无意义。"""
    docs = _docs(5)
    ctx_on, _ = eq._make_context(docs, top_k=5, reorder=True)
    ctx_off, _ = eq._make_context(docs, top_k=5, reorder=False)
    assert ctx_on != ctx_off, "重排开关似乎没生效"
    # 重排规则: 偶数位升序 + 奇数位降序, 首位应仍是最高相关的 p0
    assert "标记0" in ctx_on.split("---")[0]
    assert "标记0" in ctx_off.split("---")[0]


def test_reorder_restores_original_function():
    """A/B 用替换函数实现, 必须确保每次调用后还原, 不能污染后续请求。"""
    original = retrieval._reorder_for_attention
    eq._make_context(_docs(4), top_k=4, reorder=False)
    assert retrieval._reorder_for_attention is original, "重排函数未被还原, 会污染后续调用"


def test_reorder_restored_even_on_exception():
    original = retrieval._reorder_for_attention
    with pytest.raises(TypeError):
        eq._make_context(None, top_k=4, reorder=False)  # None 会触发异常
    assert retrieval._reorder_for_attention is original, "异常路径也必须还原重排函数"


# ────────────────────────── --no-llm 离线落数 (回归) ──────────────────────────


def test_no_llm_still_records_context_precision(tmp_path, monkeypatch):
    """回归: --no-llm 无答案时 context_precision 仍须记录 (旧代码 res is None 被 continue 漏记)。"""
    import json as _json

    chunks = [
        {"chunk_id": f"c{i}", "doc_id": f"d{i}", "title_chain": [], "doc_title": f"T{i}",
         "content": f"内容{i}唯一标记"}
        for i in range(6)
    ]
    dataset = [{"question": "q0", "expected_chunk_ids": ["c0", "c1"]} for _ in range(3)]
    edir = tmp_path / "eval"
    edir.mkdir()
    (edir / "corpus_chunks.json").write_text(_json.dumps(chunks), encoding="utf-8")
    (edir / "eval_dataset.json").write_text(_json.dumps(dataset), encoding="utf-8")

    class _FakeBM25:
        def search(self, q, k):
            return []

    class _FakeEmb:
        def embed_query(self, q):
            return [0.0] * 8

    class _FakeReranker:
        def __init__(self, **kw):
            pass

        def warmup(self):
            return True

        def rerank_fused(self, q, docs, rrf_map, top_k, alpha):
            return docs

    monkeypatch.setattr(eq, "EmbeddingService", _FakeEmb)
    monkeypatch.setattr(eq, "Reranker", _FakeReranker)
    monkeypatch.setattr(eq, "build_index", lambda chunks: (None, _FakeBM25()))
    monkeypatch.setattr(eq, "vector_topk", lambda *a, **k: [])
    monkeypatch.setattr(eq, "rrf_fuse", lambda *a, **k: [])
    monkeypatch.setattr(eq, "adaptive_alpha", lambda *a, **k: 1.0)
    monkeypatch.setattr(eq, "resolve_model_path", lambda m: m)

    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["eval_qa_quality.py", "--no-llm", "--limit", "3",
                                       "--eval-dir", str(edir), "--out", str(out)])
    eq.main()

    data = _json.loads(out.read_text(encoding="utf-8"))
    name = "重排开"
    assert data["variants"][name]["n"]["context_precision"] == 3, "离线模式应记录 context_precision"
    assert data["variants"][name]["metrics"]["context_precision"] is not None
    assert data["variants"][name]["n"]["faithfulness"] == 0, "无答案不应记录 LLM 指标"
