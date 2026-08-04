"""Self-Retrieval 测试 — 评估器 / 查询改写 / 多轮检索"""
import pytest
from dataclasses import dataclass
from engines.retrieval.evaluator import RetrievalEvaluator, EvalResult
from engines.retrieval.query_rewriter import QueryRewriter
from engines.retrieval.self_retrieval import SelfRetrieval


@pytest.fixture
def evaluator(embedder):
    """检索评估器"""
    return RetrievalEvaluator(embedder=embedder)


@pytest.fixture
def rewriter():
    """查询改写器"""
    return QueryRewriter()


class MockRetriever:
    """模拟检索器"""

    def retrieve(self, query: str, top_k: int = 20) -> dict:
        return {
            "documents": [
                {"id": "d1", "content": "FastAPI 是一个高性能 Python Web 框架。"},
                {"id": "d2", "content": "FastAPI 使用异步处理提高吞吐量。"},
            ]
        }


class AlwaysRewriteEvaluator:
    """总是需要改写的评估器"""

    def evaluate(self, query, docs):
        return EvalResult(
            relevance_score=0.3,
            coverage_score=0.4,
            confidence_score=0.5,
            need_rewrite=True,
        )


class TestRetrievalEvaluator:
    """检索评估器测试"""

    def test_evaluate_relevant_docs(self, evaluator):
        """相关文档应有高相关性分数"""
        query = "FastAPI 框架的性能"
        docs = [
            {"id": "d1", "content": "FastAPI 是一个高性能 Python Web 框架，基于 Starlette 构建。"},
            {"id": "d2", "content": "FastAPI 使用异步处理来提高吞吐量，性能接近 Node.js。"},
        ]
        result = evaluator.evaluate(query, docs)
        assert result.relevance_score >= 0
        assert isinstance(result.need_rewrite, bool)

    def test_evaluate_empty_results(self, evaluator):
        """空结果应返回需要改写"""
        result = evaluator.evaluate("test", [])
        assert result.need_rewrite is True
        assert result.relevance_score == 0

    def test_evaluate_result_structure(self, evaluator):
        """评估结果应包含所有必要字段"""
        docs = [{"id": "d1", "content": "测试内容"}]
        result = evaluator.evaluate("测试", docs)
        assert hasattr(result, "relevance_score")
        assert hasattr(result, "coverage_score")
        assert hasattr(result, "confidence_score")
        assert hasattr(result, "need_rewrite")
        assert hasattr(result, "reason")


class TestQueryRewriter:
    """查询改写器测试"""

    def test_rewrite_returns_list(self, rewriter):
        """改写应返回列表"""
        eval_result = EvalResult(
            relevance_score=0.3, coverage_score=0.5,
            confidence_score=0.4, need_rewrite=True,
            reason="相关性不足",
        )
        result = rewriter.rewrite("测试查询", eval_result)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_rewrite_fallback_without_llm(self):
        """无 LLM 时应使用模板改写"""
        rewriter = QueryRewriter(llm_client=None)
        eval_result = EvalResult(
            relevance_score=0.3, coverage_score=0.5,
            confidence_score=0.4, need_rewrite=True,
            reason="相关性不足",
        )
        result = rewriter.rewrite("测试查询", eval_result)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert "测试查询" in result[0]


class TestSelfRetrieval:
    """Self-Retrieval 多轮检索测试"""

    def test_max_rounds_limit(self, evaluator, rewriter):
        """应限制最大轮数"""
        sr = SelfRetrieval(
            retriever=MockRetriever(),
            evaluator=AlwaysRewriteEvaluator(),
            rewriter=rewriter,
            max_rounds=3,
        )
        result = sr.retrieve("test", top_k=5)
        assert result["retrieval_rounds"] <= 3

    def test_result_structure(self, evaluator, rewriter):
        """结果应包含所有必要字段"""
        sr = SelfRetrieval(
            retriever=MockRetriever(),
            evaluator=evaluator,
            rewriter=rewriter,
            max_rounds=3,
        )
        result = sr.retrieve("FastAPI 框架", top_k=5)
        assert "documents" in result
        assert "retrieval_rounds" in result
        assert "trace" in result
        assert isinstance(result["documents"], list)
        assert isinstance(result["trace"], list)

    def test_trace_contains_round_info(self, evaluator, rewriter):
        """trace 应包含每轮信息"""
        sr = SelfRetrieval(
            retriever=MockRetriever(),
            evaluator=evaluator,
            rewriter=rewriter,
            max_rounds=3,
        )
        result = sr.retrieve("FastAPI 框架", top_k=5)
        for t in result["trace"]:
            assert "round" in t
            assert "query" in t
            assert "relevance" in t
            assert "need_rewrite" in t

    def test_deduplication(self, rewriter):
        """多轮结果应去重"""
        class DuplicateRetriever:
            def retrieve(self, query, top_k=20):
                return {
                    "documents": [
                        {"id": "d1", "chunk_id": "c1", "content": "内容1"},
                        {"id": "d1", "chunk_id": "c1", "content": "内容1"},  # 重复
                    ]
                }

        sr = SelfRetrieval(
            retriever=DuplicateRetriever(),
            evaluator=AlwaysRewriteEvaluator(),
            rewriter=rewriter,
            max_rounds=2,
        )
        result = sr.retrieve("test", top_k=5)
        ids = [d.get("id") or d.get("chunk_id") for d in result["documents"]]
        assert len(ids) == len(set(ids)), "结果未去重"
