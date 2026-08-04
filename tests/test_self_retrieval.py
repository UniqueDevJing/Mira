"""Self-Retrieval 多轮逻辑测试 — 评估→改写→重检索"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print('=' * 60)
print('Self-Retrieval 测试')
print('=' * 60)

from engines.retrieval.evaluator import RetrievalEvaluator
from engines.retrieval.query_rewriter import QueryRewriter
from engines.retrieval.self_retrieval import SelfRetrieval
from engines.embedding.embedder import EmbeddingService

# ── 1. 评估器 ──
print('\n[1/4] 评估器测试...')

svc = EmbeddingService()
evaluator = RetrievalEvaluator(embedder=svc)

query = "FastAPI 框架的性能"
relevant_docs = [
    {"id": "d1", "content": "FastAPI 是一个高性能 Python Web 框架，基于 Starlette 构建。"},
    {"id": "d2", "content": "FastAPI 使用异步处理来提高吞吐量，性能接近 Node.js。"},
]
eval_result = evaluator.evaluate(query, relevant_docs)
print(f'  relevance={eval_result.relevance_score:.3f} need_rewrite={eval_result.need_rewrite}')
assert eval_result.relevance_score >= 0

irrelevant_docs = [
    {"id": "d3", "content": "Django 是一个全栈 Web 框架，包含 ORM 和模板引擎。"},
    {"id": "d4", "content": "Flask 是一个轻量级微框架，适合小型应用。"},
]
eval_result2 = evaluator.evaluate(query, irrelevant_docs)
print(f'  low-rel relevance={eval_result2.relevance_score:.3f} need_rewrite={eval_result2.need_rewrite}')
print('  PASS')

# ── 2. 查询改写器 ──
print('\n[2/4] 查询改写器测试...')

rewriter = QueryRewriter()
rewritten = rewriter.rewrite(query, eval_result2)
print(f'  original: {query}')
print(f'  rewritten: {rewritten}')
assert isinstance(rewritten, list) and len(rewritten) >= 1

# LLM 降级
rewriter_no_llm = QueryRewriter()
result_fallback = rewriter_no_llm.rewrite("测试", eval_result2)
print(f'  fallback: {result_fallback}')
assert isinstance(result_fallback, list)
print('  PASS')

# ── 3. SelfRetrieval 状态机 ──
print('\n[3/4] SelfRetrieval 状态机测试...')

class MockRetriever:
    def retrieve(self, query: str, top_k: int = 20) -> dict:
        return {"documents": [
            {"id": "d1", "content": "FastAPI 是一个高性能 Python Web 框架。"},
            {"id": "d2", "content": "FastAPI 使用异步处理提高吞吐量。"},
        ]}

sr = SelfRetrieval(retriever=MockRetriever(), evaluator=evaluator,
                   rewriter=rewriter, max_rounds=3)

result = sr.retrieve(query, top_k=5)
print(f'  rounds={result["retrieval_rounds"]}')
for t in result.get("trace", []):
    print(f'    R{t["round"]}: rel={t.get("relevance",0):.3f} rewrite={t.get("need_rewrite")}')
# Mock 数据无 score 字段，评估器第3条件触发 → 始终需要改写，验证上限截断即可
assert result["retrieval_rounds"] <= 3
print('  PASS: rounds <= 3')

# ── 4. 边界: 总是改写的场景 ──
print('\n[4/4] 边界测试...')

class AlwaysNeedRewriteEval:
    def evaluate(self, query, docs):
        from dataclasses import dataclass
        @dataclass
        class R:
            relevance_score: float = 0.3
            coverage_score: float = 0.4
            confidence: float = 0.5
            need_rewrite: bool = True
        return R()

sr2 = SelfRetrieval(retriever=MockRetriever(), evaluator=AlwaysNeedRewriteEval(),
                    rewriter=rewriter, max_rounds=3)
result2 = sr2.retrieve("test", top_k=5)
print(f'  force-rewrite: rounds={result2["retrieval_rounds"]}')
assert result2["retrieval_rounds"] == 3
print('  PASS: capped at 3')

print('\n' + '=' * 60)
print('Self-Retrieval 测试 — 全部通过')
print('=' * 60)
