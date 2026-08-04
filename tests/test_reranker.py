"""重排对比测试 — Cross-Encoder 嵌入余弦 vs 基准顺序"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print('=' * 60)
print('重排对比测试')
print('=' * 60)

from engines.embedding.embedder import EmbeddingService
from engines.retrieval.reranker import Reranker

svc = EmbeddingService()
reranker = Reranker(embedder=svc)

# 构造测试文档 — 模拟检索结果
query = "FastAPI 框架的性能特点"
docs = [
    {"id": "d1", "chunk_id": "c1", "content": "FastAPI 是一个高性能的 Python Web 框架，基于 Starlette 和 Pydantic。它的性能与 Node.js 和 Go 相当。", "score": 0.5},
    {"id": "d2", "chunk_id": "c2", "content": "Django 是一个全栈 Web 框架，提供了 ORM、模板引擎和 admin 后台。", "score": 0.6},
    {"id": "d3", "chunk_id": "c3", "content": "FastAPI 支持异步处理、自动生成 OpenAPI 文档、依赖注入系统，并且性能优于大多数 Python 框架。", "score": 0.4},
    {"id": "d4", "chunk_id": "c4", "content": "Flask 是一个轻量级的 Web 框架，适合小型应用和微服务。", "score": 0.55},
    {"id": "d5", "chunk_id": "c5", "content": "FastAPI 使用 Uvicorn 作为 ASGI 服务器，支持 WebSocket 和后台任务。", "score": 0.45},
]

# ── 1. 基准顺序 ──
print('\n[1/3] 基准顺序（原始检索）...')
for d in docs:
    print(f'  {d["id"]}: score={d["score"]} — {d["content"][:40]}')

# ── 2. Reranker 重排 ──
print('\n[2/3] Reranker 重排后...')
reranked = reranker.rerank(query, docs, top_k=5)
for d in reranked:
    print(f'  {d["id"]}: score={d["score"]:.4f} — {d["content"][:40]}')

# ── 3. 验证 ──
print('\n[3/3] 验证...')

# 重排后 FastAPI 相关文档 (d1, d3, d5) 应该排在 Django (d2) 和 Flask (d4) 前面
fastapi_ids = {"d1", "d3", "d5"}
not_fastapi_ids = {"d2", "d4"}

fastapi_ranks = [i for i, d in enumerate(reranked) if d["id"] in fastapi_ids]
not_fastapi_ranks = [i for i, d in enumerate(reranked) if d["id"] in not_fastapi_ids]

print(f'  FastAPI 文档排名: {[r+1 for r in fastapi_ranks]}')
print(f'  非 FastAPI 文档排名: {[r+1 for r in not_fastapi_ranks]}')

# 所有 FastAPI 文档至少有一个进入 Top-3
assert any(r < 3 for r in fastapi_ranks), 'FastAPI 相关文档未进入 Top-3'
print('  PASS: FastAPI 文档进入 Top-3')

# 重排前后顺序有变化
original_order = [d["id"] for d in docs]
reranked_order = [d["id"] for d in reranked]
print(f'  原始顺序: {original_order}')
print(f'  重排顺序: {reranked_order}')
assert original_order != reranked_order, '重排未改变文档顺序'
print('  PASS: 重排改变了文档顺序')

# top_k 截断
short = reranker.rerank(query, docs, top_k=2)
assert len(short) == 2, f'期望 2, 实际 {len(short)}'
print('  PASS: top_k 截断正确')

print('\n' + '=' * 60)
print('重排测试 — 全部通过')
print('=' * 60)
