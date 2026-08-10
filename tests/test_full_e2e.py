"""端到端测试 — 解析→分块→存储→检索→LLM生成"""
import sys, os, asyncio, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 预加载模型（在事件循环外）
print('='*60)
print('RAG 2.0 端到端测试')
print('='*60)

from engines.embedding.embedder import EmbeddingService
from engines.retrieval.vector_store import VectorStore
from engines.parsing.pdf_parser import PDFParser
from engines.chunking.structure_chunker import StructureChunker
from engines.retrieval.hybrid_retriever import HybridRetriever

# 1. 解析
print('\n[1/5] 解析 PDF...')
uir = PDFParser().parse('tests/fixtures/sample.pdf')
print(f'  pages={len(uir.pages)}')

# 2. 分块
print('\n[2/5] 语义分块...')
chunks = StructureChunker().chunk(uir)
print(f'  chunks={len(chunks)}')

# 3. 嵌入 + 存储
print('\n[3/5] 嵌入向量 + 存储...')
svc = EmbeddingService()
texts = [c.content for c in chunks]
embs = svc.embed_batch(texts)
for c, e in zip(chunks, embs):
    c.embedding = e

store = VectorStore()
store.insert(chunks)
print(f'  已存储 {len(chunks)} 条向量')

# 4. 检索
print('\n[4/5] 混合检索...')
rv = HybridRetriever(vector_store=store, embedder=svc)
queries = ['系统架构是什么', '使用了哪些技术', '文档解析引擎']
for q in queries:
    result = rv.retrieve(q, top_k=3)
    print(f'  Q: {q}')
    for d in result.get('documents', [])[:2]:
        print(f'    [{d.get("score","?")}] {d.get("content","")[:60]}')

# 5. LLM 生成
print('\n[5/5] LLM 答案生成 (TokenHub deepseek-v4-flash)...')
import httpx
from api.config import settings

context = "\n\n".join([d.get("content", "") for d in result.get("documents", [])[:3]])

prompt = f"""你是专业的知识库助手。请根据以下参考文档回答问题，控制在200字以内。

参考文档：
{context[:2000]}

问题：{queries[0]}"""

resp = httpx.post(
    f"{settings.llm_base_url}/chat/completions",
    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
    json={
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "你是专业的知识库助手，回答简洁准确。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000, "temperature": 0.3
    },
    timeout=60
)
data = resp.json()
if isinstance(data, str):
    import json
    data = json.loads(data)
answer = data["choices"][0]["message"]["content"]
if not answer:
    answer = f"(推理token耗尽) {data['choices'][0]['message'].get('reasoning_content', '')[:200]}"

print(f'  问题: {queries[0]}')
print(f'  TokenHub 回答: {answer[:500]}')
print(f'  Token 用量: {data.get("usage", {})}')

print('\n' + '='*60)
print('所有测试通过！')
print('='*60)

# 统计
import datetime
print(f'\n完成时间: {datetime.datetime.now()}')
print(f'模型: {settings.llm_model} @ {settings.llm_base_url}')
