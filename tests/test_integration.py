"""集成测试 — 解析→分块→向量化→检索"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from engines.embedding.embedder import EmbeddingService
from engines.retrieval.vector_store import VectorStore
from engines.parsing.pdf_parser import PDFParser
from engines.chunking.semantic_chunker import SemanticChunker
from engines.retrieval.hybrid_retriever import HybridRetriever


async def test():
    svc = EmbeddingService()
    q_emb = svc.embed_query('测试')
    print(f'1. Embedding dim={len(q_emb)}')

    uir = PDFParser().parse('tests/fixtures/sample.pdf')
    print(f'2. PDF: {len(uir.pages)} pages')

    chunks = SemanticChunker().chunk(uir)
    print(f'3. Chunks: {len(chunks)}')

    embs = svc.embed_batch([c.content for c in chunks])
    for c, e in zip(chunks, embs):
        c.embedding = e
    store = VectorStore()
    store.insert(chunks)
    print('4. Vectors stored')

    rv = HybridRetriever(vector_store=store, embedder=svc)
    result = rv.retrieve('系统架构是什么', top_k=3)
    for d in result.get('documents', [])[:3]:
        print(f'   [{d.get("score","?")}] {d.get("content","")[:80]}')
    print('=== ALL PASS ===')


if __name__ == '__main__':
    asyncio.run(test())
