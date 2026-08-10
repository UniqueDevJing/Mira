"""Phase 3+ 端到端测试 — 包含图谱集成"""
import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
logging.basicConfig(level=logging.WARNING)

print('=' * 60)
print('RAG 2.0 Phase 3+ 端到端测试（含图谱集成）')
print('=' * 60)

LLM_URL = "https://tokenhub.itcast.cn/v1"
LLM_MODEL = "deepseek-v4-flash"
LLM_KEY = os.environ.get("RAG_LLM_API_KEY", "")

# 1. 文档解析
print('\n[1/5] 文档解析...')
from engines.parsing.pdf_parser import PDFParser
uir = PDFParser().parse('tests/fixtures/sample.pdf')
print(f'  {len(uir.pages)} 页')

# 2. 语义分块
print('\n[2/5] 语义分块...')
from engines.chunking.structure_chunker import StructureChunker
chunks = StructureChunker().chunk(uir)
print(f'  {len(chunks)} chunks')

# 3. 嵌入 + 向量存储
print('\n[3/5] 嵌入 + 向量存储...')
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.vector_store import VectorStore

svc = EmbeddingService()
embs = svc.embed_batch([c.content for c in chunks])
for c, e in zip(chunks, embs):
    c.embedding = e
store = VectorStore()
store.insert(chunks)
print(f'  已存储 {len(chunks)} 条')

# 4. 图谱构建（修复后的关系抽取）
print('\n[4/5] 知识图谱构建...')
from engines.graph_rag.entity_extractor import EntityExtractor, RelationExtractor
from engines.graph_rag.graph_store import GraphStore
from engines.graph_rag.graph_retriever import GraphRAGRetriever

entity_ext = EntityExtractor(llm_url=LLM_URL, llm_model=LLM_MODEL, llm_key=LLM_KEY)
rel_ext = RelationExtractor(llm_url=LLM_URL, llm_model=LLM_MODEL, llm_key=LLM_KEY)
graph = GraphStore()
graph_rag = GraphRAGRetriever(entity_ext, rel_ext, graph)

graph_result = graph_rag.build_from_chunks(chunks)
print(f'  实体: {graph_result["entities"]}')
print(f'  关系: {graph_result["relations"]}')
print(f'  图谱统计: {graph.stats()}')

# 5. 混合检索（向量 + 图谱）
print('\n[5/5] 混合检索 + LLM 回答...')
from engines.retrieval.hybrid_retriever import HybridRetriever
from engines.retrieval.reranker import Reranker

reranker = Reranker(embedder=svc)
rv = HybridRetriever(
    vector_store=store, graph_retriever=graph_rag,
    embedder=svc, reranker=reranker
)

queries = ["系统使用了哪些技术？", "FastAPI依赖哪些组件？", "RAG 2.0的架构是什么？"]
import httpx

for q in queries:
    result = rv.retrieve(q, top_k=5)
    gc = result.get("graph_context", {})
    ents = gc.get("entities", []) if gc else []
    graph_ctx = gc.get("graph_context", []) if gc else []
    docs = result.get("documents", [])

    print(f'\n  Q: {q}')
    print(f'  图谱实体: {ents}')
    print(f'  图谱关系: {len(graph_ctx)} 条')
    for g in graph_ctx[:3]:
        print(f'    {g}')
    print(f'  检索文档: {len(docs)} 条')

    # LLM 生成
    if docs and LLM_KEY:
        context = "\n\n".join(d.get("content", "")[:500] for d in docs[:3])
        if graph_ctx:
            context += "\n\n知识图谱: " + "; ".join(graph_ctx[:5])

        resp = httpx.post(
            f"{LLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_KEY}"},
            json={"model": LLM_MODEL, "messages": [
                {"role": "system", "content": "你是知识库助手，回答简洁，200字以内。"},
                {"role": "user", "content": f"参考文档：\n{context[:2000]}\n\n问题：{q}"}
            ], "max_tokens": 500, "temperature": 0.3},
            timeout=60
        )
        data = resp.json()
        if isinstance(data, str): data = json.loads(data)
        answer = data["choices"][0]["message"]["content"] or "无回答"
        print(f'  回答: {answer[:200]}')

print('\n' + '=' * 60)
print('测试完成 — 图谱增强检索已集成')
print('=' * 60)
