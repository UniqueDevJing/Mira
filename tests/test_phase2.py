"""Phase 2 测试 — 实体抽取 + 图谱构建 + 多跳推理 + 混合检索"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print('=' * 60)
print('RAG 2.0 Phase 2 测试')
print('=' * 60)

# Config
LLM_URL = "https://tokenhub.itcast.cn/v1"
LLM_MODEL = "deepseek-v4-flash"
LLM_KEY = os.environ.get("RAG_LLM_API_KEY", "")

# 1. PDF 解析
print('\n[1] 文档解析...')
from engines.parsing.pdf_parser import PDFParser
uir = PDFParser().parse('tests/fixtures/sample.pdf')
print(f'    {len(uir.pages)} 页')

# 2. 语义分块
print('\n[2] 语义分块...')
from engines.chunking.structure_chunker import StructureChunker
chunks = StructureChunker().chunk(uir)
print(f'    {len(chunks)} chunks')

# 3. 实体抽取 + 图谱构建
print('\n[3] 实体抽取 + 知识图谱构建...')
from engines.graph_rag.entity_extractor import EntityExtractor, RelationExtractor
from engines.graph_rag.graph_store import GraphStore
from engines.graph_rag.graph_retriever import GraphRAGRetriever

entity_ext = EntityExtractor(llm_url=LLM_URL, llm_model=LLM_MODEL, llm_key=LLM_KEY)
rel_ext = RelationExtractor(llm_url=LLM_URL, llm_model=LLM_MODEL, llm_key=LLM_KEY)
graph = GraphStore()
graph_rag = GraphRAGRetriever(entity_ext, rel_ext, graph)

result = graph_rag.build_from_chunks(chunks)
print(f'    实体: {result["entities"]}')
print(f'    关系: {result["relations"]}')
print(f'    图谱统计: {graph.stats()}')

# 4. 向量嵌入 + 存储
print('\n[4] 向量嵌入 + 存储...')
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.vector_store import VectorStore

svc = EmbeddingService()
embs = svc.embed_batch([c.content for c in chunks])
for c, e in zip(chunks, embs):
    c.embedding = e
store = VectorStore()
store.insert(chunks)
print(f'    已存储 {len(chunks)} 条')

# 5. 混合检索（向量 + 图谱）
print('\n[5] 混合检索（向量 + 图谱）...')
from engines.retrieval.hybrid_retriever import HybridRetriever
rv = HybridRetriever(vector_store=store, graph_retriever=graph_rag, embedder=svc)

queries = [
    "系统使用了哪些技术？",
    "FastAPI依赖哪些组件？",
    "DeepSeek模型如何被调用？"
]
for q in queries:
    result = rv.retrieve(q, top_k=5)
    gc = result.get("graph_context", {})
    ents = gc.get("entities", []) if gc else []
    graph_ctx = gc.get("graph_context", []) if gc else []
    docs = result.get("documents", [])
    print(f'\n  Q: {q}')
    print(f'    图谱实体: {ents}')
    print(f'    图谱关系: {len(graph_ctx)} 条')
    print(f'    检索文档: {len(docs)} 条')

# 6. LLM 回答（带图谱上下文）
print('\n[6] LLM 答案生成（带图谱增强）...')
import httpx

q = "系统架构是什么？使用了哪些技术？"
result = rv.retrieve(q, top_k=5)
gc = result.get("graph_context", {})

context_parts = []
for d in result.get("documents", [])[:3]:
    context_parts.append(d.get("content", ""))
if gc and gc.get("graph_context"):
    context_parts.append("\n知识图谱关系:\n" + "\n".join(gc["graph_context"]))

context = "\n\n".join(context_parts)

resp = httpx.post(
    f"{LLM_URL}/chat/completions",
    headers={"Authorization": f"Bearer {LLM_KEY}"},
    json={"model": LLM_MODEL, "messages": [
        {"role": "system", "content": "你是知识库助手，回答简洁，200字以内。"},
        {"role": "user", "content": f"参考文档：\n{context[:3000]}\n\n问题：{q}"}
    ], "max_tokens": 2000, "temperature": 0.3},
    timeout=60
)
data = resp.json()
if isinstance(data, str): data = json.loads(data)
answer = data["choices"][0]["message"]["content"]
if not answer:
    answer = data["choices"][0]["message"].get("reasoning_content", "")[:300]

print(f'    问题: {q}')
print(f'    图谱增强回答:\n    {answer}')
print(f'    Token: {data.get("usage", {})}')

# 7. 多跳推理测试
print('\n[7] 多跳推理...')
if graph.stats()["nodes"] > 0:
    # 找第一个实体做多跳
    for name in list(graph.nodes.keys())[:2]:
        hops = graph.multi_hop(name, ["uses", "depends_on", "contains"])
        print(f'    {name}: {len(hops)} 跳关系')
        for h in hops[:3]:
            print(f'      {h["from"]} --[{h["relation"]}]--> {h["to"]}')
else:
    print('    (图谱为空，跳过)')

print('\n' + '=' * 60)
print('Phase 2 测试完成')
print('=' * 60)
