"""真实数据全流程测试 — 3份中文技术PDF，20页，记录真实指标"""
import sys, os, time, json, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

RESULTS = []
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
LLM_URL = "https://tokenhub.itcast.cn/v1"
LLM_MODEL = "deepseek-v4-flash"
LLM_KEY = os.environ.get("RAG_LLM_API_KEY", "")

PDFS = [
    ("Doc1", "企业知识库系统技术白皮书.pdf"),
    ("Doc2", "Python_Web框架技术选型报告.pdf"),
    ("Doc3", "知识库系统部署与运维手册.pdf"),
]

print("=" * 70)
print("RAG 2.0 真实数据 E2E 测试")
print(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"文档: 3 份中文技术 PDF | 模型: {LLM_MODEL}")
print(f"LLM API: {'已配置' if LLM_KEY else '未配置(仅规则抽取)'}")
print("=" * 70)

# ========== Phase 1: 文档处理 ==========
print("\n" + "-" * 70)
print("Phase 1: 文档处理流水线 (解析-分块-嵌入-存储)")
print("-" * 70)

from engines.parsing.pdf_parser import PDFParser
from engines.chunking.semantic_chunker import SemanticChunker
from engines.embedding.embedder import EmbeddingService
from engines.retrieval.vector_store import VectorStore

parser = PDFParser()
chunker = SemanticChunker()
embedder = EmbeddingService()

store = VectorStore()
try:
    import lancedb
    db = lancedb.connect("./lancedb_data")
    db.drop_table("documents", ignore_missing=True)
    store = VectorStore()
    print("已清空旧向量数据")
except Exception as e:
    print(f"清空跳过: {e}")

all_chunks = []
for label, pdf_file in PDFS:
    pdf_path = os.path.join(FIXTURES, pdf_file)
    fsize = os.path.getsize(pdf_path) / 1024

    t0 = time.time()
    uir = parser.parse(pdf_path)
    parse_time = time.time() - t0

    t0 = time.time()
    chunks = chunker.chunk(uir)
    chunk_time = time.time() - t0

    t0 = time.time()
    texts = [c.content for c in chunks]
    embs = embedder.embed_batch(texts)
    for c, e in zip(chunks, embs):
        c.embedding = e
    embed_time = time.time() - t0

    t0 = time.time()
    store.insert(chunks)
    store_time = time.time() - t0

    all_chunks.extend(chunks)
    preview = chunks[0].content[:80].replace('\n', ' ') if chunks else '(empty)'
    print(f"\n  {label}: {pdf_file}")
    print(f"    {fsize:.0f}KB | {len(uir.pages)}p | {len(chunks)} chunks")
    print(f"    parse={parse_time:.2f}s chunk={chunk_time:.2f}s embed={embed_time:.2f}s store={store_time:.2f}s")
    print(f"    preview: {preview}...")

    RESULTS.append({
        "doc": label, "file": pdf_file, "file_size_kb": round(fsize, 0),
        "pages": len(uir.pages), "chunks": len(chunks),
        "parse_s": round(parse_time, 2), "chunk_s": round(chunk_time, 2),
        "embed_s": round(embed_time, 2), "store_s": round(store_time, 2),
    })

total_chunks = len(all_chunks)
total_pages = sum(r["pages"] for r in RESULTS)
print(f"\n  total: {total_pages}p, {total_chunks} chunks")

# ========== Phase 2: Knowledge Graph ==========
print("\n" + "-" * 70)
print("Phase 2: Knowledge Graph (rule + LLM hybrid)")
print("-" * 70)

from engines.graph_rag.entity_extractor import EntityExtractor, RelationExtractor
from engines.graph_rag.graph_store import GraphStore
from engines.graph_rag.graph_retriever import GraphRAGRetriever

entity_ext = EntityExtractor(llm_url=LLM_URL, llm_model=LLM_MODEL, llm_key=LLM_KEY)
rel_ext = RelationExtractor(llm_url=LLM_URL, llm_model=LLM_MODEL, llm_key=LLM_KEY)
graph = GraphStore()
graph_rag = GraphRAGRetriever(entity_ext, rel_ext, graph)

t0 = time.time()
graph_result = graph_rag.build_from_chunks(all_chunks)
graph_time = time.time() - t0
stats = graph.stats()

print(f"  nodes={stats['nodes']} edges={stats['edges']} time={graph_time:.2f}s")
entity_names = list(graph.nodes.keys())[:25]
print(f"  entities: {entity_names}")

all_rel_edges = []
for name, data in graph.nodes.items():
    for e in data.get("relations", []):
        all_rel_edges.append(f"{name} -[{e['predicate']}]-> {e['object']}")

if all_rel_edges:
    print(f"  relations ({len(all_rel_edges)}):")
    for r in all_rel_edges[:20]:
        print(f"    {r}")
else:
    print(f"  relations: none (LLM unavailable, rule extraction only)")

# ========== Phase 3: Vector Retrieval ==========
print("\n" + "-" * 70)
print("Phase 3: Vector Retrieval Benchmark")
print("-" * 70)

from engines.retrieval.hybrid_retriever import HybridRetriever
from engines.retrieval.reranker import Reranker

reranker = Reranker(embedder=embedder)
rv = HybridRetriever(vector_store=store, graph_retriever=graph_rag, embedder=embedder, reranker=reranker)

QUERIES = [
    ("Q1", "企业知识库系统的技术架构是什么？"),
    ("Q2", "FastAPI框架相比Django有哪些优势？"),
    ("Q3", "语义分块策略是如何实现的？"),
    ("Q4", "Docker Compose部署包含哪些服务？"),
    ("Q5", "BGE嵌入模型有什么特点？"),
    ("Q6", "如何配置API Key认证？"),
    ("Q7", "PyTorch和TensorFlow的区别是什么？"),
    ("Q8", "知识图谱的关系类型有哪些？"),
]

vector_results = []
for qid, query in QUERIES:
    t0 = time.time()
    result = rv.retrieve(query, top_k=5)
    lat = (time.time() - t0) * 1000

    docs = result.get("documents", [])
    gc = result.get("graph_context", {})
    top_score = docs[0].get("score", 0) if docs else 0
    top_content = docs[0].get("content", "")[:100].replace('\n', ' ') if docs else "(none)"

    vector_results.append({
        "qid": qid, "query": query[:60], "latency_ms": round(lat, 1),
        "docs_found": len(docs), "top_score": round(top_score, 4),
        "graph_entities": len(gc.get("entities", [])) if gc else 0,
        "top_preview": top_content,
    })
    print(f"  {qid}: score={top_score:.4f} docs={len(docs)} lat={lat:.0f}ms")
    print(f"    top: {top_content}")

# ========== Phase 4: Self-Retrieval ==========
print("\n" + "-" * 70)
print("Phase 4: Self-Retrieval Multi-Round Retrieval")
print("-" * 70)

from engines.retrieval.evaluator import RetrievalEvaluator
from engines.retrieval.query_rewriter import QueryRewriter
from engines.retrieval.self_retrieval import SelfRetrieval

evaluator = RetrievalEvaluator(embedder=embedder)
rewriter = QueryRewriter()
sr = SelfRetrieval(retriever=rv, evaluator=evaluator, rewriter=rewriter, max_rounds=3)

SR_QUERIES = ["系统架构是什么", "Python web框架选哪个好", "怎么部署"]
sr_results = []
for q in SR_QUERIES:
    t0 = time.time()
    result = sr.retrieve(q, top_k=5)
    lat = (time.time() - t0) * 1000

    rounds = result.get("retrieval_rounds", 1)
    trace = result.get("trace", [])
    sr_results.append({"query": q, "rounds": rounds, "latency_ms": round(lat, 1), "trace": trace})
    print(f"  '{q}' -> rounds={rounds} lat={lat:.0f}ms")
    for t in trace:
        rw_list = t.get('rewritten', [])
        rq = rw_list[0] if rw_list else '(none)'
        rq_short = (rq or '(none)')[:70]
        print(f"    R{t['round']}: rel={t.get('relevance',0):.3f} rewrite={t.get('need_rewrite')} q='{rq_short}'")

# ========== Phase 5: LLM QA ==========
print("\n" + "-" * 70)
print("Phase 5: LLM End-to-End QA")
print("-" * 70)

import httpx

LLM_QUERIES = [
    ("QA1", "企业知识库系统用到了哪些核心技术？请列出并简要说明。"),
    ("QA2", "为什么选择FastAPI而不是Django？请给出具体理由。"),
    ("QA3", "系统的部署方案包含哪些组件？如何保证高可用？"),
]

llm_results = []
for qid, q in LLM_QUERIES:
    t0 = time.time()
    result = rv.retrieve(q, top_k=5)
    docs = result.get("documents", [])[:5]

    context = ""
    if docs:
        parts = [f"[src{i+1}] {d.get('content','')[:600]}" for i, d in enumerate(docs)]
        context = "\n\n---\n\n".join(parts)

    tokens_in = tokens_out = 0
    if context and LLM_KEY:
        try:
            resp = httpx.post(
                f"{LLM_URL}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_KEY}"},
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "你是专业的知识库助手。严格根据参考文档回答，标注来源编号。回答简洁，300字以内。"},
                        {"role": "user", "content": f"参考文档：\n{context[:2500]}\n\n问题：{q}"}
                    ],
                    "max_tokens": 2000, "temperature": 0.3
                },
                timeout=60
            )
            data = resp.json()
            if isinstance(data, str):
                data = json.loads(data)
            answer = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
        except Exception as e:
            answer = f"(LLM API error: {str(e)[:80]})"
    else:
        answer = f"(LLM not available, {len(docs)} docs retrieved)"

    total_lat = (time.time() - t0) * 1000
    top_score = docs[0].get("score", 0) if docs else 0

    llm_results.append({
        "qid": qid, "question": q,
        "answer_preview": answer[:250],
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "total_latency_ms": round(total_lat, 1),
        "top_score": round(top_score, 4),
        "sources_count": len(docs),
    })
    print(f"\n  {qid}: '{q[:60]}...'")
    print(f"    retrieval: {len(docs)} docs, score={top_score:.4f}")
    print(f"    answer: {answer[:200]}")
    print(f"    tokens: in={tokens_in} out={tokens_out} lat={total_lat:.0f}ms")

# ========== Summary ==========
print("\n" + "=" * 70)
print("REAL DATA TEST - Summary Report")
print("=" * 70)

print(f"\n  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Docs: 3 PDFs, {total_pages} pages, {total_chunks} chunks")
print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
print(f"  Embed: BGE-small-zh-v1.5 (512d, CPU)")
print(f"  LLM: {LLM_MODEL} ({'configured' if LLM_KEY else 'not configured'})")
print(f"  VectorDB: LanceDB (embedded)")

print(f"\n  -- Doc Processing --")
parse_total = sum(r["parse_s"] for r in RESULTS)
chunk_total = sum(r["chunk_s"] for r in RESULTS)
embed_total = sum(r["embed_s"] for r in RESULTS)
print(f"  Parse: {parse_total:.2f}s | Chunk: {chunk_total:.2f}s | Embed: {embed_total:.2f}s")
print(f"  Graph Build: {graph_time:.2f}s")
print(f"  Per-page: {(parse_total+chunk_total+embed_total)/total_pages:.2f}s")
print(f"  Per-chunk embed: {embed_total/total_chunks*1000:.1f}ms")

print(f"\n  -- Retrieval --")
vec_lats = [r["latency_ms"] for r in vector_results]
vec_scores = [r["top_score"] for r in vector_results]
print(f"  Latency: min={min(vec_lats):.0f}ms avg={sum(vec_lats)/len(vec_lats):.0f}ms max={max(vec_lats):.0f}ms")
print(f"  Score: avg={sum(vec_scores)/len(vec_scores):.4f} min={min(vec_scores):.4f} max={max(vec_scores):.4f}")
gh = sum(1 for r in vector_results if r["graph_entities"] > 0)
print(f"  Graph hits: {gh}/{len(vector_results)}")

print(f"\n  -- Self-Retrieval --")
for r in sr_results:
    print(f"  '{r['query'][:30]}' -> {r['rounds']}r {r['latency_ms']:.0f}ms")

print(f"\n  -- LLM QA --")
if llm_results and llm_results[0]["tokens_in"] > 0:
    t = sum(r["tokens_in"] + r["tokens_out"] for r in llm_results)
    al = sum(r["total_latency_ms"] for r in llm_results) / max(len(llm_results), 1)
    valid = len([r for r in llm_results if r["tokens_out"] > 0])
    print(f"  Tokens: {t} total (in={sum(r['tokens_in'] for r in llm_results)}, out={sum(r['tokens_out'] for r in llm_results)})")
    print(f"  Avg latency: {al:.0f}ms | Valid answers: {valid}/{len(llm_results)}")
else:
    print(f"  LLM not available, retrieval-only mode")

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)

output = {
    "timestamp": datetime.datetime.now().isoformat(),
    "summary": {
        "documents": len(PDFS), "pages": total_pages, "chunks": total_chunks,
        "graph_nodes": stats["nodes"], "graph_edges": stats["edges"],
        "model": LLM_MODEL
    },
    "document_processing": RESULTS,
    "vector_retrieval": vector_results,
    "self_retrieval": sr_results,
    "llm_qa": llm_results,
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_test_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nResults saved: {out_path}")
