"""知识问答 API"""
import time, httpx
from typing import Optional, List
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])


class QARequest(BaseModel):
    question: str
    mode: str = "hybrid"
    enable_self_retrieval: bool = True
    top_k: int = 10
    filters: Optional[dict] = None


RAG_SYSTEM_PROMPT = """你是专业的知识库助手。严格根据以下参考文档回答用户问题。
- 有明确答案：直接引用并标注来源
- 部分相关：说明信息不完整
- 无相关信息：如实说明"知识库中暂无相关内容"
- 回答简洁，控制在 300 字以内"""


@router.post("/ask")
async def ask_question(req: QARequest):
    start = time.time()

    # 构建检索流水线
    from engines.embedding.embedder import EmbeddingService
    from engines.retrieval.vector_store import VectorStore
    from engines.retrieval.hybrid_retriever import HybridRetriever
    from engines.retrieval.evaluator import RetrievalEvaluator
    from engines.retrieval.query_rewriter import QueryRewriter
    from engines.retrieval.self_retrieval import SelfRetrieval

    embedder = EmbeddingService()
    vector_store = VectorStore()
    retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
    evaluator = RetrievalEvaluator(embedder=embedder)
    rewriter = QueryRewriter()

    if req.enable_self_retrieval:
        sr = SelfRetrieval(retriever=retriever, evaluator=evaluator, rewriter=rewriter)
        result = sr.retrieve(req.question, top_k=req.top_k)
    else:
        result = retriever.retrieve(req.question, top_k=req.top_k)
        result["retrieval_rounds"] = 1

    docs = result.get("documents", [])[:req.top_k]

    # LLM 生成答案
    from api.config import settings
    context = "\n\n---\n\n".join(
        f"[来源{i+1}] {d.get('content', '')[:800]}" for i, d in enumerate(docs[:5])
    ) if docs else ""

    if context:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{settings.llm_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                    json={
                        "model": settings.llm_model,
                        "messages": [
                            {"role": "system", "content": RAG_SYSTEM_PROMPT},
                            {"role": "user", "content": f"参考文档：\n{context}\n\n问题：{req.question}"}
                        ],
                        "max_tokens": 2000, "temperature": 0.3
                    }
                )
            data = resp.json()
            if isinstance(data, str):
                import json
                data = json.loads(data)
            answer = data["choices"][0]["message"]["content"]
            if not answer:
                answer = "（推理中）增加 max_tokens 后重试"
        except Exception as e:
            answer = f"（LLM 暂时不可用）检索到 {len(docs)} 条相关内容。错误: {str(e)[:100]}"
    else:
        answer = "未在知识库中找到相关信息，请先上传文档。"

    latency_ms = (time.time() - start) * 1000

    return {
        "answer": answer,
        "sources": docs[:req.top_k],
        "graph_context": result.get("graph_context"),
        "retrieval_rounds": result.get("retrieval_rounds", 1),
        "rewritten_queries": [],
        "latency_ms": round(latency_ms, 2),
    }
