"""知识问答 API"""
import time
import logging
from fastapi import APIRouter

from api.schemas.qa import QARequest, QAResponse, SourceDocument, TokenUsage
from api.core.llm_client import get_llm_client, CircuitBreakerOpenError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/qa", tags=["qa"])

RAG_SYSTEM_PROMPT = """你是专业的知识库助手。严格根据以下参考文档回答用户问题。
- 有明确答案：直接引用并标注来源
- 部分相关：说明信息不完整
- 无相关信息：如实说明"知识库中暂无相关内容"
- 回答简洁，控制在 300 字以内"""


@router.post("/ask", response_model=QAResponse)
async def ask_question(req: QARequest):
    start = time.time()

    # 构建检索流水线
    from engines.embedding.embedder import EmbeddingService
    from engines.retrieval.hybrid_retriever import HybridRetriever
    from engines.retrieval.reranker import Reranker
    from engines.retrieval.evaluator import RetrievalEvaluator
    from engines.retrieval.query_rewriter import QueryRewriter
    from engines.retrieval.self_retrieval import SelfRetrieval
    from api.state import get_graph_rag, get_vector_store

    embedder = EmbeddingService()
    vector_store = get_vector_store()
    reranker = Reranker(embedder=embedder)
    graph_rag = get_graph_rag()

    # 根据 mode 构建检索器
    if req.mode == "vector":
        from engines.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(
            vector_store=vector_store, graph_retriever=None,
            embedder=embedder, reranker=reranker,
        )
    elif req.mode == "graph":
        from engines.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(
            vector_store=None, graph_retriever=graph_rag,
            embedder=embedder, reranker=reranker,
        )
    else:  # hybrid
        retriever = HybridRetriever(
            vector_store=vector_store, graph_retriever=graph_rag,
            embedder=embedder, reranker=reranker,
        )

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
    context = "\n\n---\n\n".join(
        f"[来源{i+1}] {d.get('content', '')[:800]}" for i, d in enumerate(docs[:5])
    ) if docs else ""

    token_usage = None
    if context:
        try:
            llm_client = get_llm_client()
            llm_response = await llm_client.chat(
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": f"参考文档：\n{context}\n\n问题：{req.question}"}
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            answer = llm_response.content
            if not answer:
                answer = "（推理中）增加 max_tokens 后重试"

            token_usage = TokenUsage(
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                total_tokens=llm_response.total_tokens,
                llm_latency_ms=llm_response.latency_ms,
            )

            logger.info(
                "QA 生成完成: tokens_in=%d, tokens_out=%d, total=%d, latency=%.1fms",
                llm_response.prompt_tokens, llm_response.completion_tokens,
                llm_response.total_tokens, llm_response.latency_ms
            )
        except CircuitBreakerOpenError:
            answer = f"（LLM 服务暂时不可用）检索到 {len(docs)} 条相关内容。"
            logger.warning("QA 降级: LLM 熔断中")
        except Exception as e:
            answer = f"（LLM 暂时不可用）检索到 {len(docs)} 条相关内容。"
            logger.error("QA LLM 调用失败: %s", str(e)[:200])
    else:
        answer = "未在知识库中找到相关信息，请先上传文档。"

    latency_ms = (time.time() - start) * 1000

    return QAResponse(
        answer=answer,
        sources=[SourceDocument(
            id=d.get("id", ""), chunk_id=d.get("chunk_id", ""),
            doc_id=d.get("doc_id", ""), content=d.get("content", ""),
            score=d.get("score", 0.0),
        ) for d in docs[:req.top_k]],
        graph_context=result.get("graph_context"),
        retrieval_rounds=result.get("retrieval_rounds", 1),
        rewritten_queries=result.get("rewritten_queries", []),
        latency_ms=round(latency_ms, 2),
        token_usage=token_usage,
    )
