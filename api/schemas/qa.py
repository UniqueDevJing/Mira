"""知识问答相关模型"""
from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="自然语言问题")
    mode: str = Field(default="hybrid", pattern=r"^(hybrid|vector|graph)$",
                      description="检索模式: hybrid(向量+图谱) / vector(纯向量) / graph(纯图谱)")
    enable_self_retrieval: bool = Field(default=True, description="启用多轮自适应检索")
    top_k: int = Field(default=10, ge=1, le=50, description="返回文档数量")
    filters: dict | None = Field(default=None, description="按字段过滤: {\"department\": \"技术部\"}")

    model_config = {"json_schema_extra": {
        "example": {
            "question": "系统使用了哪些技术？",
            "mode": "hybrid",
            "enable_self_retrieval": True,
            "top_k": 10,
            "filters": None,
        }
    }}


class SourceDocument(BaseModel):
    id: str = ""
    chunk_id: str = ""
    doc_id: str = ""
    content: str = ""
    score: float = 0.0


class GraphContext(BaseModel):
    entities: list[str] = Field(default_factory=list)
    relations: list[dict] = Field(default_factory=list)
    graph_context: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Token 消耗统计"""
    prompt_tokens: int = Field(default=0, description="输入 token 数")
    completion_tokens: int = Field(default=0, description="输出 token 数")
    total_tokens: int = Field(default=0, description="总 token 数")
    llm_latency_ms: float = Field(default=0.0, description="LLM 调用延迟 (ms)")


class QAResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)
    graph_context: GraphContext | None = None
    retrieval_rounds: int = 1
    rewritten_queries: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: TokenUsage | None = Field(default=None, description="Token 消耗统计")
