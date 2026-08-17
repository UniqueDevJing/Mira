"""知识问答相关模型"""

from pydantic import BaseModel, Field

from api.config import settings


class ChatTurn(BaseModel):
    """多轮对话单轮: 由客户端维护并随请求回传。role 限定 user/assistant。"""

    role: str = Field(pattern=r"^(user|assistant)$", description="发言方: user(用户) / assistant(系统)")
    content: str = Field(min_length=1, max_length=4000, description="该轮文本内容")


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="自然语言问题")
    mode: str = Field(
        default="hybrid",
        pattern=r"^(hybrid|vector|graph)$",
        description="检索模式: hybrid(向量+图谱) / vector(纯向量) / graph(纯图谱)",
    )
    enable_self_retrieval: bool = Field(
        default=settings.enable_self_retrieval, description="启用多轮自适应检索 (默认关, 质量敏感场景显式开启)"
    )
    temperature: float = Field(default=0.1, ge=0, le=2, description="LLM 采样温度, 越低越保守(防幻觉)")
    top_k: int = Field(default=10, ge=1, le=50, description="返回文档数量")
    skill: str | None = Field(
        default=None, pattern=r"^(service|tech|direct)$", description="手动指定技能，跳过 Router 路由"
    )
    history: list[ChatTurn] = Field(
        default_factory=list,
        description="多轮对话历史(客户端维护, 服务端无状态透传), 用于上下文连贯。最多取最近 20 轮",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "系统使用了哪些技术？",
                "mode": "hybrid",
                "enable_self_retrieval": False,
                "top_k": 10,
                "history": [{"role": "user", "content": "退款多久到账"}, {"role": "assistant", "content": "一般一到三个工作日到账"}],
            }
        }
    }


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
    qa_metrics: dict = Field(
        default_factory=dict, description="运行时评估近似指标: faithfulness/retrieval_relevance/confidence"
    )
    latency_ms: float = 0.0
    token_usage: TokenUsage | None = Field(default=None, description="Token 消耗统计")
    # Router / Skill 路由信息
    skill: str = Field(default="", description="命中的技能: service/tech/direct")
    kb_id: str | None = Field(default=None, description="命中的知识库")
    routing_source: str | None = Field(default=None, description="路由来源: rule/llm/fallback/manual")
    degradation_level: int = Field(default=0, description="降级等级: 0正常/1跳过rerank/2仅BM25/3LLM失败")
    latency_breakdown: dict = Field(default_factory=dict, description="各环节耗时 (ms)")
    retrieval_meta: dict = Field(default_factory=dict, description="检索元信息: top1_score/result_count/cross_kb")
