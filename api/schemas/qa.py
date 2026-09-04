"""知识问答相关模型"""

from pydantic import BaseModel, Field, model_validator

from api.config import settings
from engines.router.routing_rules import SKILLS

# 可手动指定的技能 = 路由表全部键 (policy/contract/product/service/tech/finance/hr/
# marketing/meeting/training/code/direct)。原先硬编码仅 service|tech|direct, 与路由器
# 实际支持的 11 个技能不一致: 调用方无法手动指定 policy/product/finance 等库,
# 离线评测也无法把"库内检索召回"与"路由准确率"拆开测量。
# 由 SKILLS 动态生成, 新增文档类型时无需再改此处。
_SKILL_PATTERN = r"^(" + "|".join(sorted(SKILLS)) + r")$"


class ChatTurn(BaseModel):
    """多轮对话单轮: 由客户端维护并随请求回传。role 限定 user/assistant。"""

    role: str = Field(pattern=r"^(user|assistant)$", description="发言方: user(用户) / assistant(系统)")
    content: str = Field(min_length=1, max_length=4000, description="该轮文本内容")


class QARequest(BaseModel):
    question: str = Field(default="", max_length=2000, description="自然语言问题 (纯图片提问时可为空, 由 OCR 提取文字)")

    @model_validator(mode="after")
    def _question_or_image(self):
        """空问题仅在有图片输入时合法 (纯图片提问)。"""
        if not self.question.strip() and not self.image_base64:
            raise ValueError("question 不能为空 (除非携带 image_base64 纯图片提问)")
        return self
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
        default=None, pattern=_SKILL_PATTERN, description="手动指定技能，跳过 Router 路由"
    )
    history: list[ChatTurn] = Field(
        default_factory=list,
        description="多轮对话历史(客户端维护, 服务端无状态透传), 用于上下文连贯。最多取最近 20 轮",
    )
    session_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="多轮会话 ID: 携带后服务端按 session 维护历史(刷新/换设备不丢), 覆盖 history 字段, TTL 30min",
    )
    # ── 多 Agent 框架扩展 (全部带默认值, 旧客户端零破坏) ──
    image_base64: str | None = Field(
        default=None,
        description="图片输入(纯 base64, 不含 data: 前缀): OCR 提取文字后并入问题。需安装 rapidocr, 失败返回 400",
    )
    confirm_operation: bool = Field(
        default=False, description="操作 Agent 二次确认标志: confirm=true + pending_operation_id 执行高危操作"
    )
    pending_operation_id: str | None = Field(
        default=None, max_length=32, description="操作 Agent 待确认操作 ID (来自上一轮响应 pending_operation.pending_id)"
    )
    force_agent: str | None = Field(
        default=None,
        pattern=r"^(consult|chat|complaint|operation)$",
        description="指定执行 Agent (前端切换条): consult=强制走 RAG 检索, 其余直接分发到对应 Agent; 不传=自动意图分流",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "系统使用了哪些技术？",
                "mode": "hybrid",
                "enable_self_retrieval": False,
                "top_k": 10,
                "history": [
                    {"role": "user", "content": "退款多久到账"},
                    {"role": "assistant", "content": "一般一到三个工作日到账"},
                ],
            }
        }
    }


class SourceDocument(BaseModel):
    id: str = ""
    chunk_id: str = ""
    doc_id: str = ""
    content: str = ""
    score: float = 0.0
    # 命中该文档的**全部** chunk id (父子文档机制下, 一条来源按 doc_id 合并了多个子块)。
    # 评测端计算 recall@k 需要精确 chunk 级匹配, 仅靠 doc_id 无法判断具体块是否命中,
    # 而 choreographer 侧的 chunk_id(单数)在分组合并后已无意义 —— 故单独透传列表。
    chunk_ids: list[str] = Field(default_factory=list)


def to_source_document(d: dict) -> "SourceDocument":
    """内部 source dict → API 模型。集中字段映射, 新增字段只改本函数与 SourceDocument 定义。

    与 SSE 路径(public_sources → build_sources_event 直接发 dict)解耦: HTTP 响应经此映射,
    SSE 响应不经 SourceDocument, 前端契约不受影响。score 强制 float 防止 None/字符串污染响应。
    """
    return SourceDocument(
        id=d.get("id", ""),
        chunk_id=d.get("chunk_id", ""),
        doc_id=d.get("doc_id", ""),
        content=d.get("content", ""),
        score=float(d.get("score", 0.0) or 0.0),
        chunk_ids=list(d.get("chunk_ids") or []),
    )


class RefusalCandidate(BaseModel):
    """拒答时附带的「检索命中但低于阈值」候选来源 —— 把死胡同变岔路口。

    doc_id + kb 是 U2「来源展开全文」端点的查询键: 前端点击候选即按
    (doc_id, kb) 拉取该文档全部 chunk 内容。chunk_ids 给出具体命中块,
    便于 U2 在全文内高亮定位。title 为展示用标题(标题链/文档标题/文件名)。
    """

    doc_id: str = Field(default="", description="命中文档 ID — U2 展开全文的查询键")
    kb: str = Field(default="", description="所属知识库 — U2 端点做 KB 级 RBAC 的依据")
    title: str = Field(default="", description="展示标题: 标题链 或 文档标题 或 文件名")
    score: float = Field(default=0.0, description="该文档最佳片段相关度")
    chunk_ids: list[str] = Field(default_factory=list, description="本次命中的全部 chunk id")


class RefusalInfo(BaseModel):
    """拒答分级元数据。answer 字段仍保留拒答文本(向后兼容), 本结构为附加的可机读信号。

    reason 枚举:
      - low_confidence: 最佳片段相关度低于置信度下限(answer_confidence_floor)
      - low_fidelity:   答案与上下文重合低于忠实度阈值(fidelity_threshold)
      - no_docs:        未检索到任何内容, 且无 LLM 兜底
    candidates 为空表示确实无任何相关文档(真拒答); 非空表示「检索命中但被护栏拦截」,
    此时附候选 + 引导追问可显著挽回体验(实测 7/8 误拒属此类)。
    """

    is_refusal: bool = Field(default=True, description="恒为 True, 便于前端快速判断")
    reason: str = Field(default="", description="拒答原因枚举: low_confidence/low_fidelity/no_docs")
    candidates: list[RefusalCandidate] = Field(default_factory=list, description="检索命中但被拦截的候选来源")
    suggested_questions: list[str] = Field(default_factory=list, description="引导追问提示(确定性生成, 不调用 LLM)")


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
    # 拒答分级: 仅当答案确为拒答时非空。前端据此渲染候选来源 + 引导追问, 替代纯文本死胡同。
    refusal: RefusalInfo | None = Field(default=None, description="拒答分级元数据(附候选来源/引导追问), 非拒答时为空")
    # ── 多 Agent 框架扩展 (全部带默认值, 旧客户端零破坏) ──
    message_type: str = Field(
        default="consult", description="意图分类: consult(咨询)/chat(闲聊)/complaint(投诉)/operation(操作)"
    )
    agent: str = Field(default="rag", description="执行 Agent: rag/chitchat/complaint/operation")
    ticket: dict | None = Field(default=None, description="投诉工单信息 (仅投诉 Agent 返回)")
    pending_operation: dict | None = Field(
        default=None, description="待确认高危操作信息 (仅操作 Agent 高危分支返回)"
    )
    memory_used: list[dict] = Field(
        default_factory=list, description="本轮命中的长期记忆条目 (question/answer/score)"
    )
