"""企业文档类型注册表 — 单一事实来源 (Single Source of Truth)。

上传(选类型→落库)、路由(问题分类→选库)、切分(类型化策略)、前端(类型下拉)
全部以此为据。新增/调整文档类型只改本文件, 无需改动业务代码。

每个文档类型(DocTypeSpec)定义:
  - type_id / label / kb / description
  - routing_keywords : 路由关键词 (词, 权重) — 汇总进 SKILL_RULES
  - prompt_hint      : 检索/生成时强调的文档性质 (拼入 system prompt)
  - chunker          : 切分策略 {"strategy", "max_chars", "overlap"}
  - parse            : 解析增强 {"ocr", "table"}
"""

from dataclasses import dataclass, field


@dataclass
class DocTypeSpec:
    type_id: str
    label: str
    kb: str | None               # None = 不走知识库(如 direct)
    description: str
    routing_keywords: list[tuple[str, float]] = field(default_factory=list)
    prompt_hint: str = ""
    chunker: dict = field(default_factory=dict)   # {"strategy": "semantic"|"faq"|"clause", "max_chars":int, "overlap":int}
    parse: dict = field(default_factory=dict)     # {"ocr": bool, "table": bool}


# ───────────────────────── 10 类企业文档 + direct ─────────────────────────
DOC_TYPES: dict[str, DocTypeSpec] = {
    "policy": DocTypeSpec(
        type_id="policy", label="制度规范", kb="policy",
        description="规章制度 / SOP / 管理办法 / 准则",
        # 退换货类关键词: 退换货的"制度依据"(能否退/期限/退款比例/七天无理由)属制度规范,
        # 与 service(客服话术: 操作步骤/到账时效)互补而非互斥。同一主题两库同时命中是
        # 有意为之 —— 由 P1' 扇出并行检索两库, 避免单路漏掉答案实际所在的库。
        # 不扩张到 物流/订单/发票/运费 等纯客服领域词, 防止制度库语义泛化。
        routing_keywords=[("制度", .9), ("规定", .9), ("规范", .85), ("流程", .8), ("SOP", .9),
                          ("章程", .85), ("办法", .8), ("条例", .8), ("准则", .8), ("实施细则", .7),
                          ("退货", .9), ("退款", .9), ("退换货", .9), ("七天无理由", .9),
                          ("赔付", .85)],
        prompt_hint="这是制度规范类文档，回答须引用具体条款/编号，措辞严谨，不擅自发挥。",
        chunker={"strategy": "semantic", "max_chars": 480, "overlap": 80},
        # 制度类文档常含奖惩标准表 / 流程时限表, 此前整体丢弃
        parse={"table": True},
    ),
    "contract": DocTypeSpec(
        type_id="contract", label="合同协议", kb="contract",
        description="合同 / 协议 / 法务条款",
        routing_keywords=[("合同", .9), ("协议", .9), ("条款", .9), ("甲方", .8), ("乙方", .8),
                          ("违约", .9), ("赔偿", .85), ("保密", .85), ("知识产权", .8),
                          ("争议解决", .85), ("签署", .7), ("生效", .6)],
        prompt_hint="这是合同/法务类文档，回答须定位具体条款编号，明确双方权利义务，不臆测未约定事项。",
        chunker={"strategy": "clause", "max_chars": 480, "overlap": 80},
        parse={"table": True},
    ),
    "product": DocTypeSpec(
        type_id="product", label="产品手册", kb="product",
        description="产品说明书 / 操作手册 / 规格参数",
        routing_keywords=[("产品", .85), ("功能", .8), ("使用", .8), ("说明", .75), ("操作", .8),
                          ("手册", .85), ("参数", .8), ("规格", .8), ("型号", .8), ("配置", .7),
                          ("安装", .6), ("步骤", .7)],
        prompt_hint="这是产品手册类文档，回答结合功能说明与操作步骤，必要时列要点。",
        chunker={"strategy": "semantic", "max_chars": 480, "overlap": 80},
        # 规格参数表是产品手册的核心内容, 此前整体丢弃
        parse={"table": True},
    ),
    "service": DocTypeSpec(
        type_id="service", label="客服话术", kb="service",
        description="客服话术 / FAQ / 售后问答",
        routing_keywords=[("退款", .9), ("退货", .9), ("物流", .9), ("订单", .9), ("售后", .9),
                          ("客服", .9), ("发票", .9), ("发货", .9), ("运费", .9), ("签收", .9),
                          ("投诉", .9), ("赔偿", .85), ("优惠", .7), ("价格", .7), ("包邮", .7),
                          # 退货/退款的口语化说法 ("能退吗"/"能退回来") 不含 "退货/退款" 子串,
                          # 纯规则易漏判 → 补高特异词, 与 policy 的 退货/退款(.9) 互补兜底
                          ("能退", .8)],
        prompt_hint="这是客服场景，回答需友好、准确、可执行，优先给出明确处理步骤与政策依据。",
        chunker={"strategy": "faq", "max_chars": 480, "overlap": 60},
        # 运费表 / 时效表 / 赔付标准表是客服高频查询对象
        parse={"table": True},
    ),
    "tech": DocTypeSpec(
        type_id="tech", label="技术文档", kb="tech",
        description="技术文档 / API / 架构 / 部署",
        routing_keywords=[("部署", .9), ("架构", .9), ("API", .9), ("接口", .9), ("数据库", .9),
                          ("缓存", .9), ("安装", .9), ("配置", .9), ("版本", .7), ("环境", .7),
                          ("报错", .7), ("FastAPI", .9), ("Docker", .9), ("系统", .6), ("技术", .6),
                          ("文档", .5),
                          # 数据归属修正后 c3dbffff6a960cc1(技术栈/上传格式) 已迁入 tech, 补词使其稳定命中:
                          # "上传格式" 让「支持上传哪些格式」这类问题归 tech 而非误入 service;
                          # "嵌入模型"/"语义搜索" 让检索架构类问题归 tech, 压过 product 的「功能」误抢
                          ("上传格式", .8), ("嵌入模型", .85), ("语义搜索", .85)],
        prompt_hint="这是技术文档，回答可含代码示例、命令与配置片段，确保准确可复现。",
        chunker={"strategy": "semantic", "max_chars": 480, "overlap": 80},
        # API 参数表 / 配置项表是技术文档的高频查询目标
        parse={"table": True},
    ),
    "finance": DocTypeSpec(
        type_id="finance", label="财务", kb="finance",
        description="财务 / 报销 / 预算 / 税务",
        routing_keywords=[("财务", .9), ("报销", .9), ("预算", .85), ("发票", .8), ("税务", .85),
                          ("成本", .8), ("核算", .8), ("审计", .8), ("资金", .8), ("会计", .8),
                          ("利润", .7), ("科目", .7)],
        prompt_hint="这是财务类文档，回答关注金额、科目、流程与合规，数字须与原文一致。",
        chunker={"strategy": "semantic", "max_chars": 480, "overlap": 80},
        # 报销标准表 / 科目表: prompt_hint 要求"数字须与原文一致", 而表格此前整体丢弃
        parse={"table": True},
    ),
    "hr": DocTypeSpec(
        type_id="hr", label="人事", kb="hr",
        description="人事 / 招聘 / 薪酬 / 考勤",
        routing_keywords=[("招聘", .85), ("面试", .8), ("入职", .8), ("离职", .8), ("薪酬", .85),
                          ("绩效", .8), ("考勤", .8), ("社保", .8), ("公积金", .8), ("员工", .7),
                          ("假期", .8), ("培训", .6)],
        prompt_hint="这是人事制度类文档，回答关注流程、权益与时效，措辞清晰。",
        chunker={"strategy": "semantic", "max_chars": 480, "overlap": 80},
        # 薪资表 / 考勤表 / 编制表是人事高频查询对象, 此前整体丢弃
        parse={"table": True},
    ),
    "marketing": DocTypeSpec(
        type_id="marketing", label="营销", kb="marketing",
        description="营销 / 市场 / 品牌 / 推广",
        routing_keywords=[("营销", .9), ("推广", .85), ("活动", .8), ("品牌", .85), ("广告", .85),
                          ("投放", .8), ("转化率", .8), ("用户增长", .8), ("市场", .7),
                          ("渠道", .8), ("私域", .8)],
        prompt_hint="这是营销类文档，回答关注策略、人群、指标与落地动作。",
        chunker={"strategy": "semantic", "max_chars": 480, "overlap": 80},
        # 投放指标表 / ROI 表 / 人群分布表是营销文档核心, 此前整体丢弃
        parse={"table": True},
    ),
    "meeting": DocTypeSpec(
        type_id="meeting", label="会议纪要", kb="meeting",
        description="会议纪要 / 内部通讯 / 决议",
        routing_keywords=[("会议", .85), ("纪要", .9), ("决议", .85), ("讨论", .8), ("议题", .85),
                          ("复盘", .8), ("周会", .8), ("月会", .8), ("同步", .7), ("对齐", .7),
                          ("待办", .8)],
        prompt_hint="这是会议纪要类文档，回答聚焦结论、决议与待办事项，引用时间与责任人。",
        chunker={"strategy": "semantic", "max_chars": 480, "overlap": 80},
        parse={},
    ),
    "training": DocTypeSpec(
        type_id="training", label="培训课件", kb="training",
        description="培训 / 课件 / 课程 / 认证",
        routing_keywords=[("培训", .9), ("课件", .9), ("课程", .85), ("学习", .8), ("考试", .8),
                          ("认证", .8), ("讲义", .85), ("教材", .85), ("习题", .8)],
        prompt_hint="这是培训资料类文档，回答围绕知识点、学习路径与考核要点。",
        chunker={"strategy": "semantic", "max_chars": 480, "overlap": 80},
        # 课表 / 成绩表 / 考核权重表是培训文档常见结构, 此前整体丢弃
        parse={"table": True},
    ),
    "code": DocTypeSpec(
        type_id="code", label="代码仓库", kb="tech",
        description="源代码 / 脚本 / 配置 / API 实现（按函数/类/模块整块切分）",
        routing_keywords=[("代码", .9), ("函数", .85), ("类", .8), ("模块", .8), ("脚本", .8),
                          ("def", .7), ("class", .7), ("python", .8), ("接口实现", .7),
                          ("配置", .6), ("实现", .6)],
        prompt_hint="这是代码类文档，回答可引用具体函数/类与实现逻辑，保持代码准确可复现。",
        chunker={"strategy": "code", "max_chars": 800, "overlap": 80},
        parse={},
    ),
    "direct": DocTypeSpec(
        type_id="direct", label="直接回答", kb=None,
        description="寒暄 / 闲聊 / 与知识库无关",
        routing_keywords=[("你好", .95), ("您好", .95), ("hello", .95), ("hi", .95), ("谢谢", .95),
                          ("感谢", .95), ("再见", .95), ("在吗", .95), ("你是谁", .95), ("你能做什么", .95)],
        prompt_hint="",
        chunker={},
        parse={},
    ),
}


# ───────────────────────── 派生视图 (供路由/上传/前端使用) ─────────────────────────
# 走知识库的文档类型 (排除 direct)
RAG_DOC_TYPES: dict[str, DocTypeSpec] = {k: v for k, v in DOC_TYPES.items() if v.kb}
# 所有检索候选库 (即各 RAG 类型的 kb)
RAG_KBS: list[str] = [spec.kb for spec in RAG_DOC_TYPES.values()]
# 类型 → kb 映射
TYPE_TO_KB: dict[str, str] = {k: v.kb for k, v in RAG_DOC_TYPES.items()}
# kb → 类型 反查 (检索结果/路由 kb 还原类型)
KB_TO_TYPE: dict[str, str] = {v.kb: k for k, v in RAG_DOC_TYPES.items()}


def get_doc_type(type_id: str | None) -> DocTypeSpec:
    """按 id 取类型规格; 非法/None 回退 policy(最通用制度库, 兜底入库不落孤儿库)。"""
    if type_id and type_id in DOC_TYPES:
        return DOC_TYPES[type_id]
    return DOC_TYPES["policy"]


def kb_to_doc_type(kb: str | None) -> DocTypeSpec | None:
    """kb 反查类型 (检索结果/路由 kb 还原类型, 供 prompt 拼 hint)。"""
    if not kb:
        return None
    t = KB_TO_TYPE.get(kb)
    return DOC_TYPES.get(t) if t else None


def resolve_doc_type(doc_type: str | None, knowledge_base: str | None = None) -> str:
    """上传时确定文档类型: 优先 doc_type, 其次 kb 反查, 旧 'documents'/非法回退 policy。

    确保最终 kb 一定落在 RAG_KBS 内 (修复旧默认库 'documents' 孤儿库 → 检索不到的 bug)。
    """
    if doc_type and doc_type in DOC_TYPES:
        return doc_type
    if knowledge_base and knowledge_base in KB_TO_TYPE:
        return KB_TO_TYPE[knowledge_base]
    return "policy"


# ─────────────────── 切分策略释义 (前端策略可视化用) ───────────────────
# 与 engines/chunking/strategies.py 的实现一一对应; 新增策略时须同步此处。
CHUNK_STRATEGY_META: dict[str, dict] = {
    "semantic": {
        "label": "结构感知语义切分",
        "cut_unit": "语义段落",
        "rationale": "按标题/段落结构聚合至 max_chars，保留完整语义单元，避免跨主题拼接",
    },
    "faq": {
        "label": "问答对切分",
        "cut_unit": "一个问答对",
        "rationale": "以「问/答」配对为单位独立成块，命中即返回完整答案；无问答结构时自动退化语义切分",
    },
    "clause": {
        "label": "条款级切分",
        "cut_unit": "一条条款",
        "rationale": "按 第X条/第X章/1.1/一、 等条款边界切分并保留条款号，避免条款被拦腰截断",
    },
    "code": {
        "label": "代码整块切分",
        "cut_unit": "一个函数/类/模块",
        "rationale": "代码按函数/类/模块边界整块切分（不从中段截断），并建父子文档：子块精确召回、父块大上下文回给 LLM",
    },
}


def doc_type_list() -> list[dict]:
    """前端下拉用: 所有走知识库的类型 (含 label/description), 不含 direct。"""
    return [
        {"type_id": k, "label": v.label, "description": v.description, "kb": v.kb}
        for k, v in RAG_DOC_TYPES.items()
    ]


def doc_type_strategy_list() -> list[dict]:
    """前端「切分策略」展示用: 每类文档的切分/解析/路由策略完整视图 (不含 direct)。

    与 doc_type_list() 的分工: 后者是上传下拉的最小集合 (保持契约稳定);
    本函数面向策略可视化, 额外给出 chunker 策略释义、max_chars/overlap、
    表格感知开关与路由关键词规模。全部取自本文件, 不触碰索引与写入逻辑。
    """
    items: list[dict] = []
    for type_id, spec in RAG_DOC_TYPES.items():
        cfg = spec.chunker or {}
        strategy = cfg.get("strategy") or "semantic"
        meta = CHUNK_STRATEGY_META.get(strategy, CHUNK_STRATEGY_META["semantic"])
        parse = spec.parse or {}
        items.append(
            {
                "type_id": type_id,
                "label": spec.label,
                "kb": spec.kb,
                "description": spec.description,
                "chunker": {
                    "strategy": strategy,
                    "strategy_label": meta["label"],
                    "cut_unit": meta["cut_unit"],
                    "rationale": meta["rationale"],
                    "max_chars": cfg.get("max_chars") or 0,
                    "overlap": cfg.get("overlap") or 0,
                },
                "parse": {"table": bool(parse.get("table")), "ocr": bool(parse.get("ocr"))},
                "routing_keywords_count": len(spec.routing_keywords or []),
                "prompt_hint": spec.prompt_hint,
            }
        )
    return items
