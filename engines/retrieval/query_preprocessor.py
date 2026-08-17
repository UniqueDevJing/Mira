"""Query 规则预处理 — 歧义查询的同义扩展与口语清理。

触发时机: 路由 source 为 llm/fallback（即规则未命中、查询偏模糊）时。
策略:
  - normalize_query: 去敬语/口语前缀 + 空白归一 → 用于向量检索（避免语义词稀释）
  - expand_query: normalize + 同义扩展 → 用于 BM25 关键词检索（提高词项召回）

同义扩展只追加不在原文中的词，防止关键词膨胀。纯引擎层，零外部依赖。
"""

# 业务词 → 同义/相关词（按主题）
SYNONYM_MAP = {
    "退款": ["退货", "退款申请"],
    "退货": ["退款", "退货退款"],
    "物流": ["快递", "运输", "配送"],
    "发货": ["发货时间", "出货"],
    "发票": ["开票", "电子发票"],
    "签收": ["收货", "到货"],
    "优惠": ["优惠券", "折扣"],
    "投诉": ["申诉", "举报"],
    "接口": ["API 接口"],
    "api": ["接口"],
    "部署": ["安装", "上线"],
    "配置": ["设置", "配置项"],
    "数据库": ["DB", "数据库配置"],
    "缓存": ["缓存策略"],
    "框架": ["Web 框架"],
}

# 口语/敬语前缀（按长度降序优先匹配）
POLITE_PREFIXES = [
    "麻烦帮我看一下",
    "帮我看一下",
    "我想问一下",
    "请问一下",
    "麻烦问一下",
    "你好请问",
    "请问",
    "您好",
    "你好",
    "麻烦",
    "谢谢",
]


def normalize_query(query: str) -> str:
    """清理口语/敬语 + 空白归一，用于向量检索。"""
    q = " ".join(query.strip().split())
    # 循环剥离敬语前缀，直到无匹配；同时去掉残留前导标点
    while True:
        stripped = False
        for p in sorted(POLITE_PREFIXES, key=len, reverse=True):
            if q.startswith(p):
                q = q[len(p) :]
                stripped = True
                break
        if not stripped:
            break
        q = q.lstrip("，,。：:、 ")
    return q


def expand_query(query: str) -> str:
    """normalize + 业务词同义扩展，用于 BM25 检索。"""
    q = normalize_query(query)
    ql = q.lower()
    additions = []
    for word, synonyms in SYNONYM_MAP.items():
        if word in ql:
            additions.extend(s for s in synonyms if s not in q)
    if additions:
        q = q + " " + " ".join(additions)
    return q


def preprocess_query(query: str) -> tuple[str, str]:
    """返回 (vector_query, bm25_query)。向量用原义，BM25 用扩展。"""
    return normalize_query(query), expand_query(query)
