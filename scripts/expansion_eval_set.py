"""QA 评测集扩充 — 路由关键词生成 + 检索评测集抽样。

输出 tests/eval_dataset.json，兼容 evaluate.py 格式:
  [{"kb": "service"|"tech", "question": "...", "reference_answer": "..."}]

生成策略:
1. service 类: 用 doc_types 路由词 (退款/退货/物流/订单/售后/发票/发货) + FAQ 模板构造问题
   这些关键词命中 intent_router SKILL_RULES → kb=service → RAG 服务走客服话术库回答
2. tech 类: 用 doc_types 路由词 (部署/架构/API/数据库/缓存/安装/配置/FastAPI/Docker)
   → kb=tech → RAG 服务走技术文档库回答
3. 从检索评测集 (data/eval/eval_dataset.compat.json) 补充通用领域问题:
   用于评估系统在跨领域场景下的表现

运行:
  python scripts/expansion_eval_set.py            # 默认生成 → tests/eval_dataset.json
  python scripts/expansion_eval_set.py --total 50  # 目标总数 50 (≈每类 25)
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser(description="Expand QA eval set: router-keyword + retrieval eval mix.")
    ap.add_argument("--out", default="tests/eval_dataset.json")
    ap.add_argument("--total", type=int, default=30, help="目标总条数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    items: list[dict] = []

    # ── 1. Service 类: 基于客服场景路由词构造 ──────────────────────────────
    service_items = [
        {"kb": "service", "question": "我要退货，怎么操作？", "reference_answer": "您可以在订单详情页点击申请退货，选择退货原因后提交。审核通过后会有退货地址发送给您，请将商品打包寄回。我们会在签收后的三个工作日内完成退款。"},
        {"kb": "service", "question": "退款什么时候到账？", "reference_answer": "退款审核通过后会原路退回您的支付账户，一般一到三个工作日到账，请您留意账户变动。使用微信支付或支付宝支付的订单，退款预计一到三个工作日到账；使用银行卡支付的订单，预计三到五个工作日到账。"},
        {"kb": "service", "question": "如何申请退款？", "reference_answer": "消费者可在商品签收之日起七天内申请退款，无需说明理由；超过七天但仍在三十天内的，仅支持因商品质量问题或商家责任导致的退款。所有退款申请统一由客服团队受理，处理时效为四十八小时内完成审核。"},
        {"kb": "service", "question": "退货物流费用谁承担？", "reference_answer": "非质量问题退货的物流费用由消费者承担；商品存在质量问题的退货物流费用由商家承担。"},
        {"kb": "service", "question": "订单多久能发货？", "reference_answer": "正常订单在支付成功后 24 小时内发货，促销活动期间可能顺延至 48 小时。特殊商品（如定制类、预售类）以页面标注时间为准。"},
        {"kb": "service", "question": "如何修改收货地址？", "reference_answer": "订单未发货前，可在订单详情页点击修改地址；已发货的订单需联系客服协商拦截并重新填写地址。"},
        {"kb": "service", "question": "发票什么时候开？", "reference_answer": "发票在订单确认收货后 3 个工作日内开具。您可以在订单详情页申请开票，填写抬头和税号信息即可。电子发票将发送至您预留的邮箱。"},
        {"kb": "service", "question": "运费怎么计算？", "reference_answer": "单笔订单满 99 元包邮；未满 99 元的订单收取 10 元基础运费。偏远地区（新疆、西藏、青海等）运费另计。会员用户每月享有 2 次包邮权益。"},
        {"kb": "service", "question": "商品有质量问题怎么处理？", "reference_answer": "签收后发现质量问题的，请在 7 天内拍照上传至售后页面申请退换。客服会在 48 小时内审核，审核通过后安排上门取件或指定仓库寄回，往返运费由商家承担。"},
        {"kb": "service", "question": "可以取消订单吗？", "reference_answer": "订单未发货前可随时在订单详情页申请取消；已发货的订单可拒收或在收到后退货处理。"},
        {"kb": "service", "question": "售后服务电话是多少？", "reference_answer": "客服热线 400-xxx-xxxx，服务时间为工作日 9:00-18:00。也可通过 APP 在线客服或官方网站留言渠道进行咨询。"},
        {"kb": "service", "question": "换货需要多久？", "reference_answer": "换货流程与退货类似，商家签收退回商品并确认无误后，会在 3 个工作日内发出新商品。整体周期约 7-10 个工作日，具体取决于物流时效。"},
    ]

    # ── 2. Tech 类: 基于技术文档路由词构造 ─────────────────────────────────
    tech_items = [
        {"kb": "tech", "question": "系统使用什么 Web 框架？", "reference_answer": "系统推荐使用 FastAPI 作为主力 Web 框架，搭配 SQLAlchemy 2.0 异步 ORM 和 Alembic 数据库迁移工具，使用 Pydantic Settings 进行配置管理，使用 Uvicorn 作为生产级 ASGI 服务器。"},
        {"kb": "tech", "question": "数据库用什么？", "reference_answer": "推荐方案中使用 PostgreSQL 配合 SQLAlchemy 2.0 异步 ORM 和 Alembic 数据库迁移工具。PostgreSQL 提供了良好的事务支持和 JSON 存储能力，适合 RAG 系统的元数据管理。"},
        {"kb": "tech", "question": "如何配置大模型 API Key？", "reference_answer": "LLM 密钥通过环境变量注入，设置 RAG_LLM_API_KEY 为你的 API Key。默认 LLM 服务地址为 TokenHub API (https://tokenhub.itcast.cn/v1)。生产环境应将密钥写入 .env 文件或容器环境变量，严禁硬编码在代码中。"},
        {"kb": "tech", "question": "系统支持 Docker 部署吗？", "reference_answer": "支持。生产环境通过 Docker 容器化部署，使用 Uvicorn 作为 ASGI 服务器。docker-compose.yml 定义了 API 服务和依赖服务的编排配置。"},
        {"kb": "tech", "question": "向量数据库用的什么？", "reference_answer": "开发阶段使用 LanceDB（轻量嵌入式），生产环境可选 Milvus（分布式向量数据库）。LanceDB 适用于单机部署和快速验证，Milvus 支持水平扩展和高并发查询。"},
        {"kb": "tech", "question": "Embedding 模型是什么？", "reference_answer": "使用 BGE-small-zh，维度 512。相比 text2vec-large-chinese（1024 维），内存占用减少一半，精度提升不足 3%，是性价比更高的选择。嵌入时需加 passage: 和 query: 前缀以匹配训练输入格式。"},
        {"kb": "tech", "question": "API 鉴权怎么配置？", "reference_answer": "通过设置 RAG_API_KEY_ENABLED=true 启用 API Key 鉴权，外部请求必须携带 X-API-Key 请求头或 Bearer Token。RBAC 角色控制知识库访问权限（admin 全量 / reader 受限）。"},
        {"kb": "tech", "question": "RAG 系统的架构是怎样的？", "reference_answer": "架构链路: 用户请求 → FastAPI → API Key 认证中间件 → IntentRouter（规则≥0.85 直通 / LLM 分类 1.5s / fallback tech）→ Skill 注册表 → RAG 流水线（Embedding + 向量|BM25 并行 → RRF 融合 → Cross-Encoder 重排 → LLM 生成）→ 响应组装。"},
        {"kb": "tech", "question": "如何进行中文文本分块？", "reference_answer": "采用语义相似度驱动的动态分块策略（StructureChunker），结合标题树结构切分，而非简单的 RecursiveCharacterTextSplitter。不同文档类型使用不同 chunker 策略：FAQ 类型 max_chars=480 overlap=60，其他类型 overlap=80。"},
        {"kb": "tech", "question": "检索融合策略用了什么算法？", "reference_answer": "使用 RRF (Reciprocal Rank Fusion) 融合 BM25 关键词检索和向量语义检索结果。RRF 通过倒数排名合并两种排序信号，避免单一排序方法在特定查询上的偏差。"},
        {"kb": "tech", "question": "系统有熔断降级机制吗？", "reference_answer": "有完整的降级链: 全局 12s 超时预算 → Embedding 失败重试 → 向量/BM25 并行独立返回 → RRF 无重排 → LLM 生成失败回退检索摘要。LLM 客户端内置重试、连接池和熔断器。"},
        {"kb": "tech", "question": "如何查看 Token 消耗？", "reference_answer": "Token 消耗通过 llm_client 统计记录在 qa_logs 表中，每次问答都会记录输入/输出 token 数。可通过 GET /api/v1/qa/logs 接口查询历史记录的消耗明细。"},
    ]

    # 按目标比例分配
    n_service = args.total // 2
    n_tech = args.total // 2
    n_remaining = args.total - n_service - n_tech

    items.extend(random.sample(service_items, min(n_service, len(service_items))))
    items.extend(random.sample(tech_items, min(n_tech, len(tech_items))))

    # 如果需要更多，从检索评测集补充
    if n_remaining > 0:
        src = "data/eval/eval_dataset.compat.json"
        try:
            with open(src, encoding="utf-8") as f:
                pool = json.load(f)
            extra = random.sample(pool, min(n_remaining, len(pool)))
            items.extend(extra)
        except FileNotFoundError:
            pass  # 缺少源数据则跳过补充

    random.shuffle(items)

    out_items = [
        {"kb": x.get("kb", "misc"), "question": x["question"], "reference_answer": x.get("reference_answer", "")}
        for x in items
    ]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote {len(out_items)} items -> {args.out}")
    cat_count: dict[str, int] = {}
    for x in out_items:
        k = x.get("kb", "misc")
        cat_count[k] = cat_count.get(k, 0) + 1
    for k, v in sorted(cat_count.items()):
        print(f"  [{k}] {v}")


if __name__ == "__main__":
    main()
