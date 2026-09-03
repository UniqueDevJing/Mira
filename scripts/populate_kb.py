"""直接写入知识库 — 绕过上传接口，用引擎层入库文本内容。

用法:
  python scripts/populate_kb.py          # 全部上传 (service + tech)
"""
import hashlib
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Service FAQ ───────────────────────────────────────────────
SERVICE_FAQS = [
    {
        "title": "退款政策与流程",
        "kb": "service",
        "doc_type": "service",
        "content": """退款政策说明

一、退款申请条件
1. 消费者可在商品签收之日起七天内申请退款，无需说明理由（七日无理由退货）。
2. 超过七天但仍在三十天内的，仅支持因商品质量问题或商家责任导致的退款申请。
3. 定制类商品、鲜活易腐商品、数字下载商品不适用七日无理由退货。

二、退款处理时效
1. 退款申请提交后，客服团队将在 48 小时内完成审核。
2. 审核通过后，款项将原路退回您的支付账户。
3. 使用微信支付或支付宝支付的订单，退款预计一到三个工作日到账。
4. 使用银行卡支付的订单，预计三到五个工作日到账。
5. 如遇银行系统维护等特殊情况，到账时间可能延长至七个工作日。

三、退款操作步骤
1. 登录账户，进入"我的订单"页面。
2. 找到需要退款的订单，点击"申请退款"按钮。
3. 选择退款原因，填写备注说明（如有质量问题请上传照片）。
4. 确认信息无误后提交申请。
5. 等待客服审核，审核结果将通过站内消息和短信通知您。

四、部分退款
1. 订单中部分商品申请退款的，按该商品实际支付金额计算退款。
2. 订单包含赠品或享受满减优惠的，需扣除赠品价值及优惠分摊金额。

五、退款咨询
如有疑问，可拨打客服热线 400-xxx-xxxx（工作日 9:00-18:00）或通过 APP 在线客服咨询。""",
    },
    {
        "title": "退货流程指南",
        "kb": "service",
        "doc_type": "service",
        "content": """退货流程指南

一、如何申请退货
1. 在"我的订单"页面找到目标订单，点击"申请退货"。
2. 选择退货数量及原因，系统将自动生成退货地址。
3. 打印退货面单（或手写收件信息），将商品打包寄回。

二、退货物流费用承担
1. 非质量问题退货：物流费用由消费者承担，基础运费 10 元/单。
2. 商品存在质量问题的退货：往返运费均由商家承担。需在售后页面拍照上传证明，客服审核后安排上门取件。

三、退货商品要求
1. 商品应保持原貌，不影响二次销售（包装完整、配件齐全、无人为损坏）。
2. 已激活的电子产品不支持无理由退货。
3. 贴身衣物、化妆品等特殊类目拆封后不支持无理由退货。

四、退货进度查询
1. 提交退货申请后，可在"售后服务"页面查看处理进度。
2. 仓库签收退货后 1 个工作日内完成验货。
3. 验货通过后 24 小时内发起退款流程。

五、退货拒收情形
以下情况商家有权拒绝退货申请：
- 超过退货期限（签收后 7 天无理由 / 15 天换货）
- 商品严重影响二次销售（已使用、有污渍、缺配件）
- 提供虚假退货理由或恶意骗货""",
    },
    {
        "title": "物流与配送服务",
        "kb": "service",
        "doc_type": "service",
        "content": """物流与配送服务说明

一、发货时间
1. 正常订单在支付成功后 24 小时内发货。
2. 促销活动期间（双11、618 等）可能顺延至 48 小时。
3. 特殊商品（定制类、预售类）以商品页面标注时间为准。

二、合作物流公司
公司与顺丰、中通、圆通、韵达、德邦合作。系统根据收货地址自动分配最优快递。用户也可在订单详情页指定可用快递。

三、运费计算
1. 单笔订单满 99 元包邮（不含偏远地区）。
2. 未满 99 元的订单收取 10 元基础运费。
3. 偏远地区（新疆、西藏、青海、内蒙古）邮费另计，约 15-30 元。
4. 会员用户每月享有 2 次包邮权益，不累计。

四、物流追踪
1. 发货后系统发送短信通知运单号。
2. 在"我的订单"页面可实时查看物流轨迹。
3. 物流异常（延误、丢件）可联系客服介入处理。

五、签收与验货
1. 快递员派送时建议当面验货，确认外包装完好再签收。
2. 如包裹破损，可在快递单上注明"外包装破损"后签收，24 小时内联系客服申请补发或退款。
3. 未当面验货而签收后发现损坏的，需提供开箱视频作为凭证。

六、修改收货地址
1. 订单未发货前，可在订单详情页直接修改收货地址。
2. 已发货的订单无法直接修改，需联系客服协商拦截并重新填写。拦截可能产生额外费用。""",
    },
    {
        "title": "发票开具服务",
        "kb": "service",
        "doc_type": "service",
        "content": """发票开具服务说明

一、开票时间
1. 电子发票：订单确认收货后 3 个工作日内开具并发送至预留邮箱。
2. 纸质发票：需在订单确认后通过 APP 或客服申请，寄送时间为发票开具后 5-7 个工作日。

二、发票类型
1. 增值税普通发票（个人/企业均可申请，无税号要求也可开普票）。
2. 增值税专用发票（仅限企业客户申请，需提供完整税号、公司名称、地址电话、开户行及账号）。

三、开票信息填写
1. 在"我的订单"页面点击"申请开票"。
2. 选择发票类型，填写发票抬头（个人填姓名，企业填公司全称）。
3. 如需专票，必须填写完整的企业税务信息。
4. 确认信息无误后提交。

四、发票变更与重开
1. 发票开出后原则上不予更改。如信息有误，可申请作废后重新开具。
2. 跨月发票作废需走财务审批流程，处理周期约 5-10 个工作日。
3. 电子发票可直接下载 PDF 自行打印，效力等同纸质发票。

五、常见问题
Q: 一张订单可以分成多张发票吗？ A: 可以，开票时选择拆分金额即可。
Q: 预付卡/积分抵扣的部分能开发票吗？ A: 不能，只有实际支付金额可以开票。
Q: 多久没收到电子发票？ A: 先检查垃圾箱，仍未收到的请联系客服核实邮箱。""",
    },
]

# ── Tech Docs ─────────────────────────────────────────────────
TECH_DOCS = [
    {
        "title": "RAG 系统架构概览",
        "kb": "tech",
        "doc_type": "tech",
        "content": """RAG 系统架构概览

一、整体架构
系统采用分层微服务模式，核心链路如下：
用户请求 → FastAPI 入口 → API Key 认证中间件 → IntentRouter（规则≥0.85 直通 / LLM 分类 1.5s / fallback tech）→ Skill 注册表 → RAG 流水线（Embedding + 向量|BM25 并行 → RRF 融合 → Cross-Encoder 重排 → LLM 生成）→ 响应组装。

二、核心技术栈
- Web 框架：FastAPI 0.115+，搭配 Pydantic v2 数据校验。
- ORM：SQLAlchemy 2.0 异步驱动。
- 向量数据库：开发阶段 LanceDB（嵌入式轻量级），生产可选 Milvus（分布式）。
- Embedding 模型：BGE-small-zh（百度智能云），维度 512。嵌入时需加 passage: 前缀用于文档，query: 前缀用于检索。
- 重排序器：Cross-Encoder（bge-reranker-base），CPU 可承受。
- LLM：默认 DeepSeek-v4-flash / Qwen 系列，通过 OpenAI 兼容协议接入。

三、降级策略（完整降级链）
1. 全局超时预算 12 秒，分阶段控制：路由 0.1s / 检索 4s / 重排 3s / LLM 生成 8s。
2. Embedding 失败：重试一次，仍失败则回退 BM25 纯关键词检索。
3. 向量/BM25 任一腿可用：即使另一腿失败也不影响返回。
4. 重排超时：跳过重排，直接使用 RRF 融合后的原始排序。
5. LLM 生成失败：回退为纯检索摘要拼接模式（degradation_level=3）。

四、Token 消耗追踪
每次 LLM 调用记录在 qa_logs 表中，包括输入 token、输出 token、耗时。可通过 GET /api/v1/qa/logs 接口查询历史记录的消耗明细。

五、配置管理
所有配置通过 api/config.py 中的 Pydantic Settings 管理，环境变量统一以 RAG_ 为前缀。生产环境密钥写入 .env 文件或容器环境变量，严禁硬编码。""",
    },
    {
        "title": "部署与运维指南",
        "kb": "tech",
        "doc_type": "tech",
        "content": """部署与运维指南

一、本地开发环境搭建
前置条件：Python >= 3.12、Git。
1. 克隆仓库并进入目录。
2. 创建虚拟环境：python -m venv venv && source venv/bin/activate（Windows: venv\\Scripts\\Activate.ps1）。
3. 安装依赖：pip install -r requirements.txt。国内镜像用清华源：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt。
4. 设置环境变量：RAG_LLM_API_KEY=your-key、HF_ENDPOINT=https://hf-mirror.com、TRANSFORMERS_OFFLINE=1。
5. 启动服务：uvicorn api.main:app --reload --port 8000。
6. 访问 Swagger 文档：http://127.0.0.1:8000/docs。

二、Docker 容器化部署
1. 项目根目录执行 docker-compose up -d，构建 API 镜像并拉取依赖服务。
2. docker-compose.yml 定义了 API 服务和 LanceDB 数据卷的编排。
3. 生产环境建议开启 RAG_API_KEY_ENABLED=true 启用鉴权。

三、模型缓存
- Embedding 模型 BAAI/bge-small-zh 首次运行会从 HuggingFace 下载，约 250MB。国内设 HF_ENDPOINT=https://hf-mirror.com 加速。
- 下载完成后缓存在 ~/.cache/huggingface/hub/，后续启动无需联网。
- 离线环境部署先在联网机器下载模型后拷贝到目标机的 cache 目录。

四、常见运维问题
1. SSL certificate verify failed → export TRANSFORMERS_OFFLINE=1 或 HF_ENDPOINT=https://hf-mirror.com
2. 内存占用过高（>4GB） → 关闭 rerank_enabled=false，或将 rerank_candidate_k 降到 5
3. 并发请求慢 → 增加 uvicorn worker 数量（--workers 4），或用 Redis 作共享状态后端
4. 向量库磁盘空间不足 → 定期清理旧文档，或迁移 LanceDB 数据目录到更大磁盘

五、健康检查
GET /api/v1/health 返回服务状态。集成到 Kubernetes liveness/readiness probe。""",
    },
    {
        "title": "API 鉴权配置",
        "kb": "tech",
        "doc_type": "tech",
        "content": """API 鉴权配置说明

一、启用方式
设置环境变量 RAG_API_KEY_ENABLED=true，然后在配置中指定 Key：
- 单 Key 模式：RAG_API_KEY=your-secret-key，所有合法请求携带此 Key 即可（视为 admin 权限）。
- 多 Key 白名单模式：RAG_API_KEY_WHITELIST='{"key1":{"name":"admin","kbs":"*","role":"admin"},"key2":{"name":"reader","kbs":["service"],"role":"reader"}}'。

二、客户端认证方式
外部请求必须通过以下方式之一携带 API Key：
1. HTTP Header: X-API-Key: your-key
2. Bearer Token: Authorization: Bearer your-key

三、RBAC 角色控制
- admin 角色：可访问全部知识库，可上传文档。
- reader 角色：只能读取授权列表 kbs 中的知识库内容。
- 多 Key 模式下每个 key 独立绑定权限，便于团队协作隔离。

四、本机开发免鉴权
调试时可设 RAG_LOOPBACK_EXEMPT=true，此时仅 127.0.0.1 的请求无需携带 Key。反向代理后务必关闭。

五、限流保护
RAG_RATE_LIMIT_ENABLED=true 开启后默认每分钟 60 次请求上限。多 worker 场景需配 Redis 存储。

六、安全最佳实践
1. 生产环境 Key 绝不硬编码，通过 Docker Secret 或环境变量注入。
2. 定期轮换 API Key，新 Key 生效后删除旧 Key。
3. 监控 qa_logs 表中的异常高频调用，识别滥用行为。
4. CORS 默认仅允许 localhost:3000，生产环境收紧来源。""",
    },
    {
        "title": "检索融合与重排策略",
        "kb": "tech",
        "doc_type": "tech",
        "content": """检索融合与重排策略

一、检索两腿架构
系统采用 BM25 关键词检索 + 向量语义检索并行两腿方案，各自独立返回 top_k 结果。

二、融合策略
当前默认使用 interp（分数插值融合），BM25 权重占 0.5，向量检索占 0.5。实测在专名/关键词驱动语料上，BM25 的单腿召回率优于纯向量检索。
- interp 权重 w=0.5 时：R@3 70.6% / R@5 86.5% / Hit@5 98.2%，全场最佳。
- MRR 0.734 接近纯 BM25 水平。
- 尾部召回靠向量检索补充，故取均衡点而非纯 BM25。

三、重排序
Cross-Encoder（bge-reranker-base）对融合候选池做精细重排。默认关闭以提升延迟：
- CPU 环境下 re-rank 10 个候选约 1.04s。
- 20 候选约 2.00s，30 候选约 2.89s。
- 设为 False 时节省约 664ms/查询平均延迟，且实测排序一致性高。

四、自适应 Alpha
按候选池近重复密度动态调整 CE 权重：
- alpha_max=1.0（池子完全干净，纯 CE 权重最高）
- alpha_min=0.40（池子极度密集，偏检索稳健）
- density_threshold=0.95 判定近重复
- 映射公式：alpha = alpha_max - (alpha_max - alpha_min) * clamp(density/density_full, 0, 1)

五、置信度阈值
- top1_score < 0.50 触发跨库兜底（4s 预算）。
- 兜底仍低 → 判定答非所问风险，改为"未找到相关内容"提示。
- 该阈值经 documents 全表实测：真命中 top1∈[0.50,1.0]，无关命中 top1∈[0.34,0.50]。""",
    },
]


def main():
    from api.core.embedder import get_embedder

    from api.config import settings
    from api.state import get_bm25_index, get_vector_store
    from engines.chunking.strategies import get_chunker
    from engines.doc_types import get_doc_type
    from engines.parsing.txt_parser import TxtParser

    all_docs = SERVICE_FAQS[:]
    if not "--skip-service" in sys.argv:
        pass  # service already included
    if "--skip-tech" not in sys.argv:
        all_docs.extend(TECH_DOCS)

    embedder = get_embedder()
    uploaded = []

    # 增量更新: 内容哈希比对, 未变化文档跳过(避免全量重切/重嵌)
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".kb_hash_state.json")
    state = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}

    for doc_data in all_docs:
        kb = doc_data["kb"]
        doc_type = doc_data["doc_type"]
        title = doc_data["title"]
        content = doc_data["content"]

        key = f"{kb}:{title}"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if state.get(key) == content_hash:
            print(f"  [{kb}] SKIP (unchanged): {title}")
            continue

        doc_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]  # 稳定 doc_id, 重切时向量库 dedup 清旧块
        spec = get_doc_type(doc_type)
        chunker = get_chunker(spec, settings)
        # TextParser(bytes, filename=) 已不存在; 落临时文件走已验证的 TxtParser
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        tmp.write(content)
        tmp.close()
        try:
            uir = TxtParser().parse(tmp.name)
        finally:
            os.unlink(tmp.name)
        uir.source["path"] = title + ".txt"  # file_name/标题显示为文档标题而非临时路径
        uir.update_time = int(time.time())
        chunks = chunker.chunk(uir)
        for c in chunks:
            c.doc_id = doc_id  # 统一稳定 doc_id, 增量重切时向量库 dedup 清理旧块

        if not chunks:
            print(f"  [{kb}] SKIPPED (no chunks): {title}")
            continue

        texts = [c.content for c in chunks]
        embeddings = embedder.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        store = get_vector_store(kb)
        store.insert(chunks)

        bm25_docs = [
            {"id": c.chunk_id, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "content": c.content}
            for c in chunks
        ]
        get_bm25_index(kb).remove_doc(doc_id)  # 增量: 先清旧 BM25 索引再追加, 防重复累积
        get_bm25_index(kb).add_documents(bm25_docs)

        item = {"id": doc_id, "title": title, "kb": kb, "chunks": len(chunks)}
        uploaded.append(item)
        state[key] = content_hash  # 标记已入库(内容未变则下次跳过)
        print(f"  [{kb}] {title} -> {len(chunks)} chunks")

    # 持久化内容哈希状态, 支撑下次增量跳过
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [warn] KB 哈希状态写入失败(不影响本次): {e}")

    total_chunks = sum(i["chunks"] for i in uploaded)
    print(f"\nDone: {len(uploaded)} docs (增量跳过 {len(all_docs) - len(uploaded)} 个未变文档), {total_chunks} total chunks")
    print(json.dumps(uploaded, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()