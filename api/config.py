"""应用配置"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    llm_base_url: str = "https://tokenhub.itcast.cn/v1"
    llm_model: str = "deepseek-v4-flash"
    llm_api_key: str = ""  # 通过环境变量 RAG_LLM_API_KEY 或 .env 文件注入

    # API 鉴权: 启用后外部请求必须带 X-API-Key / Bearer
    # 注意: security.py 以 os.environ 优先 (支持运行时/测试动态覆盖), 此处读 .env 兜底
    api_key_enabled: bool = False
    api_key: str = ""  # 单 Key 兼容 (视为 admin, 可访问全部知识库)
    # S4 安全: 本机 loopback (127.0.0.1) 免鉴权豁免 — 显式开关, 生产默认关。
    # 部署在反向代理后必须关闭: 代理未转发 XFF/Cf-Connecting-Ip 时外部请求会被误判 loopback 全匿名绕过。
    # 本机开发/测试如需免 Key 直连, 显式设 RAG_LOOPBACK_EXEMPT=true。
    loopback_exempt: bool = False
    # 多 Key 白名单: JSON 映射 "<key>": {"name": str, "kbs": [str] | "*", "role": "admin"|"reader"}
    # kbs="*" / 缺省 / null = 全部知识库; role 仅语义标记, 实际权限由 allowed_kbs 决定
    api_key_whitelist: str = ""

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    # LanceDB 向量库目录 (测试经 RAG_VECTOR_URI 注入 tmp 隔离生产数据)
    vector_uri: str = "./lancedb_data"

    # OCR
    ocr_lang: str = "ch"

    # 上传限制 (MB) — API 设计承诺 50MB 上限, 超限 413 拒绝, 防内存耗尽
    max_upload_mb: int = 50

    # Reranker: Cross-Encoder 模型名（如 "BAAI/bge-reranker-v2-m3"）。
    # 留空 = Bi-Encoder 余弦相似度（无下载依赖）；填模型名 = Cross-Encoder 延迟加载。
    reranker_model: str = ""

    # 分块
    chunk_max_chars: int = 800
    chunk_overlap: int = 128

    # Self-Retrieval 多轮自适应检索（LLM 改写 + 评估）。
    # 默认关: 每查询付 LLM 改写成本, 并发下易超时降级 (L2), 答案质量与单轮相同。
    # 质量敏感场景按请求显式开启 (QARequest.enable_self_retrieval=true)。
    enable_self_retrieval: bool = False

    # Router + Skill 编排参数（可配置化，支持按语料/环境调整）
    total_timeout_s: float = 12.0  # 全局 QA 超时预算 (>= llm_generate_timeout_s + 检索/重排余量)
    # top1 分数低于此值触发跨库兜底。0.55→0.50 (scripts/calibrate_threshold.py 校准):
    # 18 条标注 F1 打平 (0.5), 但 0.50 精度 1.0 vs 0.4, 消除 3 次无谓兜底 (延迟+污染)
    cross_kb_threshold: float = 0.50
    rerank_timeout_s: float = 0.5
    cross_kb_timeout_s: float = 0.6
    llm_generate_timeout_s: float = 8.0  # 实测 DeepSeek-v4-flash TTFT 3.5s, 完整生成 6.2s (2026-08-10)
    embed_cache_ttl_s: int = 600  # Query Embedding 缓存 TTL (秒)
    # Self-Retrieval 查询改写 per-call 超时 (推理模型下 1.5s 频繁降级, 放宽到 3s)
    rewrite_timeout_s: float = 3.0
    # 忠实度护栏: 生成答案与检索上下文词重合率低于此值 → 判定高幻觉风险, 替换为拒答提示
    fidelity_threshold: float = 0.4
    # 忠实度护栏是否启用 embedding 语义信号 (同义改写不再被词重合误拒)。关闭则退回纯词重合。
    fidelity_use_embedding: bool = True

    # QA 结果缓存 (内存 TTL): 相同输入指纹命中直接返回, 跳过路由+检索+LLM。
    # 注意: 缓存键含 temperature, 改温度即换条目; 缓存命中时不感知数据更新 (TTL 后自然过期)
    qa_cache_enabled: bool = True
    qa_cache_ttl_s: int = 3600

    # CORS — 开发环境默认仅前端本地 origin（localhost:3000），生产环境通过 RAG_CORS_ORIGINS 配置（逗号分隔）
    # 生产收紧: 仅前端 origin; 前端与 API 同源部署时 CORS 不生效, 无影响。
    # 覆盖: RAG_CORS_ORIGINS='["http://localhost:3000","https://app.example.com"]'
    cors_origins: list[str] = ["http://localhost:3000"]

    # 限流 (slowapi)。存储后端可配 Redis 实现多 worker 共享, 否则内存单进程有效。默认关, 防测试误触
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 60

    # 跨进程共享态后端: "memory" (默认, 单进程有效) 或 "redis" (多 worker 共享, 需 redis_url)。
    # QA 缓存走此后端; 限流经 limiter.storage_uri 接入 Redis。Redis 不可用/未配置时回退内存。
    shared_state_backend: str = "memory"
    redis_url: str = ""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env")


settings = Settings()
