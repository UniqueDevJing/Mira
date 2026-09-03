"""应用配置"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM (默认 DeepSeek 官方 OpenAI 兼容接口; 可用 RAG_LLM_* 或 .env 覆盖)
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
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
    # Embedding 后端: local=本地 SentenceTransformer; api=商用兼容 OpenAI 的 Embedding API(DashScope/OpenAI 等)
    embedding_backend: str = "local"
    embedding_api_base: str = ""            # OpenAI 兼容 base URL, 如 DashScope https://dashscope.aliyuncs.com/compatible-mode/v1
    embedding_api_key: str = ""             # API Key (环境变量注入, 严禁硬编码)
    embedding_api_model: str = ""           # API 模型名, 如 text-embedding-v3 / Typhoon-embedding; 缺省回退 embedding_model
    embedding_api_dims: int = 0             # 0=服务端默认; 否则请求指定维度(需 API 支持 dimensions 参数)
    embedding_api_timeout_s: float = 10.0

    # LanceDB 向量库目录 (测试经 RAG_VECTOR_URI 注入 tmp 隔离生产数据)
    vector_uri: str = "./lancedb_data"

    # OCR
    ocr_lang: str = "ch"

    # 上传限制 (MB) — API 设计承诺 50MB 上限, 超限 413 拒绝, 防内存耗尽
    max_upload_mb: int = 50

    # Reranker: Cross-Encoder 模型名。
    # 留空 = 不重排 (Bi-Encoder 重排与向量检索信号重复, 零提升); 填模型名 = Cross-Encoder 延迟加载。
    #
    # 选型依据 (embedding_device=cpu 下的现实约束):
    #   · bge-reranker-v2-m3 = XLM-R large, 24 层/1024 维/约 5.5 亿参数。
    #     CPU 上重排 20~30 条 512-token 文档基本必然超时降级, 等于没开。仅建议 GPU。
    #   · bge-reranker-base  = 12 层/768 维/约 2.8 亿参数, CPU 可承受。当前默认值。
    # 换 GPU 后可改为 BAAI/bge-reranker-v2-m3, 并把 rerank_timeout_s 压回 0.5。
    #
    # 首次加载需下载模型: 国内设 HF_ENDPOINT=https://hf-mirror.com (huggingface.co 直连不通)。
    # 下载失败会在 Reranker._get_ce_model 内静默降级为不重排, 不阻断启动, 启动日志会告警。
    reranker_model: str = "BAAI/bge-reranker-base"
    # 参与重排的候选上限。召回宽度是每路 top_k*2 (默认融合后约 40 条)。
    # CPU 下延迟随候选数线性增长。本机实测 (bge-reranker-base, 512 字符/文档):
    #     10 条 -> 1.04s      20 条 -> 2.00s      30 条 -> 2.89s     (约 100ms/pair)
    # 取 10 条: 延迟 ~1s, 对最终 top-10 几乎无损 (评测 Recall@1 仅小幅下降)。设 0 = 不限制。
    # 换硬件请用 .refactor-backup/bench_rerank.py 重测后调整。
    rerank_candidate_k: int = 10
    # rerank 分数融合 (P0-1 修复)。密集近重复语料下纯 Cross-Encoder 会把 golden 误排到
    # "实体替换"型近重复干扰项之后, Recall@3 反而下滑; 把 CE 分与 RRF 检索分融合, 用 golden
    # 天然靠前的高检索分托住它, 召回不掉且保住 rerank 精度。
    # CE 输入最大 token 数。0 = 模型默认 (通常 512)。
    # 这是 CPU 上 rerank 延迟的**最大杠杆** (批内 padding 到最长序列, 注意力 O(seq^2))。
    # 本机实测 (bge-reranker-base, 10 候选, 24 线程): 512->1112ms 384->964ms 256->595ms 192->436ms 128->291ms。
    # 调小前必须先在**你的语料**上验证排序/召回不受损:
    #     python scripts/bench_rerank_backends.py --limit 50      # 排序一致性
    #     python scripts/eval_retrieval.py --fusion --adaptive    # 端到端 Recall/MRR
    # 本机 hard 集实测 (150 问, 融合模式), 找甜点:
    #     512 -> R@1 58.7%  MRR 0.794  1107ms
    #     256 -> R@1 59.3%  MRR 0.798   717ms   <- 甜点: 快 35% 且质量更好(长文档尾部多为噪声)
    #     192 -> R@1 56.7%  MRR 0.782   500ms   (更快但开始掉质量, 不采用)
    reranker_max_length: int = 256
    # P0-2: 流式链路先把"重排前"的来源发给前端 (~50ms 可见), 再在后台做 rerank (~700ms)。
    # 只提前**来源面板**, 答案生成仍等 rerank 完成, 因此上下文/引用完全不受影响 (零错引风险)。
    # 前端收到第二个 sources 事件时整体替换渲染, 无副作用。
    stream_sources_before_rerank: bool = True
    # 推理后端: torch(默认) / onnx / auto(优先 ONNX, 失败回退 torch)。
    # 本机实测 ONNX Runtime 反而更慢 (fp32 0.76x, int8 1.21x 但 top1 排序一致率仅 84%), 故默认 torch。
    # 换硬件后用 scripts/bench_rerank_backends.py 复测再决定是否切 onnx。
    reranker_backend: str = "torch"
    # 重排总开关 (P2 诊断结论: 本专名/关键词驱动语料上 bge-reranker-base 反而伤排序 —
    # rerank 关 Recall@3 74.0% vs 开 71.4%, MRR 0.743 vs 0.732, 翻盘率 0%, 且引入 ~664ms/查询 CPU 延迟;
    # 答案支撑充分性(faithfulness 本地 proxy, 用 reference_answer 锚)持平 94.7% vs 94.7%。
    # 证据指向关闭。默认保持 True 以兼容历史行为, 建议设 False 以省延迟并微升召回精度。)
    rerank_enabled: bool = False
    # Rerank GPU 自适应 (P2#11): 用户未显式设定 rerank_enabled 且检测到 CUDA 时,
    # 自动切 BAAI/bge-reranker-v2-m3 并启用 (报告结论: 上 GPU 后精排质量上限显著提升)。
    # 显式设定 rerank_enabled(开/关)则尊重用户, 不做覆盖。CPU 环境自动保持关闭, 零回归。
    reranker_auto_gpu: bool = True
    # BM25 稀疏索引与向量库的一致性自检 (P0 数据一致性)。
    # 背景: BM25 曾因历史 pickle 索引与现行 JSON 读取不兼容, 每次启动静默回退**空索引**,
    # 导致混合检索退化为纯向量、而日志里只有一条 warning, 长期无人察觉。
    # 开启后首次取用某库索引时比对两者行数, 不一致则打 ERROR 并(可选)后台重建自愈。
    bm25_consistency_check: bool = True
    bm25_autorebuild: bool = True  # 不一致时后台从向量库重建; 关掉则只告警, 需手动跑 rebuild_bm25.py
    rerank_fusion_enabled: bool = True
    rerank_fusion_alpha: float = 0.9  # 固定 alpha 兜底值 (自适应关闭或计算失败时生效)
    # 自适应 alpha: 按候选池近重复密度动态调权重。固定 alpha 必然是妥协 —— 干净语料想要高 alpha
    # (吃满 CE 精度上限), 密集语料想要低 alpha (靠检索分托底)。映射:
    #     alpha = alpha_max - (alpha_max - alpha_min) * clamp(density / density_full, 0, 1)
    # 参数由 scripts/tune_alpha.py 在 clean/hard 双语料网格搜索得出 (data/eval/alpha_tuning.json)。
    rerank_fusion_adaptive: bool = True
    rerank_alpha_max: float = 1.00      # 池子完全干净时的 alpha (=1.0 即纯 CE, 此时 CE 最可靠)
    rerank_alpha_min: float = 0.40      # 池子极度密集时的 alpha (偏检索稳健)
    rerank_density_threshold: float = 0.95  # 判定"近重复"的余弦阈值
    rerank_density_mode: str = "pairs"  # pairs=近重复候选对占比(粒度更细, 实测优于 docs)
    rerank_density_full: float = 0.30   # 密度饱和点: 达到即取 alpha_min

    # 分块
    chunk_max_chars: int = 800
    chunk_overlap: int = 128

    # 检索融合策略: 'rrf' (排名融合) | 'interp' (分数插值融合, 放大 BM25 腿)。
    # 实测 (diag_fusion_weights.py, n=390, 结果可复现): 专名/关键词驱动语料上 BM25 远强于向量,
    # RRF 把 BM25 强信号稀释到接近向量水平 (R@1 34.7% vs BM25 单腿 43.2% 插值口径)。
    # 插值权重扫描: w=0.5 时 R@3 70.6%/R@5 86.5%/H@5 98.2% 全场最佳, MRR 0.734 接近纯 BM25;
    # w=1.0 (纯BM25) R@1/MRR 最好但 R@5 最低 —— 尾部召回靠向量补, 故取均衡点 0.5。
    # 曾因"bge-reranker-base 会抵消 interp 收益 (37.1%@1→33.1%)"而默认 rrf;
    # 2026-08-31 重排器已默认关闭 (rerank_enabled=False), 抵消因素不复存在, 故切回 interp。
    # 注: BM25 索引曾在生产静默失效 (pickle/JSON 格式不兼容, 见 rebuild_bm25.py 文档),
    # 修复后本结论才首次可直接适用于生产 —— 勿再让 BM25 缺席, 自检见 api/state.py。
    fusion_method: str = "interp"
    fusion_bm25_weight: float = 0.5

    # Self-Retrieval 多轮自适应检索（LLM 改写 + 评估）。
    # 默认关: 每查询付 LLM 改写成本, 并发下易超时降级 (L2), 答案质量与单轮相同。
    # 质量敏感场景按请求显式开启 (QARequest.enable_self_retrieval=true)。
    enable_self_retrieval: bool = False

    # P1-1: 单轮 LLM 查询改写 — 改善检索排序质量 (突破 fusion/rerank 调参天花板)。
    # 诊断(scripts/diag_bottleneck.py): golden 进重排候选池 99.2%, 但 RRF 仅把 golden 排进 top5 的 ~85%;
    # rerank/α/RRF_K 调优均无法突破 context_precision≈0.318。唯一杠杆是改写查询让 golden 排更靠前。
    # 默认开: 改写可突破 fusion/rerank 调参天花板, 让 golden 排更靠前 (diag_bottleneck 证实为唯一杠杆)。
    # 代价: 每查询多付一次 LLM 调用(~0.3-0.8s), 流式首屏 sources 顺延; 超时/失败自动回退规则预处理。
    query_rewrite_enabled: bool = True
    query_rewrite_timeout_s: float = 0.8

    # 查询嵌入增广 (打开首阶段召回天花板) — HyDE / PRF。
    # 诊断: Recall@10 触顶 ~97.9% 是首阶段召回天花板(语料覆盖上限, 增广改不动, 重排也救不了);
    #       真正可打的杠杆是"顶部排序"(Recall@1/MRR) —— 答案质量由它决定。
    # 实证 (scripts/eval_retrieval.py, 同题 150 问对比, 生产默认 --fusion --adaptive):
    #   PRF: Recall@1 59.3% -> 63.3% (+4.0pp), Hit@1 +4.6pp, MRR 0.798 -> 0.824 (+0.026)。
    #   增益穿过重排融合(首阶段 +0.014 MRR -> 端到端 +0.026, 重排放大而非抵消)。
    #   首阶段全量 390 问: Recall@1 34.7% -> 36.8% (+2.1pp), MRR 0.743 -> 0.762 (+0.019)。
    # 结论: PRF 默认开启。代价仅一次额外首阶段向量检索(相对 4.6s 重排可忽略), 质量只增不减。
    # hyde 保留可选但未经正向验证(每问多一次 LLM 调用), 不设默认。
    query_augmentation_enabled: bool = True
    query_augmentation_strategy: str = "prf"    # hyde | prf (prf 已实证有效, hyde 待验证)
    query_augmentation_weight: float = 0.5      # 反馈/假设向量融合权重 (0=无效, 1=纯反馈)
    query_augmentation_prf_k: int = 10          # PRF 首轮反馈候选数
    query_augmentation_prf_timeout_s: float = 0.4  # PRF 首轮检索预算(超时则跳过增广, 回退原向量)
    query_augmentation_hyde_timeout_s: float = 1.0

    # Router + Skill 编排参数（可配置化，支持按语料/环境调整）
    total_timeout_s: float = 12.0  # 全局 QA 超时预算 (>= llm_generate_timeout_s + 检索/重排余量)
    # top1 分数低于此值触发跨库兜底。0.55→0.50 (scripts/calibrate_threshold.py 校准):
    # 18 条标注 F1 打平 (0.5), 但 0.50 精度 1.0 vs 0.4, 消除 3 次无谓兜底 (延迟+污染)
    cross_kb_threshold: float = 0.50
    # 原 0.5s 对 CPU cross-encoder 过紧, 会频繁触发 L1 超时降级 (= 白付了模型成本却没重排)。
    # 3.0s 对应实测 20 候选 2.00s, 留约 50% 余量 (并发/慢机器)。仍远小于 total_timeout_s=12.0。
    #   · 延迟敏感: 把 rerank_candidate_k 降到 10 -> 1.04s, 本值可相应降到 1.5
    #   · GPU 环境: 改 v2-m3 并把本值压回 0.5
    # 超时只降级这一次检索, 不影响答案生成。
    rerank_timeout_s: float = 3.0
    cross_kb_timeout_s: float = 4.0  # 跨库兜底整体预算: 需并发检索多个知识库, 0.6s 过紧会导致首查即超时放弃(检索失败)
    llm_generate_timeout_s: float = 8.0  # 实测 DeepSeek-v4-flash TTFT 3.5s, 完整生成 6.2s (2026-08-10)
    embed_cache_ttl_s: int = 600  # Query Embedding 缓存 TTL (秒)
    # Self-Retrieval 查询改写 per-call 超时 (推理模型下 1.5s 频繁降级, 放宽到 3s)
    rewrite_timeout_s: float = 3.0
    # 忠实度护栏: 生成答案与检索上下文词重合率低于此值 → 判定高幻觉风险, 替换为拒答提示
    # 0.6: 实测对自然改写回答友好（纯词重合通常≥0.6），避免语义等价但措辞不同的正确答案被误判
    fidelity_threshold: float = 0.6
    # 置信度硬下限: 最佳来源分数低于此值 → 直接拒答, 不基于无关上下文生成(彻底杜绝答非所问)。
    # 标定依据(对 documents 全表实测): 真命中 top1∈[0.50,1.0], 错误/无关命中 top1∈[0.34,0.50];
    # 0.50 为清晰分界, 低于即判定答非所问风险, 改为"未找到相关内容"而非生成错误答案。
    answer_confidence_floor: float = 0.50
    # 忠实度护栏是否启用 embedding 语义信号 (同义改写不再被词重合误拒)。关闭则退回纯词重合。
    fidelity_use_embedding: bool = True
    # 数字硬校验: 关闭 — 中文区间(3至5)、乱码文本、推算数字会导致高误拒;
    # 词重合+embedding 信号已足够拦截明显幻觉。需要开启时设 RAG_FIDELITY_CHECK_NUMBERS=true。
    fidelity_check_numbers: bool = False
    # #4 幻觉前置: 生成前"上下文零词重合"额外拒答(low_relevance)。
    # 2026-09-02 起默认开启 —— 触发条件已加 embedding 语义下限兜底(low_relevance_min_sim),
    # 离线评估(scripts/eval_preguard.py, bge-small-zh):
    #   · 产品语料等价口语改写 10 对: 90% 词零重合(纯词重合逻辑必误拒), 但语义相似度全部≥0.50
    #     → 语义兜底下误拒 90%→0;
    #   · 无关高分干扰 2 对(运维问题 vs 退款/运费政策): 语义相似度 0.37-0.40, 仍 100% 拦截;
    #   · 390 合成正样本(检索完美命中): 零重合 0/390, 零误伤。
    # 语义兜底降级: embed 不可用/异常 → 保守放行(零词重合本身不可靠), 不因护栏故障误伤。
    # 分数下限拒答(low_confidence)始终生效, 不受本开关影响。
    answerability_preguard_enabled: bool = True
    # low_relevance 的语义下限: 词零重合 且 与上下文最大语义相似度 < 此值 → 判定真无关拒答;
    # ≥ 此值视为语义等价改写(如"钱通常多久到卡上" vs "退款 1-3 工作日到账"), 放行交由生成。
    # 0.50 为 bge-small-zh 评估甜点(改写样本 ≥0.50, 无关样本 ≤0.40, 分界清晰)。
    low_relevance_min_sim: float = 0.50

    # P1' 选择性扇出 + early-exit: 解决单库路由错误导致的端到端召回丢失
    # (路由准确率 79% → 端到端召回 75%→90%+ 的唯一最大杠杆)。
    # 机制: 主候选置信度 < route_early_exit 视为模糊, 对 top route_fanout_top_n 候选 KB 扇出检索并合并;
    # 主候选确定(规则>=0.85 已在 IntentRouter 内 early-exit 不调 LLM / LLM 主候选 conf>=early_exit)时单路,
    # 不扇出 → 避免每次检索延迟翻倍。route_fanout_enabled=false 整体回退单路(灰度/回滚用)。
    route_fanout_enabled: bool = True
    route_early_exit: float = 0.8
    route_fanout_top_n: int = 2

    # QA 结果缓存 (内存 TTL): 相同输入指纹命中直接返回, 跳过路由+检索+LLM。
    # 注意: 缓存键含 temperature, 改温度即换条目; 缓存命中时不感知数据更新 (TTL 后自然过期)
    qa_cache_enabled: bool = True
    qa_cache_ttl_s: int = 3600
    # 语义近重复命中: 在精确哈希之上, 对"同参数作用域"内的近义/改写问题做 cosine 命中,
    # 省 LLM+rerank 调用 (生产降本)。阈值偏高(0.92) 仅命中近重复, 防误答; 同作用域才比, 不跨 RBAC。
    # 语义索引为 in-process: 内存后端完整生效; Redis 后端下仍为 best-effort(精确命中照常跨 worker)。
    qa_cache_semantic_enabled: bool = True
    qa_cache_semantic_threshold: float = 0.92

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

    # 图谱整图 (Redis/文件) 持久化的 HMAC 完整性校验密钥。
    # 多 worker 经 Redis 共享同一份图谱时必须设置 (各进程用同一密钥签名/验签, 否则互不认);
    # 留空则进程内生成随机密钥 — 仅防外部 RCE, 不保证跨进程共享。覆盖: RAG_GRAPH_HMAC_SECRET=...
    graph_hmac_secret: str = ""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env")


settings = Settings()
