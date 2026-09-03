"""API 全局状态 — 按知识库隔离的单例（线程安全）。

每个知识库(kb)持有独立的 VectorStore / GraphStore / BM25 索引，防止跨库污染。
- get_vector_store(kb)      → LanceDB 表按 kb 隔离 (默认表 documents 向后兼容)
- get_graph_rag(kb)         → 图谱按 kb 隔离 (实体/关系抽取器共享, 图本身隔离)
- get_bm25_index(kb)        → BM25 稀疏索引按 kb 隔离

逐出策略为 LRU: 命中时 move_to_end, 容量超限时逐出最久未使用的 kb。
(原实现只在"首次创建"时插入, 高频访问的早期 kb 会被误逐出, 下次访问触发
 GraphStore 重建 = 重新做一遍实体/关系 LLM 抽取, 代价极高。)
"""

import logging
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock, Thread
from typing import TYPE_CHECKING

from api.config import settings
from engines.graph_rag.entity_extractor import EntityExtractor, RelationExtractor
from engines.graph_rag.graph_retriever import GraphRAGRetriever
from engines.graph_rag.graph_store import GraphStore

if TYPE_CHECKING:
    from engines.embedding.embedder import EmbeddingService
    from engines.retrieval.bm25_index import Bm25Index
    from engines.retrieval.reranker import Reranker
    from engines.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# BM25 持久化目录 (基于项目根, 与 document_store 一致)
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

_entity_ext = None
_rel_ext = None
_ext_lock = Lock()

_vector_map: OrderedDict[str, "VectorStore"] = OrderedDict()
_graph_map: OrderedDict[str, "GraphRAGRetriever"] = OrderedDict()
_bm25_map: OrderedDict[str, "Bm25Index"] = OrderedDict()
_vector_lock = Lock()
_graph_lock = Lock()
_bm25_lock = Lock()

# 每个知识库只做一次 BM25 一致性自检, 避免每次取索引都比对行数
_bm25_checked: set[str] = set()
_bm25_check_lock = Lock()

# 默认表名兼容旧数据；新库统一 rag_<kb> 前缀
LEGACY_KBS = {"", "documents"}

# 本地模型目录 (scripts/download_model.py 下载目标)
_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def resolve_model_path(model_id: str) -> str:
    """本地 models/ 优先, 命中则返回本地路径 — 免除运行时联网下载。

    未命中 (或本身就是路径) 原样返回, 由 sentence-transformers 走 HF 逻辑。
    顺序: 已是路径 > models/<repo 末段> 存在 > 原样返回 repo id。
    """
    if not model_id:
        return model_id
    if model_id.startswith(("./", "/", "~")) or os.path.isdir(model_id):
        return model_id
    local = _MODELS_DIR / model_id.split("/")[-1]
    return str(local) if local.is_dir() else model_id

# 每类型 kb 单例上限 — 防任意新 kb 名无限创建实例 (向量表/图谱/BM25 内存与磁盘泄漏)
_MAX_KB_INSTANCES = 32


def _evict_oldest(mapping: OrderedDict, key: str, value) -> None:
    """插入新实例并逐出最久未使用的 (OrderedDict 尾部为最近使用)。"""
    mapping[key] = value
    mapping.move_to_end(key)
    if len(mapping) > _MAX_KB_INSTANCES:
        mapping.popitem(last=False)


def _vector_table(kb: str) -> str:
    return "documents" if kb in LEGACY_KBS else f"rag_{kb}"


# 已挂载 KB 探测结果缓存: key=vector_uri。
# 必须是 dict 而非单值 —— 测试套件每测试切换 vector_uri (指向全新临时目录),
# 若只缓存单值, 第一个有数据的 vector_uri 的探测结果会泄漏给后续空目录测试,
# 把路由指向本实例并无数据的库 (实测全量测试因此 33/34 个路由相关用例随顺序失败)。
# 按 uri 隔离后, 每个 vector_uri 各自探测一次, 跨测试/跨实例零污染。
_mounted_kbs_cache: dict[str, list[str]] = {}
_mounted_lock = Lock()


def _reset_mounted_kbs() -> None:
    """清空已挂载 KB 探测缓存。

    测试每测试切换 vector_uri 时必须调用 (见 tests/conftest.py 的 _reset_state_per_test),
    否则会命中上一个 vector_uri 的陈旧探测结果, 把路由指向当前实例并无数据的库。
    生产环境 vector_uri 固定, 探测结果全程有效, 无需调用。
    """
    global _mounted_kbs_cache
    with _mounted_lock:
        _mounted_kbs_cache = {}


def mounted_kbs() -> list[str]:
    """返回真实有数据的 KB 列表 (保持 RAG_KBS 顺序、已去重), 供路由/扇出收敛候选。

    为什么需要它: DOC_TYPES 注册了 10+ 文档类型, 但并非每个类型都真的导入过数据 ——
    本部署 9 张 rag_* 表里只有 policy(12)/service(47)/tech(36) 有行, product/contract/
    finance/hr/marketing/meeting 全部 0 行。路由候选若直接取 SKILLS 全量, 会把问题判给
    这些"空壳库", 检索必空 → 路由错 + 端到端召回 0 (实测 routed=product, recall=0),
    扇出还要白白多跑一遍空检索。

    判据必须是"表存在 且 行数 > 0": 只判表存在毫无作用, 因为空表同样建得出来。
    count_rows() 读表元数据、开销极小, 且每个 vector_uri 只探测一次并缓存。

    探测失败 (lancedb 不可用等) 或一张非空表都没有时回退 RAG_KBS 全量: 宁可多查一个
    空库, 也不因探测失败让系统失去路由能力。
    """
    uri = settings.vector_uri
    with _mounted_lock:
        cached = _mounted_kbs_cache.get(uri)
    if cached is not None:
        return cached
    try:
        import lancedb

        from engines.doc_types import RAG_KBS

        db = lancedb.connect(uri)
        # 以"表能否真正打开且有数据"为准, 不依赖 lancedb 表注册表(table_names)。
        # 实测: 已入库的表(如 rag_service/rag_tech)可能因注册表脱节未出现在 table_names()
        # 中, 若以注册表做门槛会把它们永远排除出路由候选 → 库"假死"(数据完好却查不到)。
        # open_table 直接按目录打开真实数据集, 故改为逐 kb 尝试打开+计数, 失败静默跳过。
        mounted: list[str] = []
        seen: set[str] = set()
        for kb in RAG_KBS:
            # RAG_KBS 含重复项 (tech 文档类型与 code 文档类型共用 kb=tech)
            if kb in seen:
                continue
            seen.add(kb)
            tn = _vector_table(kb)
            # 行数必须 > 0: 建了表却从未导入数据的空库同样检索不到, 只判存在无效
            try:
                if db.open_table(tn).count_rows() > 0:
                    mounted.append(kb)
            except Exception as e:  # noqa: BLE001 — 表不存在/损坏: 跳过该 kb
                logger.debug("KB %s (表 %s) 不可用, 跳过挂载: %s", kb, tn, str(e)[:120])
        result = mounted if mounted else list(RAG_KBS)
        if mounted:
            logger.info("已挂载知识库: %s", mounted)
        else:
            logger.warning("未探测到任何非空向量表, 路由候选回退全量 RAG_KBS")
    except Exception as e:  # noqa: BLE001 — 探测失败不应阻断启动/请求
        logger.warning("向量表探测失败, 路由候选回退全量 RAG_KBS: %s", str(e)[:120])
        result = list(RAG_KBS)
    with _mounted_lock:
        _mounted_kbs_cache[uri] = result
    return result


def _shared_extractors():
    """实体/关系抽取器跨库共享（避免重复建 LLM 客户端），图数据本身按库隔离。"""
    global _entity_ext, _rel_ext
    if _entity_ext is None:
        with _ext_lock:
            if _entity_ext is None:
                _entity_ext = EntityExtractor(
                    llm_url=settings.llm_base_url,
                    llm_model=settings.llm_model,
                    llm_key=settings.llm_api_key,
                )
                _rel_ext = RelationExtractor(
                    llm_url=settings.llm_base_url,
                    llm_model=settings.llm_model,
                    llm_key=settings.llm_api_key,
                )
    return _entity_ext, _rel_ext


def close_extractors() -> None:
    """释放共享抽取器的 LLM 客户端连接 (应用关闭时调用, lifespan yield 后)。"""
    global _entity_ext, _rel_ext
    with _ext_lock:
        for ext in (_entity_ext, _rel_ext):
            if ext is not None:
                try:
                    ext.close()
                except Exception as e:  # noqa: BLE001 — 关闭失败不影响进程退出
                    logger.debug("抽取器客户端关闭失败: %s", str(e)[:80])
        _entity_ext = _rel_ext = None


def get_graph_rag(kb: str = "documents") -> GraphRAGRetriever:
    with _graph_lock:
        if kb not in _graph_map:
            entity_ext, rel_ext = _shared_extractors()
            # 图谱持久化到 data/graph_<kb>.pkl, 重启恢复 (与 BM25 对称); GraphStore 内部损坏回退空图。
            # redis_url 配置时整图写入 Redis, 多 worker 共享同一份 (省重复 LLM 抽取 + 图谱一致);
            # 未配置/ Redis 不可达 → GraphStore 自动回退内存/文件, 不阻断启动。
            _evict_oldest(
                _graph_map,
                kb,
                GraphRAGRetriever(
                    entity_ext,
                    rel_ext,
                    GraphStore(
                        persist_path=str(_DATA_DIR / f"graph_{kb}.pkl"),
                        redis_url=settings.redis_url or None,
                        redis_key=f"rag:graph:{kb}",
                    ),
                ),
            )
        else:
            _graph_map.move_to_end(kb)
        return _graph_map[kb]


def get_vector_store(kb: str = "documents") -> "VectorStore":
    with _vector_lock:
        if kb not in _vector_map:
            from engines.retrieval.vector_store import VectorStore

            _evict_oldest(_vector_map, kb, VectorStore(uri=settings.vector_uri, table_name=_vector_table(kb)))
        else:
            _vector_map.move_to_end(kb)
        return _vector_map[kb]


def _rebuild_bm25_bg(kb: str) -> None:
    """后台从向量库(权威数据源)全量重建 BM25 索引, 不阻塞启动/请求。

    用 Bm25Index.rebuild() 原地清空重建, 因此调用方持有的实例引用保持有效,
    不需要替换 _bm25_map 里的对象。
    """
    try:
        import lancedb

        db = lancedb.connect(settings.vector_uri)
        rows = db.open_table(_vector_table(kb)).to_pandas().to_dict("records")
        docs = [
            {
                "id": r.get("id") or "",
                "chunk_id": r.get("id") or "",
                "doc_id": r.get("doc_id") or "",
                "content": r.get("content") or "",
            }
            for r in rows
            if (r.get("content") or "").strip()
        ]
        get_bm25_index(kb).rebuild(docs)
        logger.info("[%s] BM25 后台重建完成: %d 文档", kb, len(docs))
    except Exception as e:  # noqa: BLE001
        logger.error("[%s] BM25 后台重建失败: %s", kb, str(e)[:200])


def _check_bm25_consistency(kb: str, idx: "Bm25Index") -> None:
    """校验 BM25 文档数与向量库行数一致; 不一致则告警并按配置后台自愈。

    背景 (P0): BM25 曾因历史 pickle 索引与现行 JSON 读取不兼容, 每次启动都在
    json.load 抛错后被静默吞掉、**回退空索引**; 此后 ingest 只往空索引追加并覆盖落盘,
    历史累积被反复丢弃。生产因此长期跑在"稀疏检索缺失"状态 —— 混合检索退化为纯向量,
    而日志里只有一条 warning, 极难察觉。这里把它变成显式可观测的失败。
    """
    if not settings.bm25_consistency_check:
        return
    with _bm25_check_lock:
        if kb in _bm25_checked:
            return
        _bm25_checked.add(kb)
    try:
        n_vec = get_vector_store(kb).table.count_rows()
        n_bm25 = len(idx)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] BM25 一致性自检跳过(取行数失败): %s", kb, str(e)[:120])
        return
    # 可观测性(P1): 无论一致与否都上报 Gauge —— 一致=0, 缺口=行数差, 供 /metrics 告警。
    try:
        from api.core.metrics import bm25_index_gap

        bm25_index_gap.labels(kb=kb).set(abs(n_vec - n_bm25))
    except Exception as e:  # noqa: BLE001 — 指标上报失败不影响自检逻辑
        logger.debug("[%s] 一致性 Gauge 上报失败: %s", kb, str(e)[:80])
    if n_vec == n_bm25:
        return
    logger.error(
        "[%s] BM25 索引与向量库不一致: BM25=%d 向量=%d (差 %d)。稀疏检索将不完整, "
        "混合检索退化为纯向量。%s",
        kb,
        n_bm25,
        n_vec,
        abs(n_vec - n_bm25),
        "已触发后台重建。" if settings.bm25_autorebuild else "请运行: python scripts/rebuild_bm25.py",
    )
    if settings.bm25_autorebuild:
        Thread(target=_rebuild_bm25_bg, args=(kb,), daemon=True, name=f"bm25-rebuild-{kb}").start()


def get_bm25_index(kb: str = "documents") -> "Bm25Index":
    with _bm25_lock:
        if kb not in _bm25_map:
            from engines.retrieval.bm25_index import Bm25Index

            # 持久化到 data/bm25_<kb>.json, 重启恢复索引 (与向量库持久化对称)。
            # 后缀曾为 .pkl —— 历史实现用 pickle 落盘, 现行为 JSON; 后缀误导正是当初
            # "pickle 遗留文件被按 JSON 读、静默回退空索引"的埋雷处, 故改为 .json。
            # (graph_<kb>.pkl 仍是真 pickle, 保持 .pkl 不变。)
            _evict_oldest(_bm25_map, kb, Bm25Index(persist_path=str(_DATA_DIR / f"bm25_{kb}.json")))
            idx = _bm25_map[kb]
        else:
            idx = _bm25_map[kb]
            _bm25_map.move_to_end(kb)
    _check_bm25_consistency(kb, idx)
    return idx


_reranker = None
_reranker_lock = Lock()

_embedder = None
_embedder_lock = Lock()


def get_embedder() -> "EmbeddingService":
    """全局 EmbeddingService 单例 — 嵌入模型只加载一次。

    供重排与忠实度护栏复用, 避免每请求/每路径重复建实例导致模型重复加载。
    """
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from engines.embedding.embedder import EmbeddingService

                _embedder = EmbeddingService(
                    model_name=settings.embedding_model,
                    device=settings.embedding_device,
                    backend=settings.embedding_backend,
                    api_base=settings.embedding_api_base,
                    api_key=settings.embedding_api_key,
                    api_model=settings.embedding_api_model,
                    api_dims=settings.embedding_api_dims,
                    api_timeout_s=settings.embedding_api_timeout_s,
                )
    return _embedder


def rerank_effective_enabled() -> bool:
    """重排是否实际生效: 综合显式开关与 GPU 自适应 (P2#11)。

    - 显式 rerank_enabled=True → 开。
    - 显式 rerank_enabled=False, 或用户已显式设定(环境变量) → 关(尊重用户)。
    - 未显式设定且 reranker_auto_gpu=True → 检测到 CUDA 自动开 (切 v2-m3)。
    - 其余 → 关 (CPU 默认)。
    """
    if settings.rerank_enabled:
        return True
    if not settings.reranker_auto_gpu:
        return False
    if "rerank_enabled" in settings.model_fields_set:
        return False  # 用户显式设定, 不覆盖
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 — torch 不可用 → 当作无 GPU
        return False


def get_reranker() -> "Reranker | None":
    """全局 Reranker 单例 — Cross-Encoder 模型只加载一次。

    原 orchestrator 每请求新建, 新实例导致 _ce_model 缓存失效、重复加载模型。
    复用 get_embedder() 单例, 避免与护栏路径重复加载嵌入模型。
    未生效(禁用或自适应未触发)时返回 None, 调用方据此跳过重排;
    GPU 自适应开启时自动切 BAAI/bge-reranker-v2-m3。
    """
    global _reranker
    if not rerank_effective_enabled():
        return None
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from engines.retrieval.reranker import Reranker

                model = settings.reranker_model
                # GPU 自适应: 自动开启且为默认 base 模型时, 升级到 v2-m3 (报告 P2#11)
                if (
                    settings.reranker_auto_gpu
                    and "rerank_enabled" not in settings.model_fields_set
                    and model == "BAAI/bge-reranker-base"
                ):
                    try:
                        import torch

                        if torch.cuda.is_available():
                            model = "BAAI/bge-reranker-v2-m3"
                    except Exception:  # noqa: BLE001, S110 — GPU 探测失败按无 GPU 处理
                        pass
                try:
                    _reranker = Reranker(
                        embedder=get_embedder(),
                        ce_model_name=resolve_model_path(model) if model else None,
                        max_length=settings.reranker_max_length or None,
                        backend=settings.reranker_backend,
                    )
                except Exception as e:  # noqa: BLE001 — 模型加载失败降级为不重排, 不阻断
                    logger.warning("Reranker 初始化失败, 降级为不重排: %s", str(e)[:150])
                    return None
    return _reranker
