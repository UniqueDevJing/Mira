"""API 全局状态 — 按知识库隔离的单例（线程安全）。

每个知识库(kb)持有独立的 VectorStore / GraphStore / BM25 索引，防止跨库污染。
- get_vector_store(kb)      → LanceDB 表按 kb 隔离 (默认表 documents 向后兼容)
- get_graph_rag(kb)         → 图谱按 kb 隔离 (实体/关系抽取器共享, 图本身隔离)
- get_bm25_index(kb)        → BM25 稀疏索引按 kb 隔离
"""

import logging
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

# BM25 持久化目录 (基于项目根, 与 document_store 一致)
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

from engines.graph_rag.entity_extractor import EntityExtractor, RelationExtractor
from engines.graph_rag.graph_retriever import GraphRAGRetriever
from engines.graph_rag.graph_store import GraphStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from engines.embedding.embedder import EmbeddingService
    from engines.retrieval.bm25_index import Bm25Index
    from engines.retrieval.reranker import Reranker
    from engines.retrieval.vector_store import VectorStore

_entity_ext = None
_rel_ext = None
_ext_lock = Lock()

_vector_map: dict[str, "VectorStore"] = {}
_graph_map: dict[str, "GraphRAGRetriever"] = {}
_bm25_map: dict[str, "Bm25Index"] = {}
_vector_lock = Lock()
_graph_lock = Lock()
_bm25_lock = Lock()

# 默认表名兼容旧数据；新库统一 rag_<kb> 前缀
LEGACY_KBS = {"", "documents"}

# 每类型 kb 单例上限 — 防任意新 kb 名无限创建实例 (向量表/图谱/BM25 内存与磁盘泄漏)
_MAX_KB_INSTANCES = 32


def _evict_oldest(mapping: dict, key: str, value) -> None:
    """插入新实例并逐出最旧 (dict 保持插入序)。"""
    mapping[key] = value
    if len(mapping) > _MAX_KB_INSTANCES:
        mapping.pop(next(iter(mapping)))


def _vector_table(kb: str) -> str:
    return "documents" if kb in LEGACY_KBS else f"rag_{kb}"


def _shared_extractors():
    """实体/关系抽取器跨库共享（避免重复建 LLM 客户端），图数据本身按库隔离。"""
    global _entity_ext, _rel_ext
    if _entity_ext is None:
        with _ext_lock:
            if _entity_ext is None:
                from api.config import settings

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
    if kb not in _graph_map:
        with _graph_lock:
            if kb not in _graph_map:
                entity_ext, rel_ext = _shared_extractors()
                # 图谱持久化到 data/graph_<kb>.pkl, 重启恢复 (与 BM25 对称); GraphStore 内部损坏回退空图
                _evict_oldest(
                    _graph_map,
                    kb,
                    GraphRAGRetriever(entity_ext, rel_ext, GraphStore(persist_path=str(_DATA_DIR / f"graph_{kb}.pkl"))),
                )
    return _graph_map[kb]


def get_vector_store(kb: str = "documents") -> "VectorStore":
    if kb not in _vector_map:
        with _vector_lock:
            if kb not in _vector_map:
                from api.config import settings
                from engines.retrieval.vector_store import VectorStore

                _evict_oldest(_vector_map, kb, VectorStore(uri=settings.vector_uri, table_name=_vector_table(kb)))
    return _vector_map[kb]


def get_bm25_index(kb: str = "documents") -> "Bm25Index":
    if kb not in _bm25_map:
        with _bm25_lock:
            if kb not in _bm25_map:
                from engines.retrieval.bm25_index import Bm25Index

                # 持久化到 data/bm25_<kb>.pkl, 重启恢复索引 (与向量库持久化对称)
                _evict_oldest(_bm25_map, kb, Bm25Index(persist_path=str(_DATA_DIR / f"bm25_{kb}.pkl")))
    return _bm25_map[kb]


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
                from api.config import settings
                from engines.embedding.embedder import EmbeddingService

                _embedder = EmbeddingService(model_name=settings.embedding_model, device=settings.embedding_device)
    return _embedder


def get_reranker() -> "Reranker":
    """全局 Reranker 单例 — Cross-Encoder 模型只加载一次。

    原 orchestrator 每请求新建, 新实例导致 _ce_model 缓存失效、重复加载模型。
    复用 get_embedder() 单例, 避免与护栏路径重复加载嵌入模型。
    """
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from api.config import settings
                from engines.retrieval.reranker import Reranker

                _reranker = Reranker(
                    embedder=get_embedder(),
                    ce_model_name=settings.reranker_model or None,
                )
    return _reranker
