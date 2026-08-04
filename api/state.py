"""API 全局状态 — 跨请求持久化的单例（线程安全）"""
from threading import Lock
from engines.graph_rag.entity_extractor import EntityExtractor, RelationExtractor
from engines.graph_rag.graph_store import GraphStore
from engines.graph_rag.graph_retriever import GraphRAGRetriever

_graph_rag = None
_vector_store = None
_graph_rag_lock = Lock()
_vector_store_lock = Lock()


def get_graph_rag() -> GraphRAGRetriever:
    global _graph_rag
    if _graph_rag is None:
        with _graph_rag_lock:
            if _graph_rag is None:
                from api.config import settings
                entity_ext = EntityExtractor(
                    llm_url=settings.llm_base_url,
                    llm_model=settings.llm_model,
                    llm_key=settings.llm_api_key,
                )
                rel_ext = RelationExtractor(
                    llm_url=settings.llm_base_url,
                    llm_model=settings.llm_model,
                    llm_key=settings.llm_api_key,
                )
                graph = GraphStore()
                _graph_rag = GraphRAGRetriever(entity_ext, rel_ext, graph)
    return _graph_rag


def get_vector_store() -> "VectorStore":
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                from engines.retrieval.vector_store import VectorStore
                _vector_store = VectorStore()
    return _vector_store
