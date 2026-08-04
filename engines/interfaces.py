"""引擎接口抽象层 — 面向接口编程，支持实现切换。

使用方式：
    from engines.interfaces import VectorStoreInterface, EmbedderInterface
    store: VectorStoreInterface = LanceDBVectorStore()
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """文档块数据结构"""
    chunk_id: str
    doc_id: str
    content: str
    context: Dict = None
    metadata: Dict = None
    embedding: List[float] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """检索结果"""
    id: str
    chunk_id: str
    doc_id: str
    content: str
    score: float


class VectorStoreInterface(ABC):
    """向量存储接口"""

    @abstractmethod
    def insert(self, chunks: List[Chunk]) -> None:
        """插入文档块"""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filter_expr: Optional[str] = None,
    ) -> List[Dict]:
        """向量检索"""
        pass

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        """按文档 ID 删除"""
        pass


class EmbedderInterface(ABC):
    """文本嵌入接口"""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入"""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """查询嵌入"""
        pass


class GraphStoreInterface(ABC):
    """知识图谱存储接口"""

    @abstractmethod
    def upsert_entity(
        self,
        name: str,
        etype: str,
        chunk_id: str = "",
        aliases: List[str] = None,
    ) -> None:
        """插入或更新实体"""
        pass

    @abstractmethod
    def add_relation(
        self,
        subject: str,
        predicate: str,
        object: str,
        chunk_id: str = "",
    ) -> None:
        """添加关系"""
        pass

    @abstractmethod
    def get_entity(self, name: str) -> Optional[Dict]:
        """获取实体"""
        pass

    @abstractmethod
    def get_relations(
        self,
        subject: str = None,
        predicate: str = None,
        object: str = None,
    ) -> List[Dict]:
        """获取关系"""
        pass

    @abstractmethod
    def multi_hop(
        self,
        start: str,
        relations: List[str],
        max_depth: int = 3,
    ) -> List[Dict]:
        """多跳遍历"""
        pass

    @abstractmethod
    def get_context_for_entity(self, name: str) -> str:
        """获取实体上下文"""
        pass

    @abstractmethod
    def stats(self) -> Dict:
        """图谱统计"""
        pass


class RetrieverInterface(ABC):
    """检索器接口"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20) -> Dict:
        """检索"""
        pass


class RerankerInterface(ABC):
    """重排序接口"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """重排序"""
        pass
