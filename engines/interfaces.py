"""引擎接口抽象层 — 面向接口编程，支持实现切换。

使用方式：
    from engines.interfaces import VectorStoreInterface, EmbedderInterface
    store: VectorStoreInterface = LanceDBVectorStore()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    """文档块数据结构"""

    chunk_id: str
    doc_id: str
    content: str
    context: dict = None
    metadata: dict = None
    embedding: list[float] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.metadata is None:
            self.metadata = {}


class VectorStoreInterface(ABC):
    """向量存储接口"""

    @abstractmethod
    def insert(self, chunks: list[Chunk], dedup: bool = True) -> None:
        """插入文档块。dedup=True 时先删同 doc_id 旧向量再写入 (防重复累积)。"""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """向量检索"""

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        """按文档 ID 删除"""


class EmbedderInterface(ABC):
    """文本嵌入接口"""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入"""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """查询嵌入"""


class GraphStoreInterface(ABC):
    """知识图谱存储接口"""

    @abstractmethod
    def upsert_entity(
        self,
        name: str,
        etype: str,
        chunk_id: str = "",
        aliases: list[str] | None = None,
    ) -> None:
        """插入或更新实体"""

    @abstractmethod
    def add_relation(
        self,
        subject: str,
        predicate: str,
        object: str,
        chunk_id: str = "",
    ) -> None:
        """添加关系"""

    @abstractmethod
    def get_entity(self, name: str) -> dict | None:
        """获取实体"""

    @abstractmethod
    def get_relations(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[dict]:
        """获取关系"""

    @abstractmethod
    def multi_hop(
        self,
        start: str,
        relations: list[str],
        max_depth: int = 3,
    ) -> list[dict]:
        """多跳遍历"""

    @abstractmethod
    def get_context_for_entity(self, name: str) -> str:
        """获取实体上下文"""

    @abstractmethod
    def stats(self) -> dict:
        """图谱统计"""


class RetrieverInterface(ABC):
    """检索器接口"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20) -> dict:
        """检索"""


class RerankerInterface(ABC):
    """重排序接口"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """重排序"""
