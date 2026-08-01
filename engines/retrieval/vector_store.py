"""Milvus 向量存储 — 使用 Milvus Lite（嵌入式，无需 Docker）"""
import time
from typing import List
from pymilvus import MilvusClient, DataType


class VectorStore:
    def __init__(self, uri: str = "./milvus.db"):
        """uri 为本地文件路径时使用 Milvus Lite；http://host:port 时连接远程 Milvus"""
        self.client = MilvusClient(uri=uri)
        self.collection_name = "documents"
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=32)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)
        schema.add_field("created_at", DataType.INT64)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="IVF_FLAT",
                               metric_type="COSINE", params={"nlist": 128})
        self.client.create_collection(collection_name=self.collection_name,
                                      schema=schema, index_params=index_params)
        self.client.load_collection(self.collection_name)

    def insert(self, chunks: list):
        data = [{"id": c.chunk_id, "doc_id": c.doc_id, "content": c.content[:65535],
                 "embedding": c.embedding, "created_at": int(time.time())} for c in chunks]
        self.client.insert(self.collection_name, data)
        self.client.load_collection(self.collection_name)

    def search(self, query_embedding: List[float], top_k: int = 20,
               filter_expr: str = None) -> List[dict]:
        try:
            results = self.client.search(collection_name=self.collection_name,
                                         data=[query_embedding], limit=top_k,
                                         filter=filter_expr,
                                         output_fields=["doc_id", "content"])
            if not results or not results[0]:
                return []
            # MilvusClient.search returns [[SearchResult, ...]]
            docs = []
            for r in results[0]:
                if hasattr(r, 'entity'):
                    docs.append({
                        "id": r.get("id", ""),
                        "chunk_id": r.get("id", ""),
                        "doc_id": r.entity.get("doc_id", ""),
                        "content": r.entity.get("content", ""),
                        "score": float(getattr(r, 'distance', 0)),
                    })
                else:
                    docs.append({
                        "id": r.get("id", r.get("chunk_id", "")),
                        "chunk_id": r.get("id", r.get("chunk_id", "")),
                        "doc_id": r.get("doc_id", ""),
                        "content": r.get("content", ""),
                        "score": float(r.get("distance", r.get("score", 0))),
                    })
            return docs
        except Exception as e:
            print(f"Search error: {e}")
            return []
