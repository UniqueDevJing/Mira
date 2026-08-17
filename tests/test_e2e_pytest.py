"""端到端集成测试 — 上传 → 解析 → 分块 → 嵌入 → 入库 → 检索 全链路。

外部依赖（真实 embedding 模型、LLM、图谱抽取）全部替换为确定性假实现，
磁盘状态（SQLite / LanceDB / BM25 pickle）隔离到 tmp_path，保证可重复、无网络。
验证目标：文档经真实 pipeline 处理后，能被 HybridRetriever 按语义检索命中。
"""

from unittest.mock import MagicMock

import pytest

from api.core.document_store import DocumentStore
from api.routes.documents import _process_document_pipeline
from engines.embedding import embedder as emb_mod
from engines.retrieval.bm25_index import Bm25Index
from engines.retrieval.hybrid_retriever import HybridRetriever
from engines.retrieval.vector_store import VectorStore

_EMB_DIM = 512


class _FakeEmbedder:
    """确定性假嵌入：所有文本映射到同一单位向量，仅用于打通链路。"""

    def __init__(self, model_name: str = "fake", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.batch_size = 32

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * _EMB_DIM for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0] * _EMB_DIM


@pytest.fixture
def e2e_env(tmp_path, monkeypatch):
    # 嵌入模型 -> 假实现（确定性，无需 GPU/网络）
    monkeypatch.setattr(emb_mod, "EmbeddingService", _FakeEmbedder)

    # 文档元数据 SQLite -> tmp，避免污染生产库
    ds = DocumentStore(db_path=str(tmp_path / "docs.db"))
    monkeypatch.setattr("api.core.document_store.get_document_store", lambda: ds)

    # 检索后端 -> 隔离实例（LanceDB 走 tmp；BM25 纯内存；图谱抽取 stub）
    vs = VectorStore(uri=str(tmp_path / "lancedb"), dim=_EMB_DIM, table_name="documents")
    bm = Bm25Index()
    graph_stub = MagicMock()
    graph_stub.build_from_chunks.return_value = {"entities": 0, "relations": 0}
    monkeypatch.setattr("api.state.get_vector_store", lambda kb: vs)
    monkeypatch.setattr("api.state.get_bm25_index", lambda kb: bm)
    monkeypatch.setattr("api.state.get_graph_rag", lambda kb: graph_stub)

    yield {"tmp": tmp_path, "vs": vs, "bm": bm, "ds": ds}


_SAMPLE = "RAG 2.0 的核心模块包括向量检索与图谱检索。混合检索能提升召回率，降低漏答。"


def test_e2e_pipeline_direct_and_retrieve(e2e_env):
    """直接驱动真实 pipeline 函数，断言入库后可被混合检索命中。"""
    result = _process_document_pipeline("doc001", "doc.txt", _SAMPLE.encode("utf-8"), "documents")

    assert result["chunks"] > 0, "短文本至少应产出 1 个 chunk"
    assert e2e_env["vs"].search([1.0] * _EMB_DIM, top_k=5), "向量库应已写入 chunk"

    retriever = HybridRetriever(vector_store=e2e_env["vs"], embedder=_FakeEmbedder())
    out = retriever.retrieve("混合检索", top_k=5)
    contents = [d["content"] for d in out["documents"]]
    assert any("混合检索" in c or "图谱检索" in c for c in contents), f"检索未命中预期内容: {contents}"


def test_e2e_http_upload_then_retrieve(e2e_env):
    """经 FastAPI 路由走完整 HTTP 链路：上传 -> 后台处理 -> 状态就绪 -> 检索命中。"""
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("doc.txt", _SAMPLE.encode("utf-8"), "text/plain")},
        data={"knowledge_base": "documents"},
    )
    assert resp.status_code == 200, resp.text
    doc_id = resp.json()["doc_id"]

    status = client.get(f"/api/v1/documents/{doc_id}/status").json()
    assert status["status"] == "ready", f"后台处理未完成: {status}"
    assert status["chunk_count"] > 0

    retriever = HybridRetriever(vector_store=e2e_env["vs"], embedder=_FakeEmbedder())
    out = retriever.retrieve("图谱检索", top_k=5)
    contents = [d["content"] for d in out["documents"]]
    assert any("图谱检索" in c for c in contents), f"HTTP 链路检索未命中: {contents}"
