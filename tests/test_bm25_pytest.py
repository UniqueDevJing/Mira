"""BM25 稀疏检索单元测试。"""

from engines.retrieval.bm25_index import Bm25Index

SERVICE_DOCS = [
    {
        "id": "c1",
        "chunk_id": "c1",
        "doc_id": "d1",
        "content": "退款流程：买家申请退款后，商家需在 48 小时内处理，逾期系统自动退款。",
    },
    {"id": "c2", "chunk_id": "c2", "doc_id": "d1", "content": "退货物流：商品寄回指定仓库，签收后 3 天完成退款。"},
]
TECH_DOCS = [
    {
        "id": "c3",
        "chunk_id": "c3",
        "doc_id": "d2",
        "content": "本系统使用 FastAPI 框架提供 API 接口，数据库采用 PostgreSQL，部署在 Docker 容器。",
    },
    {"id": "c4", "chunk_id": "c4", "doc_id": "d2", "content": "配置环境变量 RAG_LLM_API_KEY 后即可调用大模型接口。"},
]


def _index():
    idx = Bm25Index()
    idx.add_documents(SERVICE_DOCS + TECH_DOCS)
    return idx


def test_tech_query_ranks_tech_first():
    idx = _index()
    r = idx.search("FastAPI 部署 数据库", top_k=4)
    assert r and r[0]["chunk_id"] == "c3"
    assert all(0.0 <= d["score"] <= 1.0 for d in r)


def test_service_query_ranks_service_first():
    idx = _index()
    r = idx.search("退款流程", top_k=4)
    assert r and r[0]["chunk_id"] == "c1"


def test_empty_index():
    assert Bm25Index().search("任何查询") == []
    assert len(Bm25Index()) == 0


def test_topk_limit():
    idx = _index()
    r = idx.search("退款", top_k=1)
    assert len(r) == 1


def test_incremental_add():
    idx = Bm25Index()
    idx.add_documents(SERVICE_DOCS)
    before = len(idx)
    idx.add_documents(TECH_DOCS)
    assert len(idx) == before + len(TECH_DOCS)
    r = idx.search("FastAPI 部署", top_k=4)
    assert r[0]["chunk_id"] == "c3"


def test_remove_doc_cleans_index():
    """删除文档后索引条目移除, 检索不再返回其内容 (防检索残留)。"""
    idx = Bm25Index()
    idx.add_documents(SERVICE_DOCS)  # 2 条, 均属 d1
    idx.add_documents(TECH_DOCS)  # 2 条, 均属 d2
    n = idx.remove_doc("d1")
    assert n == 2
    assert len(idx) == 2
    # d1 已删, 查"退款"不应返回任何结果
    r = idx.search("退款", top_k=4)
    assert all(doc["doc_id"] != "d1" for doc in r)
    # 其余文档统计仍准确 (d2 内容可检索)
    r2 = idx.search("FastAPI 部署", top_k=4)
    assert r2[0]["chunk_id"] == "c3"
