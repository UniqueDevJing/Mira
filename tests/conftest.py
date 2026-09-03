"""pytest 共享 fixture"""

import os
import tempfile

import pytest

# 设置测试环境变量（在导入应用之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# 向量库指向临时目录, 防测试建表污染生产 lancedb_data/
os.environ.setdefault("RAG_VECTOR_URI", tempfile.mkdtemp(prefix="rag_test_lancedb_"))
# 认证默认关 (与 .env 解耦): 认证测试自行设 RAG_API_KEY_ENABLED=true 并在 finally 复位
os.environ["RAG_API_KEY_ENABLED"] = "false"
os.environ.pop("RAG_API_KEY", None)
# QA 结果缓存默认关 — 缓存会让"同问题多次请求期望不同行为"的测试串扰 (如 LLM 失败兜底)
# 缓存行为由 tests/test_qa_cache_pytest.py 显式开启验证
os.environ["RAG_QA_CACHE_ENABLED"] = "false"
# CORS 默认允许前端本地 origin, 与 tests/test_api_pytest.py::TestCORS 的 localhost:3000 preflight 期望一致。
# 项目 .env 可能配生产 origin (如 https://uniquejing.top) 覆盖默认值, setdefault 保证测试会话走 dev origin
# (env 变量优先级高于 .env, 且本文件在 app 导入前设置, 因此 CORSMiddleware 构造时读到的是 dev origin)。
os.environ.setdefault("RAG_CORS_ORIGINS", '["http://localhost:3000"]')


@pytest.fixture(scope="session", autouse=True)
def _isolate_storage(tmp_path_factory):
    """测试存储隔离 — 防污染生产 data/documents.db 与生产向量/图谱单例。

    上传/文档路由测试走全局 DocumentStore 单例, 指向真实 data/documents.db;
    session 级把路径改到 tmp 并重置单例, 测试不写生产库。
    """
    from api import state
    from api.core import document_store as ds

    test_data_dir = tmp_path_factory.mktemp("test_data")
    ds.DEFAULT_DB_PATH = str(test_data_dir / "documents.db")
    ds._document_store = None  # 重置单例, 下次创建指向 tmp
    # BM25 持久化也指 tmp (防写生产 data/bm25_*.pkl)
    state._DATA_DIR = test_data_dir
    # 清理生产状态单例, 避免测试间复用已连生产 lancedb_data 的实例
    state._vector_map.clear()
    state._graph_map.clear()
    state._bm25_map.clear()
    state._reset_mounted_kbs()  # 已挂载 KB 探测缓存按 vector_uri 隔离, 切目录后必须失效
    yield


@pytest.fixture(autouse=True)
def _reset_state_per_test(tmp_path_factory):
    """每个测试独立存储, 防止跨测试 KB/BM25/文档库状态泄漏。

    例如某测试上传文档写入共享 LanceDB, 若不清空, 后续测试检索会命中
    该残留文档; 置信度护栏因此误拒答, 造成测试顺序相关的偶发失败。
    通过每测试分配全新 vector_uri 与数据目录 + 清空单例 map 彻底隔离。
    """
    from api import state
    from api.config import settings
    from api.core import document_store as ds

    fresh_vec = tmp_path_factory.mktemp("vec")
    settings.vector_uri = str(fresh_vec)
    fresh_data = tmp_path_factory.mktemp("data")
    state._DATA_DIR = fresh_data
    ds.DEFAULT_DB_PATH = str(fresh_data / "documents.db")
    ds._document_store = None
    state._vector_map.clear()
    state._graph_map.clear()
    state._bm25_map.clear()
    state._reset_mounted_kbs()  # 每测试全新 vector_uri, 失效上一测试的挂载探测缓存
    # 认证/密钥状态每测试复位: 前置认证测试 (test_api/test_rbac) 用 os.environ 或
    # monkeypatch 临时开启 RAG_API_KEY_ENABLED/RAG_API_KEY, 若泄漏到后续测试,
    # 会让未带 Key 的 TestClient 命中 "启用+无 Key -> 401", 造成顺序相关的假性失败。
    # 清掉后回落 settings.api_key_enabled(默认 False), 每个测试都从"认证关闭"基线开始。
    os.environ.pop("RAG_API_KEY_ENABLED", None)
    os.environ.pop("RAG_API_KEY", None)
    yield


@pytest.fixture(scope="session")
def embedder():
    """Embedding 服务单例（整个测试会话共享）"""
    from engines.embedding.embedder import EmbeddingService

    return EmbeddingService()


@pytest.fixture(scope="session")
def vector_store():
    """向量存储（使用临时目录）"""
    import tempfile

    from engines.retrieval.vector_store import VectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(uri=tmpdir)
        yield store


@pytest.fixture(scope="session")
def sample_pdf_path():
    """测试 PDF 路径"""
    path = os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")
    if not os.path.exists(path):
        pytest.skip("测试 PDF 不存在: tests/fixtures/sample.pdf")
    return path


@pytest.fixture
def mock_llm_response():
    """模拟 LLM 响应"""
    from api.core.llm_client import LLMResponse

    return LLMResponse(
        content="这是一个测试回答。",
        reasoning_content="",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=200.0,
    )


@pytest.fixture
def sample_chunks():
    """示例文档块"""
    from engines.chunking.structure_chunker import Chunk

    return [
        Chunk(
            chunk_id="test_chunk_0001",
            doc_id="test_doc",
            content="FastAPI 是一个高性能的 Python Web 框架，基于 Starlette 和 Pydantic。",
            context={"title_chain": ["技术栈"]},
            metadata={"page_range": [1, 1], "char_count": 30},
        ),
        Chunk(
            chunk_id="test_chunk_0002",
            doc_id="test_doc",
            content="LanceDB 是一个嵌入式向量数据库，支持零锁冲突和持久化存储。",
            context={"title_chain": ["存储层"]},
            metadata={"page_range": [2, 2], "char_count": 28},
        ),
    ]


@pytest.fixture
def sample_documents():
    """示例检索文档"""
    return [
        {
            "id": "d1",
            "chunk_id": "c1",
            "doc_id": "doc1",
            "content": "FastAPI 是一个高性能的 Python Web 框架，基于 Starlette 和 Pydantic。",
            "score": 0.85,
        },
        {
            "id": "d2",
            "chunk_id": "c2",
            "doc_id": "doc1",
            "content": "Django 是一个全栈 Web 框架，提供了 ORM、模板引擎和 admin 后台。",
            "score": 0.65,
        },
        {
            "id": "d3",
            "chunk_id": "c3",
            "doc_id": "doc1",
            "content": "FastAPI 支持异步处理、自动生成 OpenAPI 文档、依赖注入系统。",
            "score": 0.75,
        },
    ]
