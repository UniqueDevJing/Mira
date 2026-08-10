"""pytest 共享 fixture"""
import os
import pytest

# 设置测试环境变量（在导入应用之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.pop('RAG_API_KEY_ENABLED', None)
os.environ.pop('RAG_API_KEY', None)


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
        {"id": "d1", "chunk_id": "c1", "doc_id": "doc1",
         "content": "FastAPI 是一个高性能的 Python Web 框架，基于 Starlette 和 Pydantic。",
         "score": 0.85},
        {"id": "d2", "chunk_id": "c2", "doc_id": "doc1",
         "content": "Django 是一个全栈 Web 框架，提供了 ORM、模板引擎和 admin 后台。",
         "score": 0.65},
        {"id": "d3", "chunk_id": "c3", "doc_id": "doc1",
         "content": "FastAPI 支持异步处理、自动生成 OpenAPI 文档、依赖注入系统。",
         "score": 0.75},
    ]
