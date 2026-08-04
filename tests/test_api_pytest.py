"""API 集成测试 — 健康检查 / 文档管理 / QA 接口 / 认证"""
import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI 测试客户端"""
    from api.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """健康检查端点测试"""

    def test_health_returns_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"

    def test_root_returns_web_ui(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


class TestDocumentEndpoints:
    """文档管理端点测试"""

    def test_upload_missing_file_returns_422(self, client):
        resp = client.post("/api/v1/documents/upload")
        assert resp.status_code == 422

    def test_list_documents_returns_empty(self, client):
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert isinstance(data["items"], list)

    def test_document_status_not_found(self, client):
        resp = client.get("/api/v1/documents/nonexistent/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_found"


class TestQAEndpoints:
    """QA 接口测试"""

    def test_ask_returns_response(self, client):
        resp = client.post("/api/v1/qa/ask", json={
            "question": "系统用了哪些技术？",
            "mode": "hybrid",
            "top_k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["answer"]  # answer 不为空

    def test_ask_empty_question_returns_422(self, client):
        resp = client.post("/api/v1/qa/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_long_question_returns_422(self, client):
        resp = client.post("/api/v1/qa/ask", json={"question": "x" * 2001})
        assert resp.status_code == 422

    def test_ask_invalid_mode_returns_422(self, client):
        resp = client.post("/api/v1/qa/ask", json={
            "question": "test", "mode": "invalid"
        })
        assert resp.status_code == 422

    def test_ask_top_k_out_of_range_returns_422(self, client):
        resp = client.post("/api/v1/qa/ask", json={
            "question": "test", "top_k": 100
        })
        assert resp.status_code == 422

    def test_ask_response_structure(self, client):
        resp = client.post("/api/v1/qa/ask", json={
            "question": "测试问题",
            "mode": "hybrid",
            "enable_self_retrieval": False,
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        required_fields = ["answer", "sources", "retrieval_rounds", "latency_ms"]
        for field in required_fields:
            assert field in data, f"缺少响应字段: {field}"
        assert isinstance(data["sources"], list)
        assert isinstance(data["latency_ms"], (int, float))


class TestCORS:
    """CORS 配置测试"""

    def test_cors_headers_present(self, client):
        resp = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        cors_headers = {
            k.lower(): v for k, v in resp.headers.items()
            if k.lower().startswith("access-control")
        }
        assert "access-control-allow-origin" in cors_headers


class TestAPIKeyAuth:
    """API Key 认证测试"""

    def test_no_key_when_enabled_returns_401(self, client):
        os.environ["RAG_API_KEY_ENABLED"] = "true"
        os.environ["RAG_API_KEY"] = "test-secret-key"
        try:
            resp = client.post("/api/v1/qa/ask", json={"question": "test"})
            assert resp.status_code == 401
        finally:
            os.environ.pop("RAG_API_KEY_ENABLED", None)
            os.environ.pop("RAG_API_KEY", None)

    def test_wrong_key_returns_403(self, client):
        os.environ["RAG_API_KEY_ENABLED"] = "true"
        os.environ["RAG_API_KEY"] = "test-secret-key"
        try:
            resp = client.post(
                "/api/v1/qa/ask",
                json={"question": "test"},
                headers={"X-API-Key": "wrong"},
            )
            assert resp.status_code == 403
        finally:
            os.environ.pop("RAG_API_KEY_ENABLED", None)
            os.environ.pop("RAG_API_KEY", None)

    def test_correct_key_passes(self, client):
        os.environ["RAG_API_KEY_ENABLED"] = "true"
        os.environ["RAG_API_KEY"] = "test-secret-key"
        try:
            resp = client.post(
                "/api/v1/qa/ask",
                json={"question": "test"},
                headers={"X-API-Key": "test-secret-key"},
            )
            assert resp.status_code == 200
        finally:
            os.environ.pop("RAG_API_KEY_ENABLED", None)
            os.environ.pop("RAG_API_KEY", None)

    def test_bearer_auth_passes(self, client):
        os.environ["RAG_API_KEY_ENABLED"] = "true"
        os.environ["RAG_API_KEY"] = "test-secret-key"
        try:
            resp = client.post(
                "/api/v1/qa/ask",
                json={"question": "test"},
                headers={"Authorization": "Bearer test-secret-key"},
            )
            assert resp.status_code == 200
        finally:
            os.environ.pop("RAG_API_KEY_ENABLED", None)
            os.environ.pop("RAG_API_KEY", None)

    def test_health_exempt_from_auth(self, client):
        os.environ["RAG_API_KEY_ENABLED"] = "true"
        os.environ["RAG_API_KEY"] = "test-secret-key"
        try:
            resp = client.get("/health")
            assert resp.status_code == 200
        finally:
            os.environ.pop("RAG_API_KEY_ENABLED", None)
            os.environ.pop("RAG_API_KEY", None)
