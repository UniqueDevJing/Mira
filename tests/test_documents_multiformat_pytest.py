"""上传路由: 扩展名校验 + 空文档状态"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_upload_rejects_unknown_extension():
    r = client.post("/api/v1/documents/upload",
                    files={"file": ("a.doc", b"content", "application/msword")})
    assert r.status_code == 400


def test_upload_accepts_markdown():
    r = client.post("/api/v1/documents/upload",
                    files={"file": ("a.md", "# 标题\n\n正文".encode("utf-8"), "text/markdown")})
    assert r.status_code == 200
    assert r.json()["status"] == "processing"


def test_upload_rejects_uppercase_unknown():
    r = client.post("/api/v1/documents/upload",
                    files={"file": ("a.XLSX", b"x", "application/octet-stream")})
    assert r.status_code == 400
