"""上传路由: 扩展名校验 + 空文档状态"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.documents import _process_document_pipeline

client = TestClient(app)


def test_upload_rejects_unknown_extension():
    r = client.post("/api/v1/documents/upload", files={"file": ("a.doc", b"content", "application/msword")})
    assert r.status_code == 400


@pytest.mark.slow
def test_upload_accepts_markdown():
    r = client.post("/api/v1/documents/upload", files={"file": ("a.md", "# 标题\n\n正文".encode(), "text/markdown")})
    assert r.status_code == 200
    assert r.json()["status"] == "processing"


def test_upload_rejects_uppercase_unknown():
    # 大写扩展名且扩展名本身不受支持 (.FOO -> .foo 不在 SUPPORTED_EXTENSIONS), 校验应在扩展名归一后拒绝 (400)。
    # 注意: .XLSX 是大写但 xlsx 本身受支持, 不能用来测"未知" — 这里用确定未知的 .FOO。
    r = client.post("/api/v1/documents/upload", files={"file": ("a.FOO", b"x", "application/octet-stream")})
    assert r.status_code == 400


def test_pipeline_empty_doc_returns_zero_chunks():
    """直接测 pipeline: 空 md 早退返回 0 chunk, 不触发 embedding/模型"""
    result = _process_document_pipeline("d1", "empty.md", b"")
    assert result == {"pages": 1, "chunks": 0}


def test_pipeline_unknown_ext_raises():
    with pytest.raises(ValueError):
        _process_document_pipeline("d1", "a.doc", b"x")


def test_empty_doc_status_becomes_empty():
    r = client.post("/api/v1/documents/upload", files={"file": ("empty.md", b"", "text/markdown")})
    assert r.status_code == 200
    doc_id = r.json()["doc_id"]
    status = "processing"
    for _ in range(10):
        status = client.get(f"/api/v1/documents/{doc_id}/status").json()["status"]
        if status != "processing":
            break
    assert status == "empty"
