"""缩略图路由测试（W3.5）。

这不是"顺手补个接口"：sources 契约里已经写了 `/api/v1/images/{sha256}`，
路由缺失会让前端静默裂图。这里锁四种行为：正常返回、非法 id、未配置、文件缺失。
"""

from __future__ import annotations

import hashlib
import struct
import zlib

from fastapi.testclient import TestClient

from api.main import app

SHA = "ab" * 32


def make_png(width: int = 4, height: int = 4, rgb: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _store(tmp_path, data: bytes):
    asset_id = hashlib.sha256(data).hexdigest()
    target = tmp_path / asset_id[:2] / f"{asset_id}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return tmp_path, asset_id


def test_serves_image_with_immutable_cache(tmp_path, monkeypatch):
    data = make_png()
    root, asset_id = _store(tmp_path, data)
    monkeypatch.setenv("MIRA_IMAGE_DIR", str(root))

    resp = TestClient(app).get(f"/api/v1/images/{asset_id}")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["cache-control"].startswith("public")
    assert resp.content == data


def test_rejects_bad_asset_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRA_IMAGE_DIR", str(tmp_path))
    client = TestClient(app)

    assert client.get("/api/v1/images/not-a-hash").status_code == 400
    assert client.get("/api/v1/images/abc").status_code == 400


def test_unconfigured_store_returns_explicit_404(tmp_path, monkeypatch):
    monkeypatch.delenv("MIRA_IMAGE_DIR", raising=False)

    resp = TestClient(app).get(f"/api/v1/images/{SHA}")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "image-store-not-configured"


def test_missing_asset_returns_404(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRA_IMAGE_DIR", str(tmp_path))

    resp = TestClient(app).get(f"/api/v1/images/{SHA}")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "asset-not-found"