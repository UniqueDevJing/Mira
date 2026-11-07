"""图片产物路由（W3.5 契约补齐）。

`sources` 事件里的 `thumb_url` 是 `/api/v1/images/{sha256}`。这个路由必须真的存在，
否则前端拿到的是一个 404——**契约里写了 URL 却不实现，比不写更糟**。

存储位置：默认指向视觉能力网关的产物库（内容寻址：`{root}/{sha[:2]}/{sha}.png`）。
用 `MIRA_IMAGE_DIR` 指定，例如网关的 `GV_STORE_DIR`。
未配置时明确返回 404 + `image-store-not-configured`，而不是假装成功。
"""

from __future__ import annotations

import os
import pathlib
import re

from fastapi import APIRouter, HTTPException, Response

router = APIRouter(prefix="/api/v1/images", tags=["images"])

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def _store_root() -> pathlib.Path | None:
    raw = os.environ.get("MIRA_IMAGE_DIR", "").strip()
    return pathlib.Path(raw) if raw else None


@router.get("/{asset_id}")
def get_image(asset_id: str) -> Response:
    if not _SHA256.match(asset_id.lower()):
        raise HTTPException(status_code=400, detail="bad-asset-id")
    root = _store_root()
    if root is None:
        raise HTTPException(status_code=404, detail="image-store-not-configured")
    path = root / asset_id[:2] / f"{asset_id}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="asset-not-found")
    return Response(content=path.read_bytes(), media_type="image/png", headers=_IMMUTABLE)