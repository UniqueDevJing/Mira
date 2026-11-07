"""在线链路接入图片检索（W3.5）。

设计要点：
- **默认关闭**：只有设置 `MIRA_IMAGE_INDEX` 指向已入库的索引文件时才生效，
  线上行为零变化，CI 也无需准备图片；
- **绝不打断流式**：任何异常（文件缺失/损坏/编码器问题）都退化成"没有图"，不抛给调用方；
- 索引按路径缓存，避免每次请求都读 JSON。
"""

from __future__ import annotations

import json
import os
import pathlib
from functools import lru_cache

DEFAULT_LIMIT = 3


@lru_cache(maxsize=4)
def _load_index(path: str, mtime: float):
    from engines.multimodal import ImageIndex

    return ImageIndex.load(path)


def _index_for(path: str):
    try:
        mtime = pathlib.Path(path).stat().st_mtime
    except OSError:
        return None
    return _load_index(path, mtime)


def image_sources(question: str, kb: str | None = None, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """返回可并入 sources 的图片命中（可能为空列表）。"""
    path = os.environ.get("MIRA_IMAGE_INDEX", "").strip()
    if not path or not question:
        return []
    try:
        index = _index_for(path)
        if index is None or len(index) == 0:
            return []
        from engines.multimodal import ImageRetriever, recall_at_k  # noqa: F401  (recall_at_k 供评测复用)

        hits = ImageRetriever(index).search(question, k=limit, kb=kb)
        return [h.to_json() for h in hits]
    except Exception:  # noqa: BLE001 - 图片是增强项，任何异常都不能打断问答
        return []


def image_sources_from_file(index_path: str, question: str, kb: str | None = None,
                            limit: int = DEFAULT_LIMIT) -> list[dict]:
    """显式路径版本（脚本/测试用，不受环境变量影响）。"""
    if not question:
        return []
    try:
        index = _index_for(index_path)
        if index is None or len(index) == 0:
            return []
        from engines.multimodal import ImageRetriever
        return [h.to_json() for h in ImageRetriever(index).search(question, k=limit, kb=kb)]
    except (OSError, json.JSONDecodeError, ValueError, ImportError):
        return []