"""在线链路图片 sources 测试（W3.5）。

锁三件事：
1. 图片项必须带 `type=image`、`thumb_url` 与**溯源**（source/page）；
2. 文本 sources 顺序与内容不受影响（图片追加在后，不插队）；
3. 环境变量未配置/索引损坏时**静默退化为空**，绝不让流式问答失败。
"""

from __future__ import annotations

import json
import struct
import zlib

from api.core.image_context import image_sources, image_sources_from_file
from api.core.response import build_image_sources, build_sources_event, merge_image_sources


def make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
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


def build_index(tmp_path):
    from engines.multimodal import ImageIndex

    index = ImageIndex()
    index.add(
        make_png(8, 8, (200, 30, 30)),
        kb="policy",
        caption="高血压分级示意图 收缩压 舒张压",
        source="内科学 第9版",
        page=112,
    )
    index.add(
        make_png(8, 8, (30, 30, 200)),
        kb="policy",
        caption="退款流程图 客服时效",
        source="售后政策.pdf",
        page=7,
    )
    path = tmp_path / "image-index.json"
    index.save(path)
    return path


TEXT_DOCS = [{"id": "d1", "chunk_id": "c1", "title": "高血压", "score": 0.9, "context_full": "x"}]


# ───────────── 构造与合并 ─────────────

def test_build_image_sources_carries_provenance():
    hits = [{
        "asset_id": "a" * 64, "caption": "高血压示意图", "score": 0.8,
        "source": "内科学 第9版", "page": 112, "width": 8, "height": 8,
    }]
    items = build_image_sources(hits)
    assert items[0]["type"] == "image"
    assert items[0]["thumb_url"] == "/api/v1/images/" + "a" * 64
    assert items[0]["source"] == "内科学 第9版"
    assert items[0]["page"] == 112


def test_build_image_sources_skips_missing_asset_id():
    assert build_image_sources([{"caption": "无 id"}]) == []


def test_merge_keeps_text_order_and_appends_images():
    text = [{"id": "d1"}, {"id": "d2"}]
    images = [{"type": "image", "asset_id": "a1"}, {"type": "image", "asset_id": "a2"}]
    merged = merge_image_sources(text, images)
    assert [m.get("id") for m in merged[:2]] == ["d1", "d2"]
    assert [m["asset_id"] for m in merged[2:]] == ["a1", "a2"]


def test_merge_dedupes_and_respects_limit():
    text = [{"type": "image", "asset_id": "a1"}]
    images = [{"type": "image", "asset_id": "a1"}, {"type": "image", "asset_id": "a2"},
              {"type": "image", "asset_id": "a3"}]
    merged = merge_image_sources(text, images, limit=1)
    assert [m["asset_id"] for m in merged] == ["a1", "a2"]


# ───────────── sources 事件 ─────────────

def test_sources_event_without_images_unchanged():
    ev = build_sources_event(TEXT_DOCS, 0.9, None, 0)
    assert ev["image_count"] == 0
    assert len(ev["sources"]) == 1
    assert ev["sources"][0]["id"] == "d1"
    assert "context_full" not in ev["sources"][0]


def test_sources_event_appends_images_with_type():
    hits = [{
        "asset_id": "b" * 64, "caption": "退款流程图", "score": 0.7,
        "source": "售后政策.pdf", "page": 7, "width": 8, "height": 8,
    }]
    ev = build_sources_event(TEXT_DOCS, 0.9, None, 0, image_hits=hits)
    assert ev["image_count"] == 1
    assert ev["sources"][0]["id"] == "d1", "文本来源必须仍在最前"
    image_item = ev["sources"][-1]
    assert image_item["type"] == "image"
    assert image_item["source"] == "售后政策.pdf"


# ───────────── 在线链路（环境变量开关） ─────────────

def test_image_sources_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MIRA_IMAGE_INDEX", raising=False)
    assert image_sources("高血压") == []


def test_image_sources_reads_configured_index(tmp_path, monkeypatch):
    path = build_index(tmp_path)
    monkeypatch.setenv("MIRA_IMAGE_INDEX", str(path))
    hits = image_sources("高血压 示意图")
    assert hits and "高血压" in hits[0]["caption"]
    assert hits[0]["source"] == "内科学 第9版"


def test_image_sources_survives_broken_index(tmp_path, monkeypatch):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("MIRA_IMAGE_INDEX", str(broken))
    assert image_sources("高血压") == [], "索引损坏必须退化为没有图，而不是抛异常"


def test_image_sources_survives_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRA_IMAGE_INDEX", str(tmp_path / "nope.json"))
    assert image_sources("高血压") == []


def test_image_sources_kb_filter(tmp_path):
    path = build_index(tmp_path)
    assert image_sources_from_file(str(path), "退款流程", kb="finance") == []
    assert image_sources_from_file(str(path), "退款流程", kb="policy")


def test_image_sources_empty_question(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRA_IMAGE_INDEX", str(build_index(tmp_path)))
    assert image_sources("") == []


def test_image_sources_json_serializable(tmp_path):
    path = build_index(tmp_path)
    hits = image_sources_from_file(str(path), "高血压")
    json.dumps(hits)  # 不抛即通过：SSE 要直接序列化
    assert {"asset_id", "caption", "score", "source", "page"} <= set(hits[0])